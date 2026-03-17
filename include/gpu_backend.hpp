// gpu_backend.hpp — Tier 2 OpenCL GPU backend for DendriteNet
//
// Included at the tail of tensor.hpp OUTSIDE namespace dendrite when
// -DDENDRITE_OPENCL is defined. Do NOT #include this file directly.
//
// Build:  g++ ... -DDENDRITE_OPENCL -lOpenCL ...

#pragma once

#ifdef DENDRITE_OPENCL

// System headers included OUTSIDE any namespace to avoid poisoning std::
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// GPU dispatch threshold (also defined in tensor.hpp; #ifndef prevents redefinition)
#ifndef DENDRITE_GPU_MIN_ELEMS
#define DENDRITE_GPU_MIN_ELEMS 65536
#endif

// ============================================================
// OpenCL kernel source strings (file scope, outside namespace)
// ============================================================

// Matrix-vector multiply: y[i] = sum_j(W[i*cols+j]*x[j]) + bias[i]
static const char* DENDRITE_MATVEC_KERNEL = R"CL(
__kernel void matvec_k(
    __global const float* W,
    __global const float* x,
    __global const float* bias,
    __global       float* y,
    int rows, int cols)
{
    int i = get_global_id(0);
    if (i >= rows) return;
    float sum = bias[i];
    for (int j = 0; j < cols; j++) sum += W[i * cols + j] * x[j];
    y[i] = sum;
}
)CL";

// Transposed matrix-vector multiply: y[j] = sum_i(W[i*in_dim+j]*delta[i])
static const char* DENDRITE_MATVEC_TRANS_KERNEL = R"CL(
__kernel void matvec_trans_k(
    __global const float* W,
    __global const float* delta,
    __global       float* y,
    int out_dim, int in_dim)
{
    int j = get_global_id(0);
    if (j >= in_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < out_dim; i++) sum += W[i * in_dim + j] * delta[i];
    y[j] = sum;
}
)CL";

// Outer product: out[i*n+j] = a[i]*b[j]
static const char* DENDRITE_OUTER_KERNEL = R"CL(
__kernel void outer_k(
    __global const float* a,
    __global const float* b,
    __global       float* out,
    int m, int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= m || j >= n) return;
    out[i * n + j] = a[i] * b[j];
}
)CL";

// Fused Adam update with gradient clipping
static const char* DENDRITE_ADAM_KERNEL = R"CL(
__kernel void adam_update_k(
    __global       float* weights,
    __global const float* grad,
    __global       float* m,
    __global       float* v,
    float lr, float bc1, float bc2,
    float beta1, float beta2, float eps,
    int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = grad[i];
    g = (g >  1.0f) ?  1.0f : ((g < -1.0f) ? -1.0f : g);
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    weights[i] -= lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + eps);
}
)CL";

// General matrix multiply: C[M×N] = A[M×K] × B[K×N]
static const char* DENDRITE_MATMUL_KERNEL = R"CL(
__kernel void matmul_k(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int K, int N)
{
    int m = get_global_id(0);
    int n = get_global_id(1);
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[m * K + k] * B[k * N + n];
    C[m * N + n] = sum;
}
)CL";

// Transpose-left multiply: C[K×N] = A.T[K×M] × B[M×N]  (A stored as [M×K])
// Used for grad_w = delta.T @ input in backward pass.
static const char* DENDRITE_MATMUL_AtB_KERNEL = R"CL(
__kernel void matmul_AtB_k(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int K, int N)
{
    int k = get_global_id(0);
    int n = get_global_id(1);
    if (k >= K || n >= N) return;
    float sum = 0.0f;
    for (int m = 0; m < M; m++)
        sum += A[m * K + k] * B[m * N + n];
    C[k * N + n] = sum;
}
)CL";

// ============================================================
// Everything below is inside namespace dendrite
// ============================================================
namespace dendrite {

// ============================================================
// GPUContext — Meyer's singleton managing the OpenCL device.
// First call to instance() triggers lazy initialisation.
// If initialisation fails, available=false and all GPUBackend
// functions fall back to CPU silently.
// ============================================================
class GPUContext {
public:
    cl_platform_id   platform  = nullptr;
    cl_device_id     device    = nullptr;
    cl_context       ctx       = nullptr;
    cl_command_queue queue     = nullptr;

    cl_kernel k_matvec       = nullptr;
    cl_kernel k_matvec_trans = nullptr;
    cl_kernel k_outer        = nullptr;
    cl_kernel k_adam         = nullptr;
    cl_kernel k_matmul       = nullptr;
    cl_kernel k_matmul_AtB   = nullptr;

    bool available      = false;
    bool init_attempted = false;

    static GPUContext& instance() {
        static GPUContext g;
        if (!g.init_attempted) g.init();
        return g;
    }

    // Upload a host float buffer to a new CL buffer. Caller owns the returned
    // cl_mem and must call clReleaseMemObject() after use.
    cl_mem upload(const std::vector<float>& v,
                  cl_mem_flags extra = 0) const {
        cl_int err;
        cl_mem buf = clCreateBuffer(ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | extra,
            v.size() * sizeof(float),
            const_cast<float*>(v.data()), &err);
        return (err == CL_SUCCESS) ? buf : nullptr;
    }

    // Allocate a write-only output buffer on the device.
    cl_mem alloc(size_t elems) const {
        cl_int err;
        cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            elems * sizeof(float), nullptr, &err);
        return (err == CL_SUCCESS) ? buf : nullptr;
    }

    // Blocking download from device buffer to host vector.
    bool download(cl_mem buf, std::vector<float>& v) const {
        return clEnqueueReadBuffer(queue, buf, CL_TRUE, 0,
            v.size() * sizeof(float), v.data(), 0, nullptr, nullptr) == CL_SUCCESS;
    }

private:
    void init() {
        init_attempted = true;
        cl_int err;

        // Find first available platform
        cl_uint num_plat = 0;
        if (clGetPlatformIDs(1, &platform, &num_plat) != CL_SUCCESS || num_plat == 0) {
            std::fprintf(stderr, "[DendriteNet] OpenCL: no platform found\n");
            return;
        }

        // Prefer GPU; fall back to CPU
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS &&
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr) != CL_SUCCESS) {
            std::fprintf(stderr, "[DendriteNet] OpenCL: no device found\n");
            return;
        }

        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return;

        // cl_command_queue_properties is 0 (default) — compatible with both
        // OpenCL 1.x (clCreateCommandQueue) and 2.x (clCreateCommandQueueWithProperties)
#ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
#else
        queue = clCreateCommandQueue(ctx, device, 0, &err);
#endif
        if (err != CL_SUCCESS) { clReleaseContext(ctx); ctx = nullptr; return; }

        if (!build(DENDRITE_MATVEC_KERNEL,       "matvec_k",       &k_matvec))       return;
        if (!build(DENDRITE_MATVEC_TRANS_KERNEL, "matvec_trans_k", &k_matvec_trans)) return;
        if (!build(DENDRITE_OUTER_KERNEL,        "outer_k",        &k_outer))        return;
        if (!build(DENDRITE_ADAM_KERNEL,         "adam_update_k",  &k_adam))         return;
        if (!build(DENDRITE_MATMUL_KERNEL,       "matmul_k",       &k_matmul))       return;
        if (!build(DENDRITE_MATMUL_AtB_KERNEL,   "matmul_AtB_k",   &k_matmul_AtB))   return;

        available = true;
        char name[256] = {};
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::fprintf(stderr, "[DendriteNet] OpenCL backend ready: %s\n", name);
    }

    bool build(const char* src, const char* entry, cl_kernel* out) {
        cl_int err;
        cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
        if (err != CL_SUCCESS) return false;
        if (clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr)
                != CL_SUCCESS) {
            size_t log_sz = 0;
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
            std::string log(log_sz, '\0');
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, &log[0], nullptr);
            std::fprintf(stderr, "[DendriteNet] OpenCL build error: %s\n%s\n", entry, log.c_str());
            clReleaseProgram(prog);
            return false;
        }
        *out = clCreateKernel(prog, entry, &err);
        clReleaseProgram(prog);
        return err == CL_SUCCESS;
    }
};

// ============================================================
// GPUBackend — definitions for functions forward-declared in tensor.hpp.
// Every function has a CPU fallback so the binary degrades gracefully
// when no OpenCL device exists. NaN guards applied after GPU→CPU transfer.
// ============================================================
namespace GPUBackend {

inline Tensor gpu_matvec(const Tensor& W, const Tensor& x, const Tensor& bias) {
    GPUContext& g = GPUContext::instance();
    int rows = static_cast<int>(W.shape[0]);
    int cols = static_cast<int>(W.shape[1]);
    if (!g.available) {
        // CPU scalar fallback (matches original matvec logic)
        Tensor y({static_cast<size_t>(rows)});
        for (int i = 0; i < rows; i++) {
            float s = bias[i];
            for (int j = 0; j < cols; j++) s += W.data[i * cols + j] * x.data[j];
            y[i] = std::isfinite(s) ? s : 0.0f;
        }
        return y;
    }
    cl_mem buf_W    = g.upload(W.data);
    cl_mem buf_x    = g.upload(x.data);
    cl_mem buf_bias = g.upload(bias.data);
    cl_mem buf_y    = g.alloc(rows);
    if (!buf_W || !buf_x || !buf_bias || !buf_y) {
        if (buf_W)    clReleaseMemObject(buf_W);
        if (buf_x)    clReleaseMemObject(buf_x);
        if (buf_bias) clReleaseMemObject(buf_bias);
        if (buf_y)    clReleaseMemObject(buf_y);
        Tensor y({static_cast<size_t>(rows)});
        for (int i = 0; i < rows; i++) {
            float s = bias[i];
            for (int j = 0; j < cols; j++) s += W.data[i*cols+j] * x.data[j];
            y[i] = std::isfinite(s) ? s : 0.0f;
        }
        return y;
    }
    clSetKernelArg(g.k_matvec, 0, sizeof(cl_mem), &buf_W);
    clSetKernelArg(g.k_matvec, 1, sizeof(cl_mem), &buf_x);
    clSetKernelArg(g.k_matvec, 2, sizeof(cl_mem), &buf_bias);
    clSetKernelArg(g.k_matvec, 3, sizeof(cl_mem), &buf_y);
    clSetKernelArg(g.k_matvec, 4, sizeof(int),    &rows);
    clSetKernelArg(g.k_matvec, 5, sizeof(int),    &cols);
    size_t gsize = static_cast<size_t>(rows);
    clEnqueueNDRangeKernel(g.queue, g.k_matvec, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    Tensor y({static_cast<size_t>(rows)});
    g.download(buf_y, y.data);
    clReleaseMemObject(buf_W);
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_bias);
    clReleaseMemObject(buf_y);
    for (auto& v : y.data) if (!std::isfinite(v)) v = 0.0f;
    return y;
}

inline Tensor gpu_matvec_transposed(const Tensor& W, const Tensor& delta) {
    GPUContext& g = GPUContext::instance();
    int out_dim = static_cast<int>(W.shape[0]);
    int in_dim  = static_cast<int>(W.shape[1]);
    if (!g.available) {
        Tensor y({static_cast<size_t>(in_dim)});
        y.zero();
        for (int i = 0; i < out_dim; i++) {
            float d = delta[i];
            if (!std::isfinite(d)) continue;
            for (int j = 0; j < in_dim; j++) y.data[j] += W.data[i * in_dim + j] * d;
        }
        return y;
    }
    cl_mem buf_W     = g.upload(W.data);
    cl_mem buf_delta = g.upload(delta.data);
    cl_mem buf_y     = g.alloc(in_dim);
    if (!buf_W || !buf_delta || !buf_y) {
        if (buf_W)     clReleaseMemObject(buf_W);
        if (buf_delta) clReleaseMemObject(buf_delta);
        if (buf_y)     clReleaseMemObject(buf_y);
        Tensor y({static_cast<size_t>(in_dim)}); y.zero();
        for (int i = 0; i < out_dim; i++) {
            float d = delta[i]; if (!std::isfinite(d)) continue;
            for (int j = 0; j < in_dim; j++) y.data[j] += W.data[i*in_dim+j] * d;
        }
        return y;
    }
    clSetKernelArg(g.k_matvec_trans, 0, sizeof(cl_mem), &buf_W);
    clSetKernelArg(g.k_matvec_trans, 1, sizeof(cl_mem), &buf_delta);
    clSetKernelArg(g.k_matvec_trans, 2, sizeof(cl_mem), &buf_y);
    clSetKernelArg(g.k_matvec_trans, 3, sizeof(int),    &out_dim);
    clSetKernelArg(g.k_matvec_trans, 4, sizeof(int),    &in_dim);
    size_t gsize = static_cast<size_t>(in_dim);
    clEnqueueNDRangeKernel(g.queue, g.k_matvec_trans, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    Tensor y({static_cast<size_t>(in_dim)});
    g.download(buf_y, y.data);
    clReleaseMemObject(buf_W);
    clReleaseMemObject(buf_delta);
    clReleaseMemObject(buf_y);
    for (auto& v : y.data) if (!std::isfinite(v)) v = 0.0f;
    return y;
}

// gpu_outer: replaces Tensor::outer() when large enough.
inline Tensor gpu_outer(const Tensor& a, const Tensor& b) {
    GPUContext& g = GPUContext::instance();
    int m = static_cast<int>(a.size());
    int n = static_cast<int>(b.size());
    if (!g.available || static_cast<size_t>(m * n) < DENDRITE_GPU_MIN_ELEMS) {
        Tensor out({static_cast<size_t>(m), static_cast<size_t>(n)});
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                out.data[i * n + j] = a[i] * b[j];
        return out;
    }
    cl_mem buf_a   = g.upload(a.data);
    cl_mem buf_b   = g.upload(b.data);
    cl_mem buf_out = g.alloc(m * n);
    if (!buf_a || !buf_b || !buf_out) {
        if (buf_a)   clReleaseMemObject(buf_a);
        if (buf_b)   clReleaseMemObject(buf_b);
        if (buf_out) clReleaseMemObject(buf_out);
        Tensor out({static_cast<size_t>(m), static_cast<size_t>(n)});
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) out.data[i*n+j] = a[i]*b[j];
        return out;
    }
    clSetKernelArg(g.k_outer, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(g.k_outer, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(g.k_outer, 2, sizeof(cl_mem), &buf_out);
    clSetKernelArg(g.k_outer, 3, sizeof(int),    &m);
    clSetKernelArg(g.k_outer, 4, sizeof(int),    &n);
    size_t gsz[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
    clEnqueueNDRangeKernel(g.queue, g.k_outer, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    Tensor out({static_cast<size_t>(m), static_cast<size_t>(n)});
    g.download(buf_out, out.data);
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_out);
    for (auto& v : out.data) if (!std::isfinite(v)) v = 0.0f;
    return out;
}

// gpu_adam_update: replaces the Adam inner loop in DenseLayer::apply_adam().
inline void gpu_adam_update(
        std::vector<float>& weights, const std::vector<float>& grad,
        std::vector<float>& m, std::vector<float>& v,
        float lr, float bc1, float bc2, float beta1, float beta2, float eps) {
    GPUContext& g = GPUContext::instance();
    int n = static_cast<int>(weights.size());
    if (!g.available || n < static_cast<int>(DENDRITE_GPU_MIN_ELEMS)) {
        for (int i = 0; i < n; i++) {
            float gg = std::clamp(grad[i], -1.0f, 1.0f);
            m[i] = beta1 * m[i] + (1.0f - beta1) * gg;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gg * gg;
            weights[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
        }
        return;
    }
    cl_mem buf_w    = clCreateBuffer(g.ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), weights.data(), nullptr);
    cl_mem buf_grad = g.upload(grad);
    cl_mem buf_m    = clCreateBuffer(g.ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), m.data(), nullptr);
    cl_mem buf_v    = clCreateBuffer(g.ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), v.data(), nullptr);
    if (!buf_w || !buf_grad || !buf_m || !buf_v) {
        if (buf_w)    clReleaseMemObject(buf_w);
        if (buf_grad) clReleaseMemObject(buf_grad);
        if (buf_m)    clReleaseMemObject(buf_m);
        if (buf_v)    clReleaseMemObject(buf_v);
        for (int i = 0; i < n; i++) {
            float gg = std::clamp(grad[i], -1.0f, 1.0f);
            m[i] = beta1*m[i]+(1-beta1)*gg; v[i] = beta2*v[i]+(1-beta2)*gg*gg;
            weights[i] -= lr*(m[i]/bc1)/(std::sqrt(v[i]/bc2)+eps);
        }
        return;
    }
    clSetKernelArg(g.k_adam, 0, sizeof(cl_mem), &buf_w);
    clSetKernelArg(g.k_adam, 1, sizeof(cl_mem), &buf_grad);
    clSetKernelArg(g.k_adam, 2, sizeof(cl_mem), &buf_m);
    clSetKernelArg(g.k_adam, 3, sizeof(cl_mem), &buf_v);
    clSetKernelArg(g.k_adam, 4, sizeof(float),  &lr);
    clSetKernelArg(g.k_adam, 5, sizeof(float),  &bc1);
    clSetKernelArg(g.k_adam, 6, sizeof(float),  &bc2);
    clSetKernelArg(g.k_adam, 7, sizeof(float),  &beta1);
    clSetKernelArg(g.k_adam, 8, sizeof(float),  &beta2);
    clSetKernelArg(g.k_adam, 9, sizeof(float),  &eps);
    clSetKernelArg(g.k_adam, 10, sizeof(int),   &n);
    size_t gsize = static_cast<size_t>(n);
    clEnqueueNDRangeKernel(g.queue, g.k_adam, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    clEnqueueReadBuffer(g.queue, buf_w, CL_TRUE, 0, n*sizeof(float), weights.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(g.queue, buf_m, CL_TRUE, 0, n*sizeof(float), m.data(),       0, nullptr, nullptr);
    clEnqueueReadBuffer(g.queue, buf_v, CL_TRUE, 0, n*sizeof(float), v.data(),       0, nullptr, nullptr);
    clReleaseMemObject(buf_w);
    clReleaseMemObject(buf_grad);
    clReleaseMemObject(buf_m);
    clReleaseMemObject(buf_v);
}

// gpu_matmul: C[M×N] = A[M×K] × B[K×N]
// Called from Tensor::matmul() when DENDRITE_OPENCL is set and M*N >= threshold.
inline Tensor gpu_matmul(const Tensor& A, const Tensor& B) {
    GPUContext& g = GPUContext::instance();
    int M = static_cast<int>(A.shape[0]);
    int K = static_cast<int>(A.shape[1]);
    int N = static_cast<int>(B.shape[1]);
    if (!g.available) {
        // CPU fallback — matches matmul loop order in tensor.hpp
        Tensor C({static_cast<size_t>(M), static_cast<size_t>(N)});
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float a = A.data[m * K + k];
                if (a == 0.0f) continue;
                for (int n = 0; n < N; n++)
                    C.data[m * N + n] += a * B.data[k * N + n];
            }
        return C;
    }
    cl_mem buf_A = g.upload(A.data);
    cl_mem buf_B = g.upload(B.data);
    cl_mem buf_C = g.alloc(M * N);
    if (!buf_A || !buf_B || !buf_C) {
        if (buf_A) clReleaseMemObject(buf_A);
        if (buf_B) clReleaseMemObject(buf_B);
        if (buf_C) clReleaseMemObject(buf_C);
        Tensor C({static_cast<size_t>(M), static_cast<size_t>(N)});
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float a = A.data[m * K + k];
                if (a == 0.0f) continue;
                for (int n = 0; n < N; n++)
                    C.data[m * N + n] += a * B.data[k * N + n];
            }
        return C;
    }
    clSetKernelArg(g.k_matmul, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(g.k_matmul, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(g.k_matmul, 2, sizeof(cl_mem), &buf_C);
    clSetKernelArg(g.k_matmul, 3, sizeof(int),    &M);
    clSetKernelArg(g.k_matmul, 4, sizeof(int),    &K);
    clSetKernelArg(g.k_matmul, 5, sizeof(int),    &N);
    size_t gsz[2] = {static_cast<size_t>(M), static_cast<size_t>(N)};
    clEnqueueNDRangeKernel(g.queue, g.k_matmul, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    Tensor C({static_cast<size_t>(M), static_cast<size_t>(N)});
    g.download(buf_C, C.data);
    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
    for (auto& v : C.data) if (!std::isfinite(v)) v = 0.0f;
    return C;
}

// gpu_matmul_AtB: C[K×N] = A.T × B  where A is stored as [M×K], B as [M×N]
// Used in backward pass: grad_w += delta.T @ input_batch
inline Tensor gpu_matmul_AtB(const Tensor& A, const Tensor& B) {
    GPUContext& g = GPUContext::instance();
    int M = static_cast<int>(A.shape[0]);
    int K = static_cast<int>(A.shape[1]);
    int N = static_cast<int>(B.shape[1]);
    if (!g.available) {
        Tensor C({static_cast<size_t>(K), static_cast<size_t>(N)});
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float a = A.data[m * K + k];
                if (!std::isfinite(a)) continue;
                for (int n = 0; n < N; n++)
                    C.data[k * N + n] += a * B.data[m * N + n];
            }
        return C;
    }
    cl_mem buf_A = g.upload(A.data);
    cl_mem buf_B = g.upload(B.data);
    cl_mem buf_C = g.alloc(K * N);
    if (!buf_A || !buf_B || !buf_C) {
        if (buf_A) clReleaseMemObject(buf_A);
        if (buf_B) clReleaseMemObject(buf_B);
        if (buf_C) clReleaseMemObject(buf_C);
        Tensor C({static_cast<size_t>(K), static_cast<size_t>(N)});
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float a = A.data[m * K + k];
                if (!std::isfinite(a)) continue;
                for (int n = 0; n < N; n++)
                    C.data[k * N + n] += a * B.data[m * N + n];
            }
        return C;
    }
    clSetKernelArg(g.k_matmul_AtB, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(g.k_matmul_AtB, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(g.k_matmul_AtB, 2, sizeof(cl_mem), &buf_C);
    clSetKernelArg(g.k_matmul_AtB, 3, sizeof(int),    &M);
    clSetKernelArg(g.k_matmul_AtB, 4, sizeof(int),    &K);
    clSetKernelArg(g.k_matmul_AtB, 5, sizeof(int),    &N);
    size_t gsz[2] = {static_cast<size_t>(K), static_cast<size_t>(N)};
    clEnqueueNDRangeKernel(g.queue, g.k_matmul_AtB, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    clFinish(g.queue);
    Tensor C({static_cast<size_t>(K), static_cast<size_t>(N)});
    g.download(buf_C, C.data);
    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
    for (auto& v : C.data) if (!std::isfinite(v)) v = 0.0f;
    return C;
}

} // namespace GPUBackend

} // namespace dendrite

#endif // DENDRITE_OPENCL
