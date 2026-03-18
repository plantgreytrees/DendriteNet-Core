#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <functional>
#include <string>
#include <sstream>

// Tier 1: AVX2 SIMD (enabled with -mavx2)
#ifdef __AVX2__
#include <immintrin.h>
#endif

// GPU dispatch threshold (must be visible before Tensor class methods)
// Lowered to 4096 so batched GEMM operations (forward_batch / backward_batch)
// hit the GPU at typical minibatch sizes (≥32 samples, hidden_dim=64).
#ifdef DENDRITE_OPENCL
#ifndef DENDRITE_GPU_MIN_ELEMS
#define DENDRITE_GPU_MIN_ELEMS 4096
#endif
#endif

namespace dendrite {

// Forward declarations for GPU dispatch (definitions in gpu_backend.hpp).
// Tensor is forward-declared so the function signatures can reference it;
// the bodies are defined later where Tensor is complete.
#ifdef DENDRITE_OPENCL
class Tensor;
namespace GPUBackend {
    Tensor gpu_matvec(const Tensor& W, const Tensor& x, const Tensor& bias);
    Tensor gpu_matvec_transposed(const Tensor& W, const Tensor& delta);
    Tensor gpu_matmul(const Tensor& A, const Tensor& B);
    Tensor gpu_matmul_AtB(const Tensor& A, const Tensor& B);
}
#endif

class Tensor {
public:
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape_) : shape(std::move(shape_)) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        data.resize(total, 0.0f);
    }
    Tensor(std::vector<size_t> shape_, std::vector<float> data_)
        : data(std::move(data_)), shape(std::move(shape_)) {}

    size_t size() const { return data.size(); }
    size_t rows() const { return shape.size() >= 1 ? shape[0] : 0; }
    size_t cols() const { return shape.size() >= 2 ? shape[1] : (shape.size() == 1 ? shape[0] : 0); }
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
    float& at(size_t r, size_t c) { return data[r * shape[1] + c]; }
    const float& at(size_t r, size_t c) const { return data[r * shape[1] + c]; }

    void xavier_init(std::mt19937& rng) {
        size_t fan_in = shape.size() >= 2 ? shape[1] : shape[0];
        size_t fan_out = shape[0];
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : data) v = dist(rng);
    }
    void he_init(std::mt19937& rng) {
        size_t fan_in = shape.size() >= 2 ? shape[1] : shape[0];
        float stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& v : data) v = dist(rng);
    }
    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }

    // Plain scalar loop — the compiler auto-vectorises to AVX2+FMA with
    // -O3 -march=native -ffast-math, producing the same throughput as hand-written
    // intrinsics without the GCC-O3 loop-vectoriser/intrinsic interaction crash.
    static float dot(const float* a, const float* b, size_t n) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    static Tensor matvec(const Tensor& W, const Tensor& x, const Tensor& bias) {
#ifdef DENDRITE_OPENCL
        if (W.size() >= DENDRITE_GPU_MIN_ELEMS)
            return GPUBackend::gpu_matvec(W, x, bias);
#endif
        size_t r = W.shape[0], c = W.shape[1];
        Tensor y({r});
        for (size_t i = 0; i < r; i++)
            y[i] = dot(&W.data[i * c], x.data.data(), c) + bias[i];
        return y;
    }

    // General matrix multiply: C = A [M×K] × B [K×N] → C [M×N].
    // Loop order m→k→n is cache-friendly for row-major C and B (both accessed sequentially).
    // AVX2 vectorises the inner n loop in blocks of 8.
    // When DENDRITE_OPENCL is set and M*N >= DENDRITE_GPU_MIN_ELEMS, dispatches to GPU.
    static Tensor matmul(const Tensor& A, const Tensor& B) {
        const size_t M = A.shape[0], K = A.shape[1], N = B.shape[1];
#ifdef DENDRITE_OPENCL
        if (M * N >= DENDRITE_GPU_MIN_ELEMS)
            return GPUBackend::gpu_matmul(A, B);
#endif
        Tensor C({M, N});
        // Scalar loop — the compiler auto-vectorises to AVX2+FMA with
        // -O3 -mavx2 -ffast-math, avoiding the OpenMP-outlined-function
        // target-mismatch error that arises from manual _mm256_fmadd_ps.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(M * K * N >= 32768)
#endif
        for (size_t m = 0; m < M; m++) {
            for (size_t k = 0; k < K; k++) {
                const float aval = A.data[m * K + k];
                if (aval == 0.0f) continue;
                for (size_t n = 0; n < N; n++)
                    C.data[m * N + n] += aval * B.data[k * N + n];
            }
        }
        return C;
    }

    // Transpose-left multiply: C[K×N] = A.T × B  where A is [M×K], B is [M×N].
    // Equivalent to A.T() followed by matmul, but avoids materialising the transpose.
    // Used in backward pass: grad_w = delta.T @ input_batch.
    // Dispatches to GPU when DENDRITE_OPENCL set and K*N >= DENDRITE_GPU_MIN_ELEMS.
    // Uses a scalar loop — the compiler auto-vectorises with -O3 -mavx2 -ffast-math,
    // and avoids FMA-inside-OpenMP outlined-function target-mismatch errors.
    static Tensor matmul_AtB(const Tensor& A, const Tensor& B) {
        const size_t M = A.shape[0], K = A.shape[1], N = B.shape[1];
#ifdef DENDRITE_OPENCL
        if (K * N >= DENDRITE_GPU_MIN_ELEMS)
            return GPUBackend::gpu_matmul_AtB(A, B);
#endif
        Tensor C({K, N});
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(M * K * N >= 32768)
#endif
        for (size_t m = 0; m < M; m++) {
            for (size_t k = 0; k < K; k++) {
                const float aval = A.data[m * K + k];
                if (!std::isfinite(aval)) continue;
                for (size_t n = 0; n < N; n++)
                    C.data[k * N + n] += aval * B.data[m * N + n];
            }
        }
        return C;
    }

    // C[M,N] = A[M,K] @ B[N,K]^T  — avoids materialising B^T.
    // Used in forward_batch: input_batch [B,in] @ weights [out,in]^T → [B,out].
    // Three-loop structure (m→n→k) with innermost k being contiguous for both A and B rows;
    // auto-vectorises the k loop to AVX2+FMA with -O3 -march=native -ffast-math.
    static Tensor matmul_A_Bt(const Tensor& A, const Tensor& B) {
        const size_t M = A.shape[0], K = A.shape[1], N = B.shape[0];
        assert(B.shape.size() >= 2 && B.shape[1] == K);
        Tensor C({M, N});
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(M * K * N >= 32768)
#endif
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;
                const float* a_row = &A.data[m * K];
                const float* b_row = &B.data[n * K];
                for (size_t k = 0; k < K; k++) sum += a_row[k] * b_row[k];
                C.data[m * N + n] = sum;
            }
        }
        return C;
    }

    // Computes W^T * delta without materialising the transpose.
    // W is {out_dim, in_dim}; delta is {out_dim}; result is {in_dim}.
    // Uses column-major accumulation to match the numerical behaviour of T()+dot.
    static Tensor matvec_transposed(const Tensor& W, const Tensor& delta) {
#ifdef DENDRITE_OPENCL
        if (W.size() >= DENDRITE_GPU_MIN_ELEMS)
            return GPUBackend::gpu_matvec_transposed(W, delta);
#endif
        size_t out_dim = W.shape[0], in_dim = W.shape[1];
        Tensor y({in_dim});
        for (size_t j = 0; j < in_dim; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_dim; i++)
                sum += delta[i] * W.data[i * in_dim + j];
            y[j] = sum;
        }
        return y;
    }

    Tensor relu() const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] > 0 ? data[i] : 0;
        return out;
    }
    Tensor relu_derivative() const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] > 0 ? 1.0f : 0.0f;
        return out;
    }
    Tensor softmax() const {
        Tensor out(shape);
        float mx = *std::max_element(data.begin(), data.end());
        float sum = 0;
        for (size_t i = 0; i < size(); i++) { out[i] = std::exp(data[i] - mx); sum += out[i]; }
        for (auto& v : out.data) v /= sum;
        return out;
    }
    [[nodiscard]] Tensor gumbel_softmax(float tau, std::mt19937& rng) const {
        std::uniform_real_distribution<float> dist(1e-20f, 1.0f);
        Tensor result = *this;
        for (size_t i = 0; i < result.size(); i++) {
            float u = dist(rng);
            float g = -std::log(-std::log(u));
            if (!std::isfinite(g)) g = 0.0f;
            result[i] += g;
            result[i] /= (tau > 1e-6f ? tau : 1e-6f);
        }
        return result.softmax();
    }
    Tensor sigmoid() const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = 1.0f / (1.0f + std::exp(-data[i]));
        return out;
    }
    Tensor tanh_act() const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = std::tanh(data[i]);
        return out;
    }

    Tensor operator+(const Tensor& o) const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] + o[i];
        return out;
    }
    Tensor operator-(const Tensor& o) const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] - o[i];
        return out;
    }
    Tensor operator*(const Tensor& o) const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] * o[i];
        return out;
    }
    Tensor operator*(float s) const {
        Tensor out(shape);
        for (size_t i = 0; i < size(); i++) out[i] = data[i] * s;
        return out;
    }

    // Concatenate two vectors
    static Tensor concat(const Tensor& a, const Tensor& b) {
        Tensor out({a.size() + b.size()});
        std::copy(a.data.begin(), a.data.end(), out.data.begin());
        std::copy(b.data.begin(), b.data.end(), out.data.begin() + a.size());
        return out;
    }

    // Concatenate multiple vectors
    static Tensor concat_many(const std::vector<Tensor>& tensors) {
        size_t total = 0;
        for (auto& t : tensors) total += t.size();
        Tensor out({total});
        size_t offset = 0;
        for (auto& t : tensors) {
            std::copy(t.data.begin(), t.data.end(), out.data.begin() + offset);
            offset += t.size();
        }
        return out;
    }

    // Weighted sum of tensors
    static Tensor weighted_sum(const std::vector<Tensor>& tensors,
                               const std::vector<float>& weights) {
        assert(!tensors.empty());
        Tensor out(tensors[0].shape);
        for (size_t t = 0; t < tensors.size(); t++)
            for (size_t i = 0; i < out.size(); i++)
                out[i] += tensors[t][i] * weights[t];
        return out;
    }

    // Mean of tensors
    static Tensor mean(const std::vector<Tensor>& tensors) {
        assert(!tensors.empty());
        Tensor out(tensors[0].shape);
        for (auto& t : tensors)
            for (size_t i = 0; i < out.size(); i++)
                out[i] += t[i];
        float scale = 1.0f / tensors.size();
        for (auto& v : out.data) v *= scale;
        return out;
    }

    static Tensor outer(const Tensor& a, const Tensor& b) {
        Tensor out({a.size(), b.size()});
        for (size_t i = 0; i < a.size(); i++)
            for (size_t j = 0; j < b.size(); j++)
                out.at(i, j) = a[i] * b[j];
        return out;
    }
    Tensor T() const {
        assert(shape.size() == 2);
        Tensor out({shape[1], shape[0]});
        for (size_t r = 0; r < shape[0]; r++)
            for (size_t c = 0; c < shape[1]; c++)
                out.at(c, r) = at(r, c);
        return out;
    }

    float norm() const { float s = 0; for (auto v : data) s += v * v; return std::sqrt(s); }
    float sum() const { float s = 0; for (auto v : data) s += v; return s; }
    float max_val() const { return *std::max_element(data.begin(), data.end()); }
    int argmax() const { return std::max_element(data.begin(), data.end()) - data.begin(); }

    /// Cosine similarity between this tensor and another (returns 0 if either is near-zero norm)
    [[nodiscard]] float cosine_similarity(const Tensor& other) const {
        size_t dim = std::min(size(), other.size());
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += data[i] * other[i];
            na  += data[i] * data[i];
            nb  += other[i] * other[i];
        }
        return (na > 1e-7f && nb > 1e-7f) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0f;
    }

    void clip(float lo, float hi) { for (auto& v : data) v = std::clamp(v, lo, hi); }

    // Slice: extract a sub-range
    Tensor slice(size_t start, size_t len) const {
        Tensor out({len});
        std::copy(data.begin() + start, data.begin() + start + len, out.data.begin());
        return out;
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ": ";
        std::cout << "[";
        for (size_t i = 0; i < std::min(size(), (size_t)6); i++) {
            if (i) std::cout << ", ";
            printf("%.3f", data[i]);
        }
        if (size() > 6) std::cout << ", ...";
        std::cout << "] (dim=" << size() << ")\n";
    }
}; // end class Tensor

} // namespace dendrite

// Tier 2: OpenCL GPU backend — included OUTSIDE namespace dendrite so that
// system headers (CL/cl.h, <cstring>) are not poisoned by the namespace.
// gpu_backend.hpp opens namespace dendrite internally for its definitions.
#ifdef DENDRITE_OPENCL
#include "gpu_backend.hpp"
#endif
