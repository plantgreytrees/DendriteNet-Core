#pragma once
// ============================================================
// OrtEncoder — thin RAII wrapper around an ONNX Runtime
// inference session.  Only compiled when -DDENDRITE_ONNX is set.
// ============================================================
// Usage:
//   OrtEncoder enc("path/to/model.onnx",
//                  {1, 3, 224, 224},   // input shape
//                  1280,               // expected output dim
//                  "input",            // ONNX input node name
//                  "output");          // ONNX output node name
//   std::vector<float> emb = enc.run(pixel_data);
// ============================================================
#ifdef DENDRITE_ONNX

// Include path works with:
//   -Ionnxruntime/include             (prebuilt ORT zip, flat headers)
//   -I/usr/include/onnxruntime        (system/vcpkg install)
//   -Iconda_env/include               (conda onnxruntime-cpp package)
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace dendrite {

/// Inference-only wrapper: one session per model file.
/// The shared Ort::Env is initialised exactly once (Meyer's singleton).
/// Sessions are thread-safe for concurrent Run() calls (ORT guarantee).
class OrtEncoder {
public:
    /// Construct and load a model.
    /// @param model_path   Path to the .onnx file.
    /// @param input_shape  Tensor dimensions expected by the model (N must be 1).
    /// @param output_dim   Number of floats expected from the output node
    ///                     (extra values are truncated; missing values are zero-padded).
    /// @param input_node   Name of the ONNX input node.
    /// @param output_node  Name of the ONNX output node.
    /// @throws Ort::Exception if the model cannot be loaded.
    OrtEncoder(const std::string&        model_path,
               std::vector<int64_t>      input_shape,
               size_t                    output_dim,
               std::string               input_node  = "input",
               std::string               output_node = "output")
        : input_shape_(std::move(input_shape))
        , output_dim_(output_dim)
        , input_node_(std::move(input_node))
        , output_node_(std::move(output_node))
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#ifdef _WIN32
        // ORT on Windows requires a wide-char path
        std::wstring wpath(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(shared_env(), wpath.c_str(), opts);
#else
        session_ = std::make_unique<Ort::Session>(shared_env(), model_path.c_str(), opts);
#endif

        total_input_ = 1;
        for (auto d : input_shape_)
            total_input_ *= static_cast<size_t>(d > 0 ? d : 1);
    }

    /// Run a single inference pass.
    /// @param data  Pointer to a flat float buffer of exactly input_size() elements.
    /// @return      Flat output embedding of length output_dim().
    /// @throws Ort::Exception on runtime error.
    [[nodiscard]] std::vector<float> run(const float* data) const {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ORT CreateTensor takes a non-const pointer; input is not mutated.
        Ort::Value in_val = Ort::Value::CreateTensor<float>(
            mem,
            const_cast<float*>(data),
            total_input_,
            input_shape_.data(),
            input_shape_.size());

        const char* in_ptr  = input_node_.c_str();
        const char* out_ptr = output_node_.c_str();

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            &in_ptr,  &in_val, 1,
            &out_ptr, 1);

        const float* p = outputs[0].GetTensorData<float>();
        size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> result(output_dim_, 0.0f);
        size_t copy_n = std::min(n, output_dim_);
        for (size_t i = 0; i < copy_n; i++)
            result[i] = std::isfinite(p[i]) ? p[i] : 0.0f;  // NaN guard
        return result;
    }

    size_t input_size()  const { return total_input_; }
    size_t output_dim()  const { return output_dim_;  }

private:
    /// Shared Ort::Env — created once per process (ORT requirement).
    static Ort::Env& shared_env() {
        static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "DendriteNet");
        return instance;
    }

    std::unique_ptr<Ort::Session> session_;
    std::vector<int64_t>          input_shape_;
    size_t                        total_input_  = 0;
    size_t                        output_dim_   = 0;
    std::string                   input_node_;
    std::string                   output_node_;
};

} // namespace dendrite

#endif // DENDRITE_ONNX
