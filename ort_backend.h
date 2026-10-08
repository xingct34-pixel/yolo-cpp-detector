#pragma once

#include "inference_backend.h"

#include <onnxruntime_cxx_api.h>
#include <string>

class OrtBackend : public InferenceBackend {
public:
    explicit OrtBackend(const std::string& model_path);

    std::vector<float> infer(
        const std::vector<float>& input_data) override;

private:
    static Ort::SessionOptions make_session_options();

private:
    Ort::Env env_;
    Ort::Session session_;
};
