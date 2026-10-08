#pragma once

#include "inference_backend.h"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

// ==================== ONNX Runtime 推理后端 ====================
//
// Runtime（工厂管理系统）
// Session（已经准备好的执行环境）
// Tensor（输入/输出货物）
//
// 支持：
// CPU Execution Provider
// CUDA Execution Provider

class OrtBackend : public InferenceBackend
{
public:
    OrtBackend(
        const std::string& model_path,
        bool use_cuda = false);

    ~OrtBackend() override = default;

    // ==================== 推理模块 ====================

    std::vector<float> infer(
        const std::vector<float>& input_data) override;

    // ==================== 性能统计模块 ====================

    double last_latency_ms() const override;

    // ==================== 后端信息模块 ====================

    std::string name() const override;

private:
    // ==================== Session 配置模块 ====================

    static Ort::SessionOptions
    makeSessionOptions(
        bool use_cuda);

private:
    // ==================== Runtime 模块 ====================

    // Runtime（工厂管理系统）
    Ort::Env env_;

    // Session（已经准备好的执行环境）
    Ort::Session session_;

    // ==================== Tensor 信息模块 ====================

    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;

    // ==================== 性能信息模块 ====================

    double last_latency_ms_ = 0.0;
};
