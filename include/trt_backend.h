#pragma once

#include "inference_backend.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ==================== TensorRT 日志模块 ====================

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(
        Severity severity,
        const char* msg) noexcept override;
};

// ==================== TensorRT 推理后端 ====================

class TrtBackend : public InferenceBackend
{
public:
    explicit TrtBackend(
        const std::string& engine_path);

    ~TrtBackend() override;

    // ==================== 推理模块 ====================

    std::vector<float> infer(
        const std::vector<float>& input_data) override;

    // ==================== 性能统计模块 ====================

    double last_latency_ms() const override;

    // ==================== 后端信息模块 ====================

    std::string name() const override;

private:
    // ==================== Engine 加载模块 ====================

    void loadEngine(
        const std::string& engine_path);

    // ==================== Tensor 信息模块 ====================

    void inspectTensors();

    // ==================== CUDA 内存模块 ====================

    void allocateBuffers();

    void releaseBuffers();

private:
    // ==================== TensorRT 核心对象 ====================

    TrtLogger logger_;

    // Runtime（工厂管理系统）
    std::unique_ptr<nvinfer1::IRuntime>
        runtime_;

    // Engine（已经建好的生产线）
    std::unique_ptr<nvinfer1::ICudaEngine>
        engine_;

    // Context（这次生产任务）
    std::unique_ptr<nvinfer1::IExecutionContext>
        context_;

    // ==================== CUDA Stream ====================

    // Stream（生产线上的传送带）
    cudaStream_t stream_ = nullptr;

    // ==================== Tensor 信息 ====================

    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    size_t input_elements_ = 0;
    size_t output_elements_ = 0;

    // ==================== GPU Buffer ====================

    // GPU 显存中的输入
    void* device_input_ = nullptr;

    // GPU 显存中的输出
    void* device_output_ = nullptr;

    // ==================== 性能信息 ====================

    double last_latency_ms_ = 0.0;
};
