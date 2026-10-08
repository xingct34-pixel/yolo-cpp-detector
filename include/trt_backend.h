#pragma once

#include "inference_backend.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtBackend : public InferenceBackend
{
public:
    explicit TrtBackend(const std::string& engine_path);
    ~TrtBackend() override;

    std::vector<float> infer(
        const std::vector<float>& input_data) override;

    double last_latency_ms() const override;
    std::string name() const override;

private:
    // ==================== Engine 加载模块 ====================
    void loadEngine(const std::string& engine_path);

    // ==================== Tensor 信息模块 ====================
    void inspectTensors();

    // ==================== CUDA 内存管理模块 ====================
    void allocateBuffers();

    void releaseBuffers();

private:
    TrtLogger logger_;

    // Runtime（工厂管理系统）
    std::unique_ptr<nvinfer1::IRuntime> runtime_;

    // Engine（已经建好的生产线）
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;

    // Context（这次生产任务单）
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA Stream（生产线上的传送带）
    cudaStream_t stream_ = nullptr;

    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    size_t input_elements_ = 0;
    size_t output_elements_ = 0;

    // GPU 显存地址
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;

    double last_latency_ms_ = 0.0;
};
