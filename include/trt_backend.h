// YOLO 检测部署 | include/trt_backend.h
// 功能：声明实验性的 TensorRT 后端。它和 OrtBackend 接受同一种预处理输入，
// 内部需要管理 NVIDIA 执行上下文、CUDA 流和显存。资源由这个对象管理。
// 先读共同接口 inference_backend.h，再到 src/trt_backend.cpp 找这些资源何时创建与释放。
// 目的：让 Detector 复用推理接口，并把实验后端专用的显存、流和上下文交给后端对象管理。
//
// 阅读顺序：记录 TensorRT 日志 → 声明推理接口 → 管理 Engine、执行流和缓冲区。


#pragma once

#include "inference_backend.h"  // 声明不同后端都必须实现的 infer 等方法

#include <NvInfer.h>  // TensorRT 构建网络和执行推理的类型
#include <cuda_runtime_api.h>  // 操作 CUDA 缓冲区、异步复制和执行流

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


// 模块：TensorRT日志（TrtLogger）
// 功能：输出后端警告和错误
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};


// 模块：TensorRT后端接口（TrtBackend）
// 功能：声明实验实现，具体逻辑位于trt_backend.cpp
class TrtBackend : public InferenceBackend {
public:
    explicit TrtBackend(const std::string& engine_path);

    // 资源跟随后端对象的生命周期；具体哪几处释放显存、销毁流，去 trt_backend.cpp 的析构函数看。
    ~TrtBackend() override;


    // 推理模块（infer）
    std::vector<float> infer(const std::vector<float>& input_data) override;


    // 性能统计模块（last_latency_ms）
    double last_latency_ms() const override;


    // 后端信息模块（name）
    std::string name() const override;

private:


    // Engine加载（loadEngine）
    // 功能：读取执行计划并创建Runtime和Context
    void loadEngine(const std::string& engine_path);


    // Tensor信息（inspectTensors）
    // 功能：保存输入输出的名称、形状和元素数量
    void inspectTensors();


    // CUDA缓冲（allocateBuffers）
    // 功能：申请和释放输入输出显存
    void allocateBuffers();

    void releaseBuffers();

private:


    // TensorRT 核心对象
    // Runtime 读入序列化的 Engine；Engine 描述可执行的网络；Context 保存执行这次推理所需的状态。具体创建顺序在 TrtBackend 的构造和 loadEngine 中。
    TrtLogger logger_;

    // Runtime（工厂管理系统）
    std::unique_ptr<nvinfer1::IRuntime> runtime_;

    // Engine（已经建好的生产线）
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;

    // Context（这次生产任务）
    std::unique_ptr<nvinfer1::IExecutionContext> context_;


    // CUDA Stream
    // Stream（生产线上的传送带）
    // 一次 infer 内的复制和执行沿同一 stream 提交。实现里同步之后，CPU 才可以放心读回传结果。
    cudaStream_t stream_ = nullptr;


    // Tensor 信息
    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    size_t input_elements_ = 0;
    size_t output_elements_ = 0;


    // GPU Buffer
    // GPU 显存中的输入
    void* device_input_ = nullptr;

    // GPU 显存中的输出
    void* device_output_ = nullptr;


    // 性能信息
    double last_latency_ms_ = 0.0;
};
