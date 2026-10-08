#include "trt_backend.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstring>

// ==================== TensorRT 日志模块 ====================

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

// ==================== 工具函数模块 ====================

namespace
{

size_t getElementCount(const nvinfer1::Dims& dims)
{
    size_t count = 1;

    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            throw std::runtime_error(
                "TensorRT tensor contains dynamic dimension.");
        }

        count *= static_cast<size_t>(dims.d[i]);
    }

    return count;
}

}

// ==================== 构造 / 析构模块 ====================

TrtBackend::TrtBackend(const std::string& engine_path)
{
    loadEngine(engine_path);
    inspectTensors();
    allocateBuffers();

    cudaError_t err = cudaStreamCreate(&stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaStreamCreate failed: " +
            std::string(cudaGetErrorString(err)));
    }

    // 设置 Tensor 地址
    context_->setTensorAddress(
        input_name_.c_str(),
        device_input_);

    context_->setTensorAddress(
        output_name_.c_str(),
        device_output_);
}

TrtBackend::~TrtBackend()
{
    releaseBuffers();

    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

// ==================== Engine 加载模块 ====================

void TrtBackend::loadEngine(const std::string& engine_path)
{
    std::ifstream file(
        engine_path,
        std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            "无法打开 TensorRT Engine: " +
            engine_path);
    }

    file.seekg(0, std::ios::end);

    const std::streamsize size = file.tellg();

    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(
        static_cast<size_t>(size));

    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error(
            "读取 TensorRT Engine 失败。");
    }

    // Runtime（工厂管理系统）
    runtime_.reset(
        nvinfer1::createInferRuntime(logger_));

    if (!runtime_)
    {
        throw std::runtime_error(
            "createInferRuntime failed.");
    }

    // Engine（已经建好的生产线）
    engine_.reset(
        runtime_->deserializeCudaEngine(
            buffer.data(),
            buffer.size()));

    if (!engine_)
    {
        throw std::runtime_error(
            "deserializeCudaEngine failed.");
    }

    // Context（这次生产任务单）
    context_.reset(
        engine_->createExecutionContext());

    if (!context_)
    {
        throw std::runtime_error(
            "createExecutionContext failed.");
    }
}

// ==================== Tensor 信息模块 ====================

void TrtBackend::inspectTensors()
{
    const int tensor_count =
        engine_->getNbIOTensors();

    for (int i = 0; i < tensor_count; ++i)
    {
        const char* name =
            engine_->getIOTensorName(i);

        const auto mode =
            engine_->getTensorIOMode(name);

        const auto dims =
            engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_name_ = name;

            input_shape_.clear();

            for (int j = 0; j < dims.nbDims; ++j)
            {
                input_shape_.push_back(dims.d[j]);
            }

            input_elements_ =
                getElementCount(dims);
        }
        else
        {
            output_name_ = name;

            output_shape_.clear();

            for (int j = 0; j < dims.nbDims; ++j)
            {
                output_shape_.push_back(dims.d[j]);
            }

            output_elements_ =
                getElementCount(dims);
        }
    }

    if (input_name_.empty() ||
        output_name_.empty())
    {
        throw std::runtime_error(
            "TensorRT Engine 缺少输入或输出 Tensor。");
    }

    std::cout
        << "TensorRT 输入 Tensor: "
        << input_name_
        << std::endl;

    std::cout
        << "TensorRT 输出 Tensor: "
        << output_name_
        << std::endl;
}

// ==================== CUDA 内存管理模块 ====================

void TrtBackend::allocateBuffers()
{
    cudaError_t err;

    // GPU 显存仓库：输入
    err = cudaMalloc(
        &device_input_,
        input_elements_ * sizeof(float));

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaMalloc input failed: " +
            std::string(cudaGetErrorString(err)));
    }

    // GPU 显存仓库：输出
    err = cudaMalloc(
        &device_output_,
        output_elements_ * sizeof(float));

    if (err != cudaSuccess)
    {
        cudaFree(device_input_);
        device_input_ = nullptr;

        throw std::runtime_error(
            "cudaMalloc output failed: " +
            std::string(cudaGetErrorString(err)));
    }
}

void TrtBackend::releaseBuffers()
{
    if (device_input_)
    {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }

    if (device_output_)
    {
        cudaFree(device_output_);
        device_output_ = nullptr;
    }
}

// ==================== TensorRT 推理模块 ====================

std::vector<float> TrtBackend::infer(
    const std::vector<float>& input_data)
{
    if (input_data.size() != input_elements_)
    {
        throw std::runtime_error(
            "TensorRT 输入数据大小不匹配。");
    }

    std::vector<float> output(
        output_elements_);

    // ==================== H2D 模块 ====================
    // H2D（仓库 → 工厂）：CPU 内存 → GPU 显存
    cudaError_t err = cudaMemcpyAsync(
        device_input_,
        input_data.data(),
        input_elements_ * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "H2D failed: " +
            std::string(cudaGetErrorString(err)));
    }

    // ==================== TensorRT 异步执行模块 ====================
    // enqueueV3：让 Engine（生产线）在 Stream（传送带）上开始工作
    auto start =
        std::chrono::steady_clock::now();

    if (!context_->enqueueV3(stream_))
    {
        throw std::runtime_error(
            "TensorRT enqueueV3 failed.");
    }

    err = cudaStreamSynchronize(stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaStreamSynchronize failed: " +
            std::string(cudaGetErrorString(err)));
    }

    auto end =
        std::chrono::steady_clock::now();

    last_latency_ms_ =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    // ==================== D2H 模块 ====================
    // D2H（工厂 → 仓库）：GPU 显存 → CPU 内存
    err = cudaMemcpyAsync(
        output.data(),
        device_output_,
        output_elements_ * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "D2H failed: " +
            std::string(cudaGetErrorString(err)));
    }

    err = cudaStreamSynchronize(stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "D2H synchronize failed: " +
            std::string(cudaGetErrorString(err)));
    }

    return output;
}

// ==================== 性能信息模块 ====================

double TrtBackend::last_latency_ms() const
{
    return last_latency_ms_;
}

std::string TrtBackend::name() const
{
    return "TensorRT";
}
