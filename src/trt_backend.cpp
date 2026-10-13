#include "trt_backend.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// ==================== TensorRT 日志模块 ====================

void TrtLogger::log(
    Severity severity,
    const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout
            << "[TensorRT] "
            << msg
            << std::endl;
    }
}

// ==================== 工具函数模块 ====================

namespace
{

size_t getElementCount(
    const nvinfer1::Dims& dims)
{
    size_t count = 1;

    for (int i = 0;
         i < dims.nbDims;
         ++i)
    {
        if (dims.d[i] < 0)
        {
            throw std::runtime_error(
                "TensorRT tensor contains "
                "dynamic dimension.");
        }

        count *=
            static_cast<size_t>(
                dims.d[i]);
    }

    return count;
}

}

// ==================== 构造模块 ====================

TrtBackend::TrtBackend(
    const std::string& engine_path)
{
    // ==================== Engine 加载 ====================

    loadEngine(engine_path);

    // ==================== Tensor 信息 ====================

    inspectTensors();

    // ==================== CUDA Stream ====================

    cudaError_t err =
        cudaStreamCreate(&stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaStreamCreate failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    // ==================== CUDA Buffer ====================

    allocateBuffers();

    // ==================== Tensor 地址绑定 ====================
    //
    // 把 TensorRT 的输入输出 Tensor
    // 连接到我们申请的 GPU Buffer。

    context_->setTensorAddress(
        input_name_.c_str(),
        device_input_);

    context_->setTensorAddress(
        output_name_.c_str(),
        device_output_);
}

// ==================== 析构模块 ====================

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

void TrtBackend::loadEngine(
    const std::string& engine_path)
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

    file.seekg(
        0,
        std::ios::end);

    const std::streamsize size =
        file.tellg();

    file.seekg(
        0,
        std::ios::beg);

    std::vector<char> buffer(
        static_cast<size_t>(size));

    if (!file.read(
            buffer.data(),
            size))
    {
        throw std::runtime_error(
            "读取 TensorRT Engine 失败。");
    }

    // Runtime（工厂管理系统）
    runtime_.reset(
        nvinfer1::createInferRuntime(
            logger_));

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

    // Context（这次生产任务）
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

    for (int i = 0;
         i < tensor_count;
         ++i)
    {
        const char* name =
            engine_->getIOTensorName(i);

        const auto mode =
            engine_->getTensorIOMode(name);

        const auto dims =
            engine_->getTensorShape(name);

        if (mode ==
            nvinfer1::TensorIOMode::kINPUT)
        {
            input_name_ = name;

            input_shape_.clear();

            for (int j = 0;
                 j < dims.nbDims;
                 ++j)
            {
                input_shape_.push_back(
                    dims.d[j]);
            }

            input_elements_ =
                getElementCount(dims);
        }
        else
        {
            output_name_ = name;

            output_shape_.clear();

            for (int j = 0;
                 j < dims.nbDims;
                 ++j)
            {
                output_shape_.push_back(
                    dims.d[j]);
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

    // GPU 显存：输入 Buffer
    err = cudaMalloc(
        &device_input_,
        input_elements_ *
            sizeof(float));

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaMalloc input failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    // GPU 显存：输出 Buffer
    err = cudaMalloc(
        &device_output_,
        output_elements_ *
            sizeof(float));

    if (err != cudaSuccess)
    {
        cudaFree(device_input_);

        device_input_ = nullptr;

        throw std::runtime_error(
            "cudaMalloc output failed: " +
            std::string(
                cudaGetErrorString(err)));
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
    if (input_data.size() !=
        input_elements_)
    {
        throw std::runtime_error(
            "TensorRT 输入数据大小不匹配。");
    }

    std::vector<float> output(
        output_elements_);

    // ==================== H2D ====================
    //
    // H2D（仓库 → 工厂）
    // CPU 内存 → GPU 显存

    cudaError_t err =
        cudaMemcpyAsync(
            device_input_,
            input_data.data(),
            input_elements_ *
                sizeof(float),
            cudaMemcpyHostToDevice,
            stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "H2D failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    // ==================== TensorRT 执行 ====================
    //
    // Engine（生产线）
    // Context（本次任务）
    // Stream（传送带）

    auto start =
        std::chrono::steady_clock::now();

    if (!context_->enqueueV3(stream_))
    {
        throw std::runtime_error(
            "TensorRT enqueueV3 failed.");
    }

    // 等待 GPU 执行完成
    err =
        cudaStreamSynchronize(stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaStreamSynchronize failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    auto end =
        std::chrono::steady_clock::now();

    last_latency_ms_ =
        std::chrono::duration<
            double,
            std::milli>(
                end - start)
            .count();

    // ==================== D2H ====================
    //
    // D2H（工厂 → 仓库）
    // GPU 显存 → CPU 内存

    err =
        cudaMemcpyAsync(
            output.data(),
            device_output_,
            output_elements_ *
                sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "D2H failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    err =
        cudaStreamSynchronize(stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "D2H synchronize failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    return output;
}

// ==================== 性能信息模块 ====================

double TrtBackend::last_latency_ms() const
{
    return last_latency_ms_;
}

// ==================== 后端名称模块 ====================

std::string TrtBackend::name() const
{
    return "TensorRT";
}
/*

1. Engine 怎么加载？
用 ifstream 以二进制方式把整个 .engine 文件读进 std::vector<char>，再交给 runtime_->deserializeCudaEngine(buffer.data(), buffer.size())。.engine 文件只是序列化后的字节，必须反序列化才能得到可运行的生产线。

2. Runtime 干什么？
Runtime（工厂管理系统，唯一的职责是把字节流还原成 Engine。它必须比 Engine 活得久，所以作为成员变量保存。

3. Engine 干什么？
Engine（已经建好的生产线）是 TensorRT 优化后的成品，包含算子融合、选好的 kernel 和权重。它建成后不可变，创建代价高，所以只建一次。代码里还用它查询输入输出 Tensor 的名字、形状和模式（inspectTensors）。

4. Context 干什么？
Context（这次生产任务）保存一次推理的可变状态：输入输出地址、动态形状设置、中间激活值所需的临时空间。enqueueV3 就是在 Context 上调用的。

5. 为什么需要 Context？
为了把不变的部分（Engine）和可变的部分（一次任务的状态）分开。同一条生产线可以同时开多个任务，互不干扰，每个线程或每个并发请求一个 Context，Engine 只加载一份，显存也省。

6. cudaMalloc() 干什么？
在 GPU 显存里申请仓库。代码为输入和输出各申请 元素数 × sizeof(float) 字节，在构造函数里申请一次，之后每次推理复用，避免反复申请的开销。申请后还要用 setTensorAddress 告诉 Context 这两个仓库的位置。

7. H2D 是什么？
Host to Device，仓库 → 工厂。 把 CPU 内存里的输入数据搬进 GPU 的输入仓库。GPU 只能直接处理自己显存里的数据。

8. enqueueV3() 是什么？
它把"执行这条生产线"的任务提交到指定 Stream，是异步的，提交完立刻返回，GPU 此时可能还没开始算。返回的 bool 只表示提交是否成功，不表示计算是否成功。

9. Stream 是什么？
Stream（传送带）是 GPU 上按顺序执行的操作队列。H2D、推理、D2H 都放在同一个 stream_ 上，所以先后顺序有保证：数据没运进来，推理不会开始。

10. 为什么需要 cudaStreamSynchronize()？
因为上面的操作都是异步的，CPU 提交完就往下走了。不同步的话：计时会只测到"提交"的几微秒；D2H 之后立刻读 output，数据可能还没运回来。同步就是让 CPU 等传送带上的活全部干完。

11. D2H 是什么？
Device to Host，工厂 → 仓库。把 GPU 输出仓库里的结果搬回 CPU 的 std::vector<float>，然后再同步一次，确认搬运完成才能返回结果。

12. 为什么不能直接把 GPU 内存当普通 CPU 内存读？
device_output_ 是 GPU 显存地址，和 CPU 内存不在同一个地址空间，CPU 直接解引用会崩溃或读到垃圾。数据必须通过 cudaMemcpy 这类显式搬运才能跨过去，这就是 H2D、D2H 存在的原因。

13. 推理耗时究竟测到了什么？
计时从 enqueueV3 之前开始，到第一个 cudaStreamSynchronize 之后结束，所以包含：

提交推理任务的 CPU 开销
GPU 上推理的实际执行时间
在同一个 Stream 上排在推理前面、尚未完成的 H2D 搬运
*/
