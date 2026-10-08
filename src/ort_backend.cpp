#include "ort_backend.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

// ==================== Session 配置模块 ====================

Ort::SessionOptions
OrtBackend::makeSessionOptions(
    bool use_cuda)
{
    Ort::SessionOptions options;

    // 控制 CPU 线程数量
    options.SetIntraOpNumThreads(1);

    // ==================== CUDA Execution Provider 模块 ====================
    //
    // CUDA Execution Provider
    // = GPU 推理执行后端
    //
    // 当前机器如果暂时无法稳定运行 CUDA，
    // 可以使用 CPU 模式启动项目。

    if (use_cuda)
    {
        OrtCUDAProviderOptions cuda_options{};

        cuda_options.device_id = 0;

        options.AppendExecutionProvider_CUDA(
            cuda_options);
    }

    return options;
}

// ==================== 构造模块 ====================

OrtBackend::OrtBackend(
    const std::string& model_path,
    bool use_cuda)
    : env_(
          ORT_LOGGING_LEVEL_WARNING,
          "YOLO"),
      session_(
          env_,
          model_path.c_str(),
          makeSessionOptions(use_cuda))
{
    // ==================== 输入输出 Tensor 信息模块 ====================

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name =
        session_.GetInputNameAllocated(
            0,
            allocator);

    auto output_name =
        session_.GetOutputNameAllocated(
            0,
            allocator);

    input_name_ =
        input_name.get();

    output_name_ =
        output_name.get();

    // 获取输入 Shape
    auto input_type_info =
        session_.GetInputTypeInfo(0);

    auto input_tensor_info =
        input_type_info.GetTensorTypeAndShapeInfo();

    input_shape_ =
        input_tensor_info.GetShape();

    std::cout
        << "ONNX Runtime 输入 Tensor: "
        << input_name_
        << std::endl;

    std::cout
        << "ONNX Runtime 输出 Tensor: "
        << output_name_
        << std::endl;

    std::cout
        << "模型加载成功"
        << std::endl;
}

// ==================== 推理模块 ====================

std::vector<float> OrtBackend::infer(
    const std::vector<float>& input_data)
{
    // ==================== 输入 Tensor 创建模块 ====================
    //
    // Tensor（输入货物）
    //
    // 当前使用 CPU 内存作为输入，
    // CUDA Execution Provider 会负责后续的数据传输。

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(
                input_data.data()),
            input_data.size(),
            input_shape_.data(),
            input_shape_.size());

    const char* input_names[] =
    {
        input_name_.c_str()
    };

    const char* output_names[] =
    {
        output_name_.c_str()
    };

    // ==================== Runtime 执行模块 ====================
    //
    // Runtime（工厂管理系统）
    // Session（执行环境）
    // Run（开始生产）

    auto start =
        std::chrono::steady_clock::now();

    auto outputs =
        session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

    auto end =
        std::chrono::steady_clock::now();

    last_latency_ms_ =
        std::chrono::duration<
            double,
            std::milli>(
                end - start)
            .count();

    // ==================== 输出 Tensor 获取模块 ====================

    if (outputs.empty())
    {
        throw std::runtime_error(
            "ONNX Runtime 没有返回输出。");
    }

    float* output_data =
        outputs[0]
            .GetTensorMutableData<float>();

    auto output_info =
        outputs[0]
            .GetTensorTypeAndShapeInfo();

    const size_t output_count =
        output_info.GetElementCount();

    return std::vector<float>(
        output_data,
        output_data + output_count);
}

// ==================== 性能信息模块 ====================

double OrtBackend::last_latency_ms() const
{
    return last_latency_ms_;
}

// ==================== 后端名称模块 ====================

std::string OrtBackend::name() const
{
    return "ONNX Runtime";
}
