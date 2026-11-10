#include "ort_backend.h"
#include <cstdlib>

#include <chrono>
#include <iostream>
#include <stdexcept>

// ==================== Session 配置模块 ====================

Ort::SessionOptions
OrtBackend::makeSessionOptions(
    bool use_cuda)
{
    Ort::SessionOptions options;

    // 单个算子内部默认使用3个CPU线程，可由环境变量覆盖；与视频Pipeline的角色线程数不同。
    int intra_op_threads = 3;

    if (const char* env_value =
            std::getenv("YOLO_ORT_INTRA_OP_THREADS"))
    {
        char* end = nullptr;
        const long parsed =
            std::strtol(env_value, &end, 10);

        if (end == env_value ||
            *end != 0 ||
            parsed < 1 ||
            parsed > 64)
        {
            throw std::invalid_argument(
                "YOLO_ORT_INTRA_OP_THREADS must be an integer from 1 to 64.");
        }

        intra_op_threads = static_cast<int>(parsed);
    }

    options.SetIntraOpNumThreads(intra_op_threads);

    std::cout
        << "ONNX Runtime intra-op threads: "
        << intra_op_threads
        << std::endl;

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

    // 保存到std::string中：allocator管理的临时名称退出作用域后，成员名称仍有效。
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

    // 本适配层只接固定float32输入[1,3,640,640]、输出[1,84,8400]。
    // 动态尺寸、不同类别数或带NMS的导出模型需要另行适配，不能只改一个8400常量。
    if(session_.GetInputCount()!=1 || session_.GetOutputCount()!=1 ||
       input_tensor_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
       input_shape_!=std::vector<int64_t>({1,3,640,640}))
        throw std::runtime_error("expected one float32 input [1,3,640,640] and one output");
    auto out_type=session_.GetOutputTypeInfo(0);
    auto out_info=out_type.GetTensorTypeAndShapeInfo();
    if(out_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
       out_info.GetShape()!=std::vector<int64_t>({1,84,8400}))
        throw std::runtime_error("expected float32 output [1,84,8400]");
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

    if(input_data.size()!=3u*640*640)throw std::invalid_argument("invalid YOLO input size");
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    // 包装现成vector的内存，不复制输入；同步Run结束前input_data必须保持有效且不能扩容。
    // input_data.size()是float元素个数，不是字节数；shape说明这些元素怎样分组。
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

    // 只计同步Run调用：不含输入张量包装、后面的输出复制及Detector预后处理。
    // CUDA路径还可能含EP管理的数据传输，这不是单独GPU kernel耗时，也未在本机验收。
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

    const auto output_shape =
        output_info.GetShape();

    if (output_shape.size() != 3 ||
        output_shape[0] != 1 ||
        output_shape[1] != 84 ||
        output_shape[2] != 8400)
    {
        throw std::runtime_error(
            "YOLO 输出 Tensor 形状异常，期望 [1, 84, 8400]。");
    }

    const size_t output_count =
        output_info.GetElementCount();

    // 输出指针属于outputs中的Ort::Value；复制到自己的vector，函数返回后数据才仍有效。
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


/*




2. Session
session_(env_, model_path.c_str(), makeSessionOptions(use_cuda)) 在构造时就读取 .onnx、解析优化计算图、分配权重内存，之后它就是"已加载好的模型"。

3. 为什么只加载一次
加载动作全在构造函数里，每帧不会重新加载。

4. Input Tensor 怎么创建
在 infer() 里用 CreateCpu 描述内存位置，再用 CreateTensor<float> 包装 input_data 的内存，不拷贝。const_cast 用于适配此 API 的非 const 指针签名，此处只作为模型输入使用，不能把去掉 const 理解成任何写入都安全。同步 Run() 结束前 input_data 必须有效且地址不变。

5. input_shape_ 从哪里来
来自构造函数里的 GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()。本项目明确检查 [1,3,640,640]；若是动态维度 -1 或其他形状，会在构造阶段拒绝加载，尚不支持动态输入。

6. session.Run() 做了什么
按名字绑定输入输出，在所选 Execution Provider 上按拓扑顺序执行各算子，返回 std::vector<Ort::Value>。

7. 输出为什么能通过 float* 读取
输出 Tensor 是连续内存，GetTensorMutableData<float>() 返回首地址，GetElementCount() 给出元素个数。本项目在构造时检查输出为 float32；直接换成 FP16 输出模型会被拒绝，不会继续按 float32 解析。

8. CPU 和 CUDA Execution Provider 的区别
use_cuda 为 true 时，AppendExecutionProvider_CUDA 让算子在 GPU 上跑。输入仍是 CPU 内存，ORT 内部自动做主机到设备的拷贝，输出也会拷回 CPU，所以 float* 才能直接读。如果 CUDA 环境或 onnxruntime-gpu 包不可用，这个调用会抛异常。另外 CUDA 首次运行有预热开销，统计延迟时建议跳过前几帧。

9. SetIntraOpNumThreads(intra_op_threads) 是干什么的
控制算子内部的 CPU 并行度，本项目默认 3，可用 YOLO_ORT_INTRA_OP_THREADS 设置 1~64。线程不是越多越快，需在同一负载下实测；启用 CUDA 时也不代表所有算子一定由 GPU 执行。

10. 推理耗时从哪里开始计时，到哪里结束
计时只包住同步 session_.Run(...)，不包含 Tensor 创建和最后的输出复制，所以 last_latency_ms() 是 Run API 延迟，不是整个 infer() 或整帧延迟；CUDA EP 路径还可能含传输开销，不能当作纯 GPU kernel 时间。
*/
