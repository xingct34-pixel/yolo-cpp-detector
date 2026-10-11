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
/*
先说明一点：你只贴了 ort_backend.h，.cpp 没贴，所以下面关于 .cpp 的回答是 ONNX Runtime 的标准写法，具体实现请对照你自己的代码确认。

⑧ 头文件：三个角色
成员	角色
Ort::Env env_	负责 Runtime 环境：全局状态、日志、内部线程池
Ort::Session session_	代表已加载好的模型：解析后的计算图、权重、选定的 Execution Provider
session_.Run()	真正执行推理的是 Session 的 Run()

env_ 要声明在 session_ 之前。C++ 成员按声明顺序构造、按相反顺序析构，这样 Session 先销毁、Env 后销毁，符合 Env 必须比 Session 活得久的要求。

input_name_、output_name_、input_shape_ 是从模型里查出来缓存的，避免每帧重复查询。

⑨ 实现：10 个问题
Env：ONNX Runtime 的全局运行环境，持有日志级别、线程池等，必须先于 Session 创建、晚于 Session 销毁。
Session：模型加载后的执行实体。它读取 .onnx、解析并优化计算图、分配权重内存，并把算子分配给 CPU 或 CUDA。
只加载一次：加载要读文件、解析图、做图优化、分配内存，CUDA 还要初始化上下文和 kernel，耗时远大于一次推理，每帧加载会让延迟暴涨。
创建 Input Tensor：用 Ort::MemoryInfo::CreateCpu(...) 描述内存位置，再用 Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size()) 创建。它只是包装 vector 的内存而不拷贝，所以 input_data 在 Run() 结束前必须有效。
input_shape_ 的来源：构造时从模型元数据读取，即 session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()。动态维度会显示为 -1，通常要替换成 1 之类的具体值。
Run() 做的事：校验输入名和形状，按拓扑顺序调用各节点在所选 Execution Provider 上的 kernel，分配输出内存，返回 std::vector<Ort::Value>。
输出能用 float* 读：输出 Tensor 是一块连续的 CPU 内存，GetTensorMutableData<float>() 返回首地址，元素个数由 GetTensorTypeAndShapeInfo().GetElementCount() 给出，再拷进 std::vector<float> 返回。
CPU 与 CUDA EP：
CPU EP 在 CPU 上用多线程 kernel 计算，无需额外依赖。
CUDA EP 通过 AppendExecutionProvider_CUDA 启用，算子在 GPU 上执行，需要匹配的 CUDA/cuDNN 版本。
使用 CPU 输入时，ORT 内部会做主机与设备之间的拷贝，首次运行也有预热开销，所以延迟统计应跳过前几帧。
SetIntraOpNumThreads(1)：限制单个算子内部（比如一次卷积或矩阵乘）使用的线程数为 1。好处是延迟稳定，也不会和管线里其他线程抢核。代价是单次推理在 CPU 上可能更慢。对 CUDA EP 影响很小。
计时范围：常见写法是用 std::chrono::steady_clock，在创建输入 Tensor 或调用 Run() 之前取起点，在 Run() 返回并把输出拷出后取终点，结果存进 last_latency_ms_。要看你的起点放在 Tensor 创建前还是 Run() 前，两者含义不同，前者包含数据准备开销。
*/
