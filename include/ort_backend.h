// YOLO 检测部署 | include/ort_backend.h
// 功能：声明 ONNX Runtime 后端用到的会话、输入输出信息和计时接口。
// 同一个 OrtBackend 跟着 Detector 使用多帧，成员里的 Session 可以一直复用。
// 对照 src/ort_backend.cpp 的构造函数和 infer，就能区分启动工作与逐帧工作。
// 目的：区分模型加载一次需要保存的资源与每一帧临时产生的推理结果。
//
// 阅读顺序：公开给 Detector 的接口 → 初始化 Session 所需的配置 → 在多帧之间复用的状态。


#pragma once

#include "inference_backend.h"  // 声明不同后端都必须实现的 infer 等方法

#include <onnxruntime_cxx_api.h>  // ORT 的 C++ 接口，包括 Session 和 Tensor

#include <string>
#include <vector>


// 模块：ORT后端接口（OrtBackend）
// 功能：声明资源和方法，由ort_backend.cpp实现
//
// Runtime（工厂管理系统）
// Session（已经准备好的执行环境）
// Tensor（输入/输出货物）
//
// 支持：
// CPU Execution Provider
// CUDA Execution Provider
class OrtBackend : public InferenceBackend {
public:
    OrtBackend(const std::string& model_path, bool use_cuda = false);

    ~OrtBackend() override = default;


    // 推理模块（infer）
    // override 让编译器检查这个签名是否与 InferenceBackend::infer 对得上。调用基类接口时会执行这里对应的实现。
    std::vector<float> infer(const std::vector<float>& input_data) override;


    // 性能统计模块（last_latency_ms）
    double last_latency_ms() const override;


    // 后端信息模块（name）
    std::string name() const override;

private:


    // Session配置（makeSessionOptions）
    // 功能：设置CPU线程数和可选执行后端
    static Ort::SessionOptions makeSessionOptions(bool use_cuda);

private:


    // Runtime 模块
    // 这里的 Session 是对象成员。处理第 2 帧时仍可使用第 1 帧所用的会话；每次 infer 创建的输入张量只需为当次推理服务。
    // Runtime（工厂管理系统）
    // 成员按声明的逆序析构。先声明 env_ 后声明 session_，意味着先关闭会话，再释放它使用的环境。
    Ort::Env env_;

    // Session（已经准备好的执行环境）
    Ort::Session session_;


    // Tensor信息
    // 功能：保存输入输出的名称、形状和元素数量
    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;


    // 计时查询
    // 功能：返回最近一次后端调用统计
    double last_latency_ms_ = 0.0;
};
