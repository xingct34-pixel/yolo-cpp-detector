// YOLO 检测部署 | include/ort_backend.h
// 声明 ONNX Runtime 后端用到的会话、输入输出信息和计时接口。
// 同一个 OrtBackend 跟着 Detector 使用多帧，成员里的 Session 可以一直复用。
// 对照 src/ort_backend.cpp 的构造函数和 infer，就能区分启动工作与逐帧工作。
//
// 阅读地图（按源码顺序）
//   1 ORT后端接口 → 2 推理模块 → 3 性能统计模块
//   4 后端信息模块 → 5 Session配置 → 6 Runtime 模块
//   7 Tensor信息 → 8 计时查询


#pragma once

#include "inference_backend.h"  // 声明不同后端都必须实现的 infer 等方法

#include <onnxruntime_cxx_api.h>  // ORT 的 C++ 接口，包括 Session 和 Tensor

#include <string>
#include <vector>  // 元素连续存放，适合装浮点张量或一组检测记录


// 模块 1：ORT后端接口
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


    // 模块 2：推理模块


    // override 让编译器检查这个签名是否与 InferenceBackend::infer 对得上。调用基类接口时会执行这里对应的实现。
    std::vector<float> infer(const std::vector<float>& input_data) override;


    // 模块 3：性能统计模块


    double last_latency_ms() const override;


    // 模块 4：后端信息模块


    std::string name() const override;

private:


    // 模块 5：Session配置
    // 功能：设置CPU线程数和可选执行后端


    static Ort::SessionOptions makeSessionOptions(bool use_cuda);

private:


    // 模块 6：Runtime 模块


    // Runtime（工厂管理系统）
    // 成员按声明的逆序析构。先声明 env_ 后声明 session_，意味着先关闭会话，再释放它使用的环境。
    Ort::Env env_;

    // Session（已经准备好的执行环境）
    Ort::Session session_;


    // 模块 7：Tensor信息
    // 功能：保存输入输出的名称、形状和元素数量


    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;


    // 模块 8：计时查询
    // 功能：返回最近一次后端调用统计


    double last_latency_ms_ = 0.0;
};
