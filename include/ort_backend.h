// YOLO 检测部署 | include/ort_backend.h
// 声明 ONNX Runtime 后端用到的会话、输入输出信息和计时接口。
// 同一个 OrtBackend 跟着 Detector 使用多帧，成员里的 Session 可以一直复用。
// 对照 src/ort_backend.cpp 的构造函数和 infer，就能区分启动工作与逐帧工作。


#pragma once

#include "inference_backend.h"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>


// 模块：ORT后端接口：声明资源和方法，由ort_backend.cpp实现
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


    // 模块：推理模块


    std::vector<float> infer(const std::vector<float>& input_data) override;


    // 模块：性能统计模块


    double last_latency_ms() const override;


    // 模块：后端信息模块


    std::string name() const override;

private:


    // 模块：Session配置：设置CPU线程数和可选执行后端


    static Ort::SessionOptions makeSessionOptions(bool use_cuda);

private:


    // 模块：Runtime 模块


    // Runtime（工厂管理系统）
    Ort::Env env_;

    // Session（已经准备好的执行环境）
    Ort::Session session_;


    // 模块：Tensor信息：保存输入输出的名称、形状和元素数量


    std::string input_name_;
    std::string output_name_;

    std::vector<int64_t> input_shape_;


    // 模块：计时查询：返回最近一次后端调用统计


    double last_latency_ms_ = 0.0;
};
