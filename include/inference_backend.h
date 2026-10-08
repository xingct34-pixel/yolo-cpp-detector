#pragma once

#include <string>
#include <vector>

// ==================== 推理后端统一接口 ====================
//
// InferenceBackend：推理后端的统一“插座”
//
// ORT / TensorRT 都只需要实现下面几个函数，
// Detector 不需要关心底层到底使用哪个推理框架。

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    // ==================== 推理模块 ====================
    //
    // 输入：
    //   已经完成预处理的 CHW float 数据
    //
    // 输出：
    //   模型原始输出
    virtual std::vector<float> infer(
        const std::vector<float>& input_data) = 0;

    // ==================== 性能统计模块 ====================
    //
    // 返回最近一次推理耗时
    virtual double last_latency_ms() const = 0;

    // ==================== 后端信息模块 ====================
    //
    // 例如：
    //   "ONNX Runtime"
    //   "TensorRT"
    virtual std::string name() const = 0;
};
