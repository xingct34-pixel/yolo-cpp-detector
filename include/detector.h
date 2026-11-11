#pragma once
#include "inference_backend.h"
#include "yolo_processing.h"
#include <memory>
#include <string>

// 模块：Detector接口：对外提供单帧检测和最近一次结果/计时
class Detector {
public:
    // 模块：后端选择：构造时确定使用ORT或TensorRT
    enum class BackendType { ORT, TensorRT };
    Detector(const std::string&,
             const std::string&,
             BackendType,
             const std::string& engine_path = "",
             bool use_cuda = false);

    // 模块：检测入口：输入原图，返回画好框的图片
    cv::Mat detect(const cv::Mat&);

    // 模块：只读结果接口：查询检测记录和各阶段统计
    const std::vector<Detection>& detections() const {
        return detections_;
    }

    double last_inference_ms() const {
        return backend_->last_latency_ms();
    }

    double last_image_clone_ms() const {
        return clone_ms_;
    }

    double last_preprocess_ms() const {
        return preprocess_ms_;
    }

    double last_resize_ms() const {
        return times_.resize_ms;
    }

    double last_padding_ms() const {
        return times_.padding_ms;
    }

    double last_tensor_prepare_ms() const {
        return times_.tensor_ms;
    }

    double last_backend_call_ms() const {
        return backend_ms_;
    }

    double last_postprocess_ms() const {
        return postprocess_ms_;
    }

    std::string backend_name() const {
        return backend_->name();
    }

private:
    // 模块：内部资源：保存类别名、独占后端和最近一帧状态
    std::vector<std::string> names_;
    // 独占后端所有权；通过统一虚接口调用ORT/TRT，Detector销毁时自动释放后端资源。
    // detect()会更新检测记录和计时，当前Pipeline仅让一个推理线程使用此对象。
    std::unique_ptr<InferenceBackend> backend_;
    std::vector<Detection> detections_;
    PreprocessTimes times_;
    double clone_ms_ = 0, preprocess_ms_ = 0, backend_ms_ = 0, postprocess_ms_ = 0;
};
