#pragma once

#include "inference_backend.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

// ==================== 预处理信息模块 ====================
//
// LetterboxInfo：记录 resize + padding 后的信息
//
// 后处理时需要利用这些信息，
// 把模型坐标重新映射回原始图片坐标。

struct LetterboxInfo
{
    float scale = 1.0f;

    int pad_x = 0;
    int pad_y = 0;

    int input_width = 640;
    int input_height = 640;
};

// ==================== Detector 模型推理模块 ====================

class Detector
{
public:
    enum class BackendType
    {
        ORT,
        TensorRT
    };

public:
    // ==================== 构造模块 ====================

    Detector(
        const std::string& model_path,
        const std::string& classes_path,
        BackendType backend_type,
        const std::string& engine_path = "",
        bool use_cuda = false);

    ~Detector() = default;

    // ==================== 完整检测流程 ====================

    cv::Mat detect(
        const cv::Mat& image);

    // ==================== 性能信息模块 ====================

    double last_inference_ms() const;

    std::string backend_name() const;

private:
    // ==================== 图像预处理模块 ====================

    std::vector<float> preprocess(
        const cv::Mat& image,
        LetterboxInfo& info);

    // ==================== YOLO 后处理模块 ====================

    void postprocess(
        cv::Mat& image,
        const std::vector<float>& output,
        const LetterboxInfo& info);

    // ==================== NMS 模块 ====================

    void nms(
        std::vector<cv::Rect>& boxes,
        std::vector<float>& scores,
        std::vector<int>& class_ids,
        float iou_threshold,
        std::vector<int>& keep);

    // ==================== 后端创建模块 ====================

    std::unique_ptr<InferenceBackend>
    createBackend(
        const std::string& model_path,
        BackendType backend_type,
        const std::string& engine_path,
        bool use_cuda);

private:
    // ==================== 模型配置模块 ====================

    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;

    static constexpr float CONF_THRESHOLD = 0.50f;
    static constexpr float NMS_THRESHOLD = 0.45f;

    // ==================== 类别信息模块 ====================

    std::vector<std::string> class_names_;

    // ==================== 推理后端模块 ====================

    std::unique_ptr<InferenceBackend>
        backend_;
};
