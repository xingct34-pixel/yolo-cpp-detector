#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class InferenceBackend;

/*
 * ============================================================
 * Detector
 *
 * 功能：
 *
 * 负责完整的 YOLO 检测流程：
 *
 * 原始图像
 *    ↓
 * preprocess（预处理）
 *    ↓
 * inference backend（推理后端）
 *    ↓
 * raw output（模型原始输出）
 *    ↓
 * postprocess（后处理）
 *    ↓
 * 检测结果图像
 *
 * Detector 不直接依赖 ONNX Runtime / TensorRT。
 *
 * 具体推理工作交给：
 *
 * OrtBackend
 *      或
 * TrtBackend
 *
 * ============================================================
 */

class Detector
{
public:

    // ==================== 初始化模块 ====================

    Detector(
        const std::string& model_path,
        const std::string& classes_path
    );

    // ==================== 检测模块 ====================

    cv::Mat detect(cv::Mat& img);

private:

    // ==================== 前处理模块 ====================

    std::vector<float> preprocess(
        cv::Mat& img,
        int& img_w,
        int& img_h
    );

    // ==================== 后处理模块 ====================

    void postprocess(
        cv::Mat& img,
        const std::vector<float>& output_data,
        int img_w,
        int img_h
    );

private:

    /*
     * InferenceBackend：
     *
     * 实际运行时可以指向：
     *
     * OrtBackend
     * TrtBackend
     *
     * unique_ptr（独占指针）负责对象生命周期。
     */
    std::unique_ptr<InferenceBackend> backend_;

    // COCO 80 类名称
    std::vector<std::string> class_names_;
};
