#include "detector.h"

#include "ort_backend.h"
#include "trt_backend.h"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ==================== 构造模块 ====================

Detector::Detector(
    const std::string& model_path,
    const std::string& classes_path,
    BackendType backend_type,
    const std::string& engine_path,
    bool use_cuda)
{
    // ==================== 类别加载模块 ====================

    std::ifstream file(classes_path);

    if (!file)
    {
        throw std::runtime_error(
            "无法打开类别文件: " +
            classes_path);
    }

    std::string class_name;

    while (std::getline(file, class_name))
    {
        if (!class_name.empty())
        {
            class_names_.push_back(
                class_name);
        }
    }

    std::cout
        << "加载类别数量："
        << class_names_.size()
        << std::endl;

    // ==================== 推理后端创建模块 ====================

    backend_ =
        createBackend(
            model_path,
            backend_type,
            engine_path,
            use_cuda);

    std::cout
        << "推理后端："
        << backend_->name()
        << std::endl;
}

// ==================== 后端创建模块 ====================

std::unique_ptr<InferenceBackend>
Detector::createBackend(
    const std::string& model_path,
    BackendType backend_type,
    const std::string& engine_path,
    bool use_cuda)
{
    if (backend_type == BackendType::ORT)
    {
        return std::make_unique<OrtBackend>(
            model_path,
            use_cuda);
    }

    if (backend_type ==
        BackendType::TensorRT)
    {
        if (engine_path.empty())
        {
            throw std::runtime_error(
                "TensorRT 后端需要提供 Engine 路径。");
        }

        return std::make_unique<TrtBackend>(
            engine_path);
    }

    throw std::runtime_error(
        "未知的推理后端。");
}

// ==================== 图像预处理模块 ====================
//
// Letterbox：保持原始图像比例进行缩放，
// 剩余区域使用灰色进行 Padding。
//
// 相比直接 resize 到 640x640，
// Letterbox 能减少目标形状被拉伸的问题。

std::vector<float> Detector::preprocess(
    const cv::Mat& image,
    LetterboxInfo& info)
{
    if (image.empty())
    {
        throw std::runtime_error(
            "输入图像为空。");
    }

    const int original_width =
        image.cols;

    const int original_height =
        image.rows;

    // ==================== 计算缩放比例 ====================

    info.scale =
        std::min(
            static_cast<float>(
                INPUT_WIDTH) /
                original_width,
            static_cast<float>(
                INPUT_HEIGHT) /
                original_height);

    const int resized_width =
        static_cast<int>(
            std::round(
                original_width *
                info.scale));

    const int resized_height =
        static_cast<int>(
            std::round(
                original_height *
                info.scale));

    // ==================== Resize 模块 ====================

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(
            resized_width,
            resized_height));

    // ==================== Padding 模块 ====================

    const int pad_x =
        (INPUT_WIDTH -
         resized_width) /
        2;

    const int pad_y =
        (INPUT_HEIGHT -
         resized_height) /
        2;

    info.pad_x = pad_x;
    info.pad_y = pad_y;

    cv::Mat letterboxed;

    cv::copyMakeBorder(
        resized,
        letterboxed,
        pad_y,
        INPUT_HEIGHT -
            resized_height -
            pad_y,
        pad_x,
        INPUT_WIDTH -
            resized_width -
            pad_x,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            114,
            114,
            114));

    // ==================== 类型转换模块 ====================

    cv::Mat float_image;

    letterboxed.convertTo(
        float_image,
        CV_32FC3,
        1.0 / 255.0);

    // ==================== BGR → RGB 模块 ====================

    cv::cvtColor(
        float_image,
        float_image,
        cv::COLOR_BGR2RGB);

    // ==================== HWC → CHW 模块 ====================
    //
    // OpenCV 默认：
    // HWC = Height × Width × Channel
    //
    // YOLO Tensor：
    // CHW = Channel × Height × Width

    std::vector<cv::Mat> channels;

    cv::split(
        float_image,
        channels);

    std::vector<float> input_data;

    input_data.reserve(
        3 *
        INPUT_WIDTH *
        INPUT_HEIGHT);

    for (const auto& channel :
         channels)
    {
        input_data.insert(
            input_data.end(),
            reinterpret_cast<float*>(
                channel.data),
            reinterpret_cast<float*>(
                channel.data) +
                INPUT_WIDTH *
                INPUT_HEIGHT);
    }

    return input_data;
}

// ==================== 完整检测流程 ====================

cv::Mat Detector::detect(
    const cv::Mat& image)
{
    cv::Mat result =
        image.clone();

    // ==================== 预处理 ====================

    LetterboxInfo info;

    std::vector<float> input =
        preprocess(
            image,
            info);

    // ==================== 推理模块 ====================

    std::vector<float> output =
        backend_->infer(input);

    // ==================== 后处理模块 ====================

    postprocess(
        result,
        output,
        info);

    return result;
}

// ==================== YOLO 后处理模块 ====================
//
// YOLO11n 默认输出：
// [1, 84, 8400]
//
// 84 =
// 4 个 box 参数
// +
// 80 个类别分数
//
// 8400 = 候选框数量

void Detector::postprocess(
    cv::Mat& image,
    const std::vector<float>& output,
    const LetterboxInfo& info)
{
    constexpr int NUM_CLASSES = 80;
    constexpr int NUM_BOXES = 8400;

    if (output.size() <
        4 * NUM_BOXES)
    {
        throw std::runtime_error(
            "YOLO 输出数据大小异常。");
    }

    std::vector<cv::Rect> boxes;

    std::vector<float> scores;

    std::vector<int> class_ids;

    // ==================== 候选框解析模块 ====================

    for (int i = 0;
         i < NUM_BOXES;
         ++i)
    {
        // YOLO 输出布局：
        //
        // output[0 * NUM_BOXES + i] → cx
        // output[1 * NUM_BOXES + i] → cy
        // output[2 * NUM_BOXES + i] → w
        // output[3 * NUM_BOXES + i] → h

        const float cx =
            output[
                0 * NUM_BOXES + i];

        const float cy =
            output[
                1 * NUM_BOXES + i];

        const float width =
            output[
                2 * NUM_BOXES + i];

        const float height =
            output[
                3 * NUM_BOXES + i];

        // ==================== 找最大类别分数 ====================

        float best_score = 0.0f;

        int best_class = -1;

        for (int c = 0;
             c < NUM_CLASSES;
             ++c)
        {
            const float score =
                output[
                    (4 + c) *
                    NUM_BOXES +
                    i];

            if (score > best_score)
            {
                best_score = score;

                best_class = c;
            }
        }

        // ==================== Confidence 过滤 ====================

        if (best_score <
                CONF_THRESHOLD ||
            best_class < 0)
        {
            continue;
        }

        // ==================== Letterbox 坐标还原 ====================
        //
        // 模型坐标
        // ↓
        // 去除 Padding
        // ↓
        // 除以 scale
        // ↓
        // 原始图像坐标

        const float x1 =
            (cx - width / 2.0f -
             info.pad_x) /
            info.scale;

        const float y1 =
            (cy - height / 2.0f -
             info.pad_y) /
            info.scale;

        const float x2 =
            (cx + width / 2.0f -
             info.pad_x) /
            info.scale;

        const float y2 =
            (cy + height / 2.0f -
             info.pad_y) /
            info.scale;

        // ==================== 坐标限制模块 ====================

        const int left =
            std::max(
                0,
                std::min(
                    image.cols - 1,
                    static_cast<int>(
                        std::round(x1))));

        const int top =
            std::max(
                0,
                std::min(
                    image.rows - 1,
                    static_cast<int>(
                        std::round(y1))));

        const int right =
            std::max(
                0,
                std::min(
                    image.cols - 1,
                    static_cast<int>(
                        std::round(x2))));

        const int bottom =
            std::max(
                0,
                std::min(
                    image.rows - 1,
                    static_cast<int>(
                        std::round(y2))));

        if (right <= left ||
            bottom <= top)
        {
            continue;
        }

        boxes.emplace_back(
            left,
            top,
            right - left,
            bottom - top);

        scores.push_back(
            best_score);

        class_ids.push_back(
            best_class);
    }

    // ==================== NMS 模块 ====================

    std::vector<int> keep;

    nms(
        boxes,
        scores,
        class_ids,
        NMS_THRESHOLD,
        keep);

    // ==================== 绘制检测结果模块 ====================

    for (int index : keep)
    {
        const cv::Rect& box =
            boxes[index];

        const int class_id =
            class_ids[index];

        const float score =
            scores[index];

        cv::rectangle(
            image,
            box,
            cv::Scalar(
                0,
                255,
                0),
            2);

        std::string label =
            "unknown";

        if (class_id >= 0 &&
            class_id <
                static_cast<int>(
                    class_names_.size()))
        {
            label =
                class_names_[class_id];
        }

        label +=
            " " +
            std::to_string(score)
            .substr(0, 4);

        int baseline = 0;

        const cv::Size text_size =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline);

        const int text_y =
            std::max(
                box.y,
                text_size.height + 2);

        cv::rectangle(
            image,
            cv::Point(
                box.x,
                text_y -
                    text_size.height -
                    2),
            cv::Point(
                box.x +
                    text_size.width,
                text_y +
                    baseline),
            cv::Scalar(
                0,
                255,
                0),
            cv::FILLED);

        cv::putText(
            image,
            label,
            cv::Point(
                box.x,
                text_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(
                0,
                0,
                0),
            1);
    }
}

// ==================== NMS 模块 ====================
//
// 使用 OpenCV NMS。
// 为避免不同类别之间互相抑制，
// 给不同类别的框增加不同的坐标偏移。

void Detector::nms(
    std::vector<cv::Rect>& boxes,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    float iou_threshold,
    std::vector<int>& keep)
{
    if (boxes.empty())
    {
        return;
    }

    std::vector<cv::Rect> offset_boxes =
        boxes;

    constexpr float MAX_WH = 4096.0f;

    for (size_t i = 0;
         i < offset_boxes.size();
         ++i)
    {
        const int offset =
            static_cast<int>(
                class_ids[i] *
                MAX_WH);

        offset_boxes[i].x += offset;
        offset_boxes[i].y += offset;
    }

    cv::dnn::NMSBoxes(
        offset_boxes,
        scores,
        CONF_THRESHOLD,
        iou_threshold,
        keep);
}

// ==================== 性能信息模块 ====================

double Detector::last_inference_ms() const
{
    return backend_->last_latency_ms();
}

std::string Detector::backend_name() const
{
    return backend_->name();
}
