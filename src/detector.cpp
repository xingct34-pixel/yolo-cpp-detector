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
    //postprocess()	解析预测结果、筛选目标、去除重复框、恢复坐标
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
        //opencv更习惯左上角和右下角定位，yolo更习惯于中心点+宽高定位
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

/*
1. 为什么要 Letterbox？

模型的输入尺寸是固定的（这里是 640×640），而真实图片的尺寸和宽高比各不相同。Letterbox 的做法是：等比缩放到刚好能放进 640×640，剩下的空白用固定灰色（通常是 114）填充。
2. 为什么不直接 resize？

直接 resize 到 640×640 会把宽高比拉变形。比如 1920×1080 的图被强行压成正方形

3. 为什么 BGR 转 RGB？

OpenCV 读图的通道顺序是 BGR，而绝大多数 YOLO 模型是用 RGB 顺序训练的。通道顺序错了，模型看到的是红蓝对调的图：不会报错，但置信度会偏低

4. 为什么除以 255？

归一化，图像像素是 0 到 255 的 uint8，而模型训练时输入被归一化到 0.0 到 1.0。

5. 为什么 HWC 转 CHW？

OpenCV 的 cv::Mat 内存布局是 HWC（高、宽、通道），也就是每个像素的 B、G、R 挨在一起。而模型输入张量的布局是 NCHW（批次、通道、高、宽），即先放完整张 R 通道，再放 G，再放 B。

6. 为什么最后使用 vector<float>？
类型匹配：模型输入是 float32，归一化后的值（0 到 1 的小数）只能用浮点数存。
内存连续：vector 保证元素连续存放，可以直接用 .data() 取指针交给后端，不需要再拷贝一次，ORT 和 TensorRT 都吃这种裸指针加形状。
后端无关：preprocess() 返回标准库类型，InferenceBackend 接口不需要暴露任何 OpenCV 或 ORT 的类型。
自动管理内存：不用手动 new/delete，也不怕异常时泄漏。

1. [1,84,8400] 是什么意思?

这是模型输出张量的形状:1 是 batch size(一张图),84 是每个候选框的数值个数,8400 是候选框的数量。内存布局是"按通道排列",所以代码里用 output[c * 8400 + i] 取第 i 个候选框的第 c 个数值。

2. 84 从哪里来?

84 = 4 + 80。前 4 个是框参数 cx, cy, w, h,后 80 个是 COCO 80 个类别各自的分数。YOLO11 是 anchor-free,没有单独的 objectness 分数,所以只有这两部分。

3. 8400 是什么?

是模型在三个尺度的特征图上产生的预测总数。输入 640×640,步长分别为 8、16、32:

80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400

每个格子产生一个预测,其中绝大多数是背景。

4. 为什么要找最高类别分数?

每个候选框有 80 个类别分数,但这里把一个框当作一个物体。取分数最大的类别作为它的类别,这个最大分数同时作为置信度。

5. confidence threshold 干什么?

8400 个候选框里大部分是背景或低质量预测。阈值(CONF_THRESHOLD)把分数过低的直接丢掉,既减少误检,也减少后面 NMS 的计算量。阈值太高会漏检,太低会误检增多。

6. NMS 解决什么问题?

同一个物体周围的多个格子都会预测出框,结果是一堆高度重叠的框。NMS 的做法是:保留分数最高的框,删除与它 IoU 超过 NMS_THRESHOLD 的其他框,重复直到处理完,最终每个物体只留一个框。

7. 为什么 NMS 不能把所有类别混在一起?

如果混在一起,重叠很大的不同物体会互相抑制。比如人骑自行车,”人"和"自行车"的框重叠很多,分数低的那个会被错误删除。

代码的做法是给每个框加上 class_id * 4096 的坐标偏移,不同类别的框被推到互不相交的位置,IoU 为 0,于是只有同类别的框之间才会互相抑制。

8. 为什么要坐标映射?

模型输出的坐标是在 640×640 的 letterbox 图上的,而你要画框的是原图,尺寸不同,还多了灰边。不还原的话,框会整体偏移、大小也不对。

9. scale / padding 怎么影响原图坐标?

预处理的顺序是"先缩放,再补边",所以还原要反过来:先减 padding,再除以 scale。

x_原 = (x_模型 − pad_x) / scale
y_原 = (y_模型 − pad_y) / scale

以 1920×1080 的图为例:

scale = min(640/1920, 640/1080) = 1/3,缩放后是 640×360
pad_x = 0,pad_y = (640 − 360) / 2 = 140
模型坐标 (320, 320) 还原为 x = (320 − 0) × 3 = 960,y = (320 − 140) × 3 = 540,正好是原图中心

最后 std::max/min 把坐标限制在图像范围内,防止框越界。
*/



/*
3. Detector 和 OrtBackend 的区别
Detector 更关心的是如何完成一次目标检测：
1. 把图片调整到模型需要的输入格式。
2. 调用后端执行模型。
3. 把模型输出解析成检测框、类别和置信度。
4. 将检测框坐标映射回原始图片。
而 OrtBackend 更关心的是如何让 ONNX Runtime 把模型运行起来，包括创建 Session、准备输入张量、执行推理和获取输出。


/*
1. 为什么同一个目标会出现多个框？

主要有三个原因：

密集预测机制：YOLO、SSD、Faster R-CNN 等检测器会在特征图的每个位置、每个尺度放置大量锚框（或直接预测框），目标附近的多个网格或锚点都会"认为"自己负责这个目标，各自输出一个框和较高的置信度。
训练时的正样本分配：训练时通常把 IoU 超过阈值的多个锚框都当作正样本，模型因此学会了"目标附近的多个位置都要响应"，推理时自然会出现重复预测。
多尺度重复检测：同一艘船可能同时在不同尺度的特征层上被检出。船只尺寸跨度大、长宽比极端（细长），这种情况更明显。

2. NMS 的核心思路

NMS 的思路是"保留最好的，抑制与它高度重叠的"。以单个类别为例：

把所有候选框按置信度从高到低排序。
取出得分最高的框，保留它。
计算它与剩余所有框的 IoU，把 IoU 超过阈值（如 0.5）的框视为重复检测，直接删除。
在剩下的框中重复第 2、3 步，直到没有框为止。

最终每个目标只留下一个框。多类别检测时，通常按类别分别做 NMS。

几个进阶要点：

阈值的取舍：阈值太低，会误删相邻的不同目标（港口里并排停靠的船很容易被误删）；阈值太高，重复框清除不干净。
Soft-NMS：不直接删除重叠框，而是按 IoU 大小衰减其置信度，对密集目标更友好。
旋转框：船只检测常用旋转框，需要用旋转 IoU 做 NMS，计算更复杂。
其他变体：DIoU-NMS 在 IoU 之外考虑中心点距离，Weighted Box Fusion 则对重叠框加权融合而不是只保留一个。

3. 置信度过滤与 NMS 的区别

置信度过滤：看单个框，用一个分数阈值（如 0.25）去掉"不太像目标"的框，解决的是误检、背景噪声问题。它不考虑框与框之间的关系。
NMS：看框与框之间的空间关系，去掉"对同一目标的重复预测"，解决的是冗余重复问题。它不判断框本身是否可信。
*/
*/

/*
    
模型输出 [1, 84, 8400]
        ↓
解析每个候选位置的预测
        ↓
提取框坐标和类别分数
        ↓
置信度过滤：去掉低分候选框
        ↓
坐标格式转换：中心点/宽高 → 角点
        ↓
NMS：抑制重复检测框
        ↓
坐标还原：模型输入坐标 → 原图坐标
        ↓
输出最终检测结果
        ↓
在视频帧上绘制检测框和类别
*/
