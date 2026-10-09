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
/*
    backend_ 是指向抽象接口的 unique_ptr，是 Detector 与具体推理引擎之间的唯一连接点。
    detect() 里只需要调用类似 backend_->infer(input)，运行时实际执行的是 ORT 还是 TensorRT 的实现（多态）。用 unique_ptr 表示 Detector 独占后端，析构时自动释放，所以 ~Detector() = default 就够了。
*/

/*
    BackendType 是干什么的？

它是构造时的选择开关，告诉工厂函数 createBackend() 创建哪种后端。用嵌套的 enum class 有两个好处：

名字被限定在 Detector::BackendType::ORT 内，不会污染全局命名空间。
不会被隐式转换成 int，避免传错参数。

engine_path 和 use_cuda 也只在特定后端下才有意义，这些差异都被封装在 createBackend() 里。
*/

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

/*
4. LetterboxInfo 为什么存在？

预处理会把原图等比缩放，再补边到 640×640。这一步改变了坐标系，而模型输出的框是相对于 640×640 的。后处理想把框还原到原图，就必须知道预处理做了什么。
*/

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

/*
1. 为什么要 Letterbox？

模型的输入尺寸是固定的（这里是 640×640），而真实图片的尺寸和宽高比各不相同。Letterbox 的做法是：等比缩放到刚好能放进 640×640，剩下的空白用固定灰色（通常是 114）填充。这样既满足了固定尺寸，又不改变图中物体的形状。

2. 为什么不直接 resize？

直接 resize 到 640×640 会把宽高比拉变形。比如 1920×1080 的图被强行压成正方形，人会变瘦变高，车会变短变高。

模型训练时（Ultralytics 的 YOLO 训练流程）用的就是 letterbox，推理时形变的物体和训练分布不一致，精度会下降。
形变后框的宽高比也是失真的，还原坐标时更难处理。

代价是多了一步记录 scale / pad_x / pad_y，后处理还要反向映射，这正是 LetterboxInfo 存在的原因。

3. 为什么 BGR 转 RGB？

OpenCV 读图的通道顺序是 BGR，而绝大多数 YOLO 模型是用 RGB 顺序训练的。通道顺序错了，模型看到的是红蓝对调的图：不会报错，但置信度会偏低、类别可能出错，是个很隐蔽的 bug。

要不要转，取决于你的模型训练和导出时用的顺序，常见的 Ultralytics 导出模型需要转 RGB。

4. 为什么除以 255？

图像像素是 0 到 255 的 uint8，而模型训练时输入被归一化到 0.0 到 1.0。输入的数值范围必须和训练时一致，否则第一层卷积的激活值尺度全错，检测效果会严重变差。

归一化也有助于数值稳定，是网络训练时通用的做法。同样地，有些模型还会再减均值除标准差，这要看具体模型，YOLO 系列通常只做除以 255。

5. 为什么 HWC 转 CHW？

OpenCV 的 cv::Mat 内存布局是 HWC（高、宽、通道），也就是每个像素的 B、G、R 挨在一起。而模型输入张量的布局是 NCHW（批次、通道、高、宽），即先放完整张 R 通道，再放 G，再放 B。

这是模型导出时就固定下来的约定，布局不对，数据会被错误地解释，结果是乱的。
CHW 对卷积运算在 GPU 上的内存访问更友好。

所以预处理里要把交错存放的像素重排成按通道分开的平面。批次维度 N=1，只是在最前面补一个 1，不需要额外搬运数据。

6. 为什么最后使用 vector<float>？
类型匹配：模型输入是 float32，归一化后的值（0 到 1 的小数）只能用浮点数存。
内存连续：vector 保证元素连续存放，可以直接用 .data() 取指针交给后端，不需要再拷贝一次，ORT 和 TensorRT 都吃这种裸指针加形状。
后端无关：preprocess() 返回标准库类型，InferenceBackend 接口不需要暴露任何 OpenCV 或 ORT 的类型。
自动管理内存：不用手动 new/delete，也不怕异常时泄漏。
*/
