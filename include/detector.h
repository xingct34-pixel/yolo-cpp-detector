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

    double last_image_clone_ms() const;

    double last_preprocess_ms() const;

    double last_resize_ms() const;
    double last_padding_ms() const;
    double last_tensor_prepare_ms() const;

    double last_backend_call_ms() const;

    double last_postprocess_ms() const;

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

    // 最近一次检测的阶段耗时，单位为毫秒。
    double last_image_clone_ms_ = 0.0;
    double last_resize_ms_ = 0.0;
    double last_padding_ms_ = 0.0;
    double last_tensor_prepare_ms_ = 0.0;
    double last_preprocess_ms_ = 0.0;
    double last_backend_call_ms_ = 0.0;
    double last_postprocess_ms_ = 0.0;
};
/*
FP32	32 bit	4 字节
FP16	16 bit	2 字节
INT8	8 bit	1 字节
    */

/*
​基类性质
​纯虚函数：包含纯虚函数的类是抽象类，绝对不能直接实例化（如 Base b; 会编译报错）。
​普通虚函数：基类是具体类（若无其他纯虚函数），可以直接实例化。
​子类约束
​纯虚函数：强制重写。非抽象子类必须实现该函数，否则子类也会变成抽象类，无法实例化。
​普通虚函数：可选重写。子类可以直接复用基类的默认实现，也可以重写覆盖。
​底层 vtable（虚函数表）
​纯虚函数：虚函数表对应槽位通常指向 std::__cxa_pure_virtual（错误处理函数，防止非法调用）。
​普通虚函数：虚函数表对应槽位指向基类 Base::infer 的实际函数代码地址。
*/