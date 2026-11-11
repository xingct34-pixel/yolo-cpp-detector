#include "detector.h"
#include "ort_backend.h"
#ifdef ENABLE_TENSORRT
#include "trt_backend.h"
#endif
#include <chrono>
#include <fstream>
#include <stdexcept>

// 模块：计时辅助：把两个时间点的差转换成毫秒
namespace {
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}

// 模块：模型初始化：读取类别名称并创建选定后端
// 构造时读取类别名并创建后端；同一Detector处理后续各帧时复用已加载的模型。
Detector::Detector(const std::string& model,
                   const std::string& classes,
                   BackendType type,
                   const std::string& engine,
                   bool cuda) {
    std::ifstream file(classes);
    if (!file)
        throw std::runtime_error("cannot open classes: " + classes);
    std::string name;
    while (std::getline(file, name)) {
        if (!name.empty() && name.back() == '\r')
            name.pop_back();
        if (!name.empty())
            names_.push_back(name);
    }
    if (names_.size() != 80)
        throw std::runtime_error("this YOLO model requires exactly 80 class names");
    if (type == BackendType::ORT)
        backend_ = std::make_unique<OrtBackend>(model, cuda);
    else {
#ifdef ENABLE_TENSORRT
        backend_ = std::make_unique<TrtBackend>(engine);
#else
        (void)engine;
        throw std::runtime_error("TensorRT is not enabled in this build");
#endif
    }
}

// 模块：单帧检测：串联预处理、推理、后处理与绘图
// 单帧主线：原图副本用于绘制，input浮点数组用于推理，两者用途不同。
// info贯穿预处理和后处理，output是模型原始数字，返回值才是画好框的BGR图片。
cv::Mat Detector::detect(const cv::Mat& image) {
    auto begin = Clock::now();
    auto result = image.clone();
    auto cloned = Clock::now();

    // 模块：1. 输入准备：保存几何变换并生成模型输入
    LetterboxInfo info;
    auto input = prepare_yolo(image, info, times_);
    auto prepared = Clock::now();

    // 模块：2. 后端执行：复用已加载模型获取原始输出
    auto output = backend_->infer(input);
    auto inferred = Clock::now();

    // 模块：3. 结果处理：解析检测框并画到图片副本上
    detections_ = decode_yolo(output, image.size(), info);
    draw_detections(result, detections_, names_);
    auto end = Clock::now();
    // backend_ms_包含infer()中的包装/输出复制；后端自己的last_latency_ms只计其内部范围。
    // postprocess_ms_包含解析、NMS与绘制，不要当成单独NMS的耗时。
    clone_ms_ = ms(begin, cloned);
    preprocess_ms_ = ms(cloned, prepared);
    backend_ms_ = ms(prepared, inferred);
    postprocess_ms_ = ms(inferred, end);
    return result;
}
