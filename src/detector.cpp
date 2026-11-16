// YOLO 检测部署 | src/detector.cpp
// 功能：图片模式和视频模式最后都会走到这里。构造 Detector 时读类别表并创建推理后端；
// 每次 detect 只处理一帧，依次准备浮点输入、执行模型、解码框、在图片副本上绘制。
// 要看像素怎样变成张量，顺着 prepare_yolo 进入 yolo_processing.cpp。
// 目的：让每次 detect 只负责一个画面，并在同一条调用链上取得可画的框和分阶段计时。
//
// 阅读顺序：构造时选后端并加载资源 → 每帧预处理 → 后端推理 → 解码与绘图。


#include "detector.h"  // 单帧检测入口，调用它即可走完预处理、推理和绘图
#include "ort_backend.h"  // 让 C++ 程序通过 ONNX Runtime 执行一次模型推理
#ifdef ENABLE_TENSORRT
#include "trt_backend.h"
#endif
#include <chrono>
#include <fstream>
#include <stdexcept>


// 模块：计时辅助
// 功能：把两个时间点的差转换成毫秒
namespace {
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}


// 模块：模型初始化（Detector::Detector）
// 功能：读取类别名称并创建选定后端
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
    // 类别文件一行一个名字。Windows 行尾的回车若保留下来，看着相同的名字也可能比较失败。
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


// 模块：单帧检测
// 功能：串联预处理、推理、后处理与绘图
// 单帧主线：原图副本用于绘制，input浮点数组用于推理，两者用途不同。
// info贯穿预处理和后处理，output是模型原始数字，返回值才是画好框的BGR图片。
cv::Mat Detector::detect(const cv::Mat& image) {
    auto begin = Clock::now();
    // cv::Mat 直接赋值通常共享像素；这里 clone 是为了把框画到独立图上，保留调用者的原图。
    auto result = image.clone();
    auto cloned = Clock::now();


    // 输入准备
    // 功能：保存几何变换并生成模型输入
    LetterboxInfo info;
    auto input = prepare_yolo(image, info, times_);
    auto prepared = Clock::now();


    // 后端执行
    // 功能：复用已加载模型获取原始输出
    // backend_ 指向构造时选好的对象。这里每帧只调用 infer，无需再读权重或重新选后端。
    auto output = backend_->infer(input);
    auto inferred = Clock::now();


    // 结果处理
    // 功能：解析检测框并画到图片副本上
    // 跟一帧结果走：decode_yolo 给出类别下标、分数和原图坐标；画图时才用类别下标查名字。因此只想导出框的调用者可以读 detections()，无需从渲染后的像素反推框。
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
