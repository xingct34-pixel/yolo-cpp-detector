// YOLO 检测部署 | src/main.cpp
// 运行 yolo_detector 时从 main 开始。它读命令行，决定这次处理一张图片还是一段视频。
// 图片直接交给 Detector；视频交给 Pipeline，后者会反复调用同一个 Detector。
// 读代码时可以跟着 --image 看一遍，找到第一张画好检测框的图片在哪里保存。
//
// 阅读地图（按源码顺序）
//   1 启动辅助 → 2 参数校验 → 3 程序入口
//   4 参数配置 → 5 退出通知 → 6 图片模式
//   7 图片入口 → 8 写 JSON → 9 视频模式


#include "pipeline.h"  // 视频帧、处理选项和各线程之间的交接接口
#include <opencv2/imgcodecs.hpp>
#include <csignal>
#include <iostream>
#include <limits>
#include <stdexcept>


// 模块 1：启动辅助
// 功能：处理退出信号并校验整数参数


namespace {
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock free");

void stop(int) {
    interrupted.store(true, std::memory_order_relaxed);
}


// 模块 2：参数校验
// 整数参数必须整串解析完，再检查各参数自己的范围。范围是这个程序的接口约定。


int integer(const std::string& text, const std::string& key, int low, int high) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(key + " must be an integer");
    unsigned long long n = 0;
    try {
        n = std::stoull(text);
    } catch (...) {
        throw std::invalid_argument(key + " out of range");
    }
    if (n < static_cast<unsigned>(low) || n > static_cast<unsigned>(high))
        throw std::invalid_argument(key + " out of range");
    return int(n);
}
}


// 模块 3：程序入口
// 功能：解析参数，选择图片或视频处理流程
// 入口只负责参数校验、选择图片/视频路径及统一异常出口；检测算法留给Detector。


int main(int argc, char** argv) {
    try {


        // 模块 4：参数配置
        // 功能：设置默认值并解析命令行


        std::string video = "test.mp4", image, output, model = "models/yolo11n.onnx",
                    classes = "data/coco.txt", engine = "models/yolo11n.engine";
        auto type = Detector::BackendType::ORT;
        bool cuda = false;
        int capacity = 3;
        PipelineOptions options;
        for (int i = 1; i < argc; ++i) {
            // argv[0] 是程序名，从 argv[1] 才开始处理用户给的选项。这里先记下选项名字，后面才决定它需不需要值。
            std::string key = argv[i];
            if (key == "--help") {
                std::cout
                    << "YOLO C++ CPU inference\n--image PATH --output PATH | --video PATH\n--model PATH --classes PATH\n"
                    << "--backend ort|trt --ort-device cpu|cuda --engine PATH\n--mode pipeline|sequential --queue-policy block|drop-oldest\n"
                    << "--queue-size 1..64 --max-frames 0..1000000 --warmup 0..100 --report PATH --show --headless\n";
                return 0;
            }
            if (key == "--show") {
                options.show = true;
                continue;
            }
            if (key == "--headless") {
                options.show = false;
                continue;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument(key + " missing value");
            // 例如 --image boat.jpg：此刻 i 指向 --image，++i 先移动到 boat.jpg，再保存路径。循环末尾还会再移动到下一个选项。
            std::string value = argv[++i];
            if (key == "--video")
                video = value;
            else if (key == "--image")
                image = value;
            else if (key == "--output")
                output = value;
            else if (key == "--model")
                model = value;
            else if (key == "--classes")
                classes = value;
            else if (key == "--engine")
                engine = value;
            else if (key == "--report")
                options.report = value;
            else if (key == "--queue-size")
                // 容量 1 到 64 是项目给队列设的范围。0、4abc 和纯数字 65 都拒绝，但分别违背不同的检查。
                capacity = integer(value, key, 1, 64);
            else if (key == "--max-frames")
                options.max_frames = integer(value, key, 0, 1000000);
            else if (key == "--warmup")
                options.warmup = integer(value, key, 0, 100);
            else if (key == "--backend") {
                if (value == "ort")
                    type = Detector::BackendType::ORT;
                else if (value == "trt")
                    type = Detector::BackendType::TensorRT;
                else
                    throw std::invalid_argument("invalid backend");
            } else if (key == "--ort-device") {
                if (value == "cpu")
                    cuda = false;
                else if (value == "cuda")
                    cuda = true;
                else
                    throw std::invalid_argument("invalid ort-device");
            } else if (key == "--mode") {
                if (value == "pipeline")
                    options.sequential = false;
                else if (value == "sequential")
                    options.sequential = true;
                else
                    throw std::invalid_argument("invalid mode");
            } else if (key == "--queue-policy") {
                if (value == "block")
                    options.policy = QueuePolicy::Block;
                else if (value == "drop-oldest")
                    options.policy = QueuePolicy::DropOldest;
                else
                    throw std::invalid_argument("invalid queue-policy");
            } else
                throw std::invalid_argument("unknown argument: " + key);
        }


        // 模块 5：退出通知
        // 功能：把信号标志交给视频流水线
        // 信号回调只置原子标志；视频Pipeline的监视线程再正常关闭队列，回调中不做复杂清理。


        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        // 把同一个退出标志的地址交给流水线，监视线程才能看到信号处理函数后来写入的状态。
        options.external_stop = &interrupted;


        // 模块 6：图片模式
        // 功能：读取图片、检测、保存图片及可选报告


        if (!image.empty()) {
            if (output.empty())
                throw std::invalid_argument("--image requires --output");
            auto input = cv::imread(image);
            if (input.empty())
                throw std::runtime_error("cannot read image: " + image);


            // 模块 7：图片入口
            // Detector 构造时打开模型，detect 用这一张图片推理。模型路径错和图片打不开是两个不同的失败点。


            Detector detector(model, classes, type, engine, cuda);
            auto result = detector.detect(input);
            if (!cv::imwrite(output, result))
                throw std::runtime_error("cannot write image: " + output);
            if (!options.report.empty()) {
                cv::FileStorage fs(options.report,
                                   cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
                if (!fs.isOpened())
                    throw std::runtime_error("cannot write report");


                // 模块 8：写 JSON
                // 绘图给人看；这里还写出坐标及分数，脚本可以直接检查检测结果而不用识别画面。


                fs << "schema_version" << 1 << "mode" << "image" << "detections" << "[";
                for (const auto& d : detector.detections())
                    fs << "{" << "class_id" << d.class_id << "score" << d.score << "x" << d.box.x
                       << "y" << d.box.y << "width" << d.box.width << "height" << d.box.height
                       << "}";
                fs << "]" << "inference_ms" << detector.last_inference_ms();
            }
        } else {


            // 模块 9：视频模式
            // 功能：构造流水线并开始处理


            Pipeline pipeline(video, model, classes, type, engine, cuda, capacity, options);
            pipeline.run();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "程序异常：" << e.what() << '\n';
        return 1;
    }
}
