#include "pipeline.h"
#include <opencv2/imgcodecs.hpp>
#include <csignal>
#include <iostream>
#include <limits>
#include <stdexcept>

// 模块：启动辅助：处理退出信号并校验整数参数
namespace {
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock free");

void stop(int) {
    interrupted.store(true, std::memory_order_relaxed);
}

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

// 模块：程序入口：解析参数，选择图片或视频处理流程
// 入口只负责参数校验、选择图片/视频路径及统一异常出口；检测算法留给Detector。
int main(int argc, char** argv) {
    try {

        // 模块：1. 参数配置：设置默认值并解析命令行
        std::string video = "test.mp4", image, output, model = "models/yolo11n.onnx",
                    classes = "data/coco.txt", engine = "models/yolo11n.engine";
        auto type = Detector::BackendType::ORT;
        bool cuda = false;
        int capacity = 3;
        PipelineOptions options;
        for (int i = 1; i < argc; ++i) {
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
        // 模块：2. 退出通知：把信号标志交给视频流水线
        // 信号回调只置原子标志；视频Pipeline的监视线程再正常关闭队列，回调中不做复杂清理。
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        options.external_stop = &interrupted;

        // 模块：3. 图片模式：读取图片、检测、保存图片及可选报告
        if (!image.empty()) {
            if (output.empty())
                throw std::invalid_argument("--image requires --output");
            auto input = cv::imread(image);
            if (input.empty())
                throw std::runtime_error("cannot read image: " + image);
            Detector detector(model, classes, type, engine, cuda);
            auto result = detector.detect(input);
            if (!cv::imwrite(output, result))
                throw std::runtime_error("cannot write image: " + output);
            if (!options.report.empty()) {
                cv::FileStorage fs(options.report,
                                   cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
                if (!fs.isOpened())
                    throw std::runtime_error("cannot write report");
                fs << "schema_version" << 1 << "mode" << "image" << "detections" << "[";
                for (const auto& d : detector.detections())
                    fs << "{" << "class_id" << d.class_id << "score" << d.score << "x" << d.box.x
                       << "y" << d.box.y << "width" << d.box.width << "height" << d.box.height
                       << "}";
                fs << "]" << "inference_ms" << detector.last_inference_ms();
            }
        } else {
            // 模块：4. 视频模式：构造流水线并开始处理
            Pipeline pipeline(video, model, classes, type, engine, cuda, capacity, options);
            pipeline.run();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "程序异常：" << e.what() << '\n';
        return 1;
    }
}
