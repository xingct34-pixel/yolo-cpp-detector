#include "pipeline.h"

#include <iostream>
#include <stdexcept>
#include <string>

// ==================== 命令行配置模块 ====================
//cpp文件里面可以加using namespace std；，后面就不用加std：：了，，.h的头文件不能加，以防出现冲突

struct AppConfig
{
    std::string video_path =
        "test.mp4";

    std::string model_path =
        "models/yolo11n.onnx";

    std::string classes_path =
        "data/coco.txt";

    std::string engine_path =
        "models/yolo11n.engine";

    Detector::BackendType backend =
        Detector::BackendType::ORT;

    bool use_cuda = false;

    size_t queue_size = 3;
};

// ==================== 帮助信息模块 ====================

void printHelp()
{
    std::cout
        << "\nYOLO C++ Inference Deployment\n\n"

        << "Usage:\n"
        << "  ./yolo_detector [options]\n\n"

        << "Options:\n"

        << "  --backend ort|trt\n"
        << "      选择推理后端\n\n"

        << "  --ort-device cpu|cuda\n"
        << "      ORT 使用 CPU 或 CUDA\n\n"

        << "  --video <path>\n"
        << "      视频文件路径\n\n"

        << "  --model <path>\n"
        << "      ONNX 模型路径\n\n"

        << "  --engine <path>\n"
        << "      TensorRT Engine 路径\n\n"

        << "  --classes <path>\n"
        << "      类别文件路径\n\n"

        << "  --queue-size <N>\n"
        << "      Pipeline 队列大小\n\n"

        << "  --help\n"
        << "      显示帮助\n\n";
}

// ==================== 参数解析模块 ====================

AppConfig parseArgs(
    int argc,
    char* argv[])
{
    AppConfig config;

    for (int i = 1;
         i < argc;
         ++i)
    {
        const std::string arg =
            argv[i];

        if (arg == "--help")
        {
            printHelp();

            std::exit(0);
        }

        if (arg == "--backend")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--backend 缺少参数。");
            }

            const std::string value =
                argv[++i];

            if (value == "ort")
            {
                config.backend =
                    Detector::BackendType::ORT;
            }
            else if (value == "trt")
            {
                config.backend =
                    Detector::BackendType::TensorRT;
            }
            else
            {
                throw std::runtime_error(
                    "backend 必须是 ort 或 trt。");
            }

            continue;
        }

        if (arg == "--ort-device")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--ort-device 缺少参数。");
            }

            const std::string value =
                argv[++i];

            if (value == "cpu")
            {
                config.use_cuda = false;
            }
            else if (value == "cuda")
            {
                config.use_cuda = true;
            }
            else
            {
                throw std::runtime_error(
                    "ort-device 必须是 cpu 或 cuda。");
            }

            continue;
        }

        if (arg == "--video")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--video 缺少参数。");
            }

            config.video_path =
                argv[++i];

            continue;
        }

        if (arg == "--model")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--model 缺少参数。");
            }

            config.model_path =
                argv[++i];

            continue;
        }

        if (arg == "--engine")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--engine 缺少参数。");
            }

            config.engine_path =
                argv[++i];

            continue;
        }

        if (arg == "--classes")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--classes 缺少参数。");
            }

            config.classes_path =
                argv[++i];

            continue;
        }

        if (arg == "--queue-size")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "--queue-size 缺少参数。");
            }

            config.queue_size =
                std::stoul(argv[++i]);

            if (config.queue_size == 0)
            {
                throw std::runtime_error(
                    "queue-size 必须大于 0。");
            }

            continue;
        }

        throw std::runtime_error(
            "未知参数: " + arg);
    }

    // TensorRT 不使用 ORT CUDA 配置
    if (config.backend ==
        Detector::BackendType::TensorRT)
    {
        config.use_cuda = false;
    }

    return config;
}

// ==================== 主函数模块 ====================

int main(
    int argc,
    char* argv[])
{
    try
    {
        AppConfig config =
            parseArgs(
                argc,
                argv);

        std::cout
            << "==============================\n"
            << "YOLO C++ Inference Deployment\n"
            << "==============================\n";

        Pipeline pipeline(
            config.video_path,
            config.model_path,
            config.classes_path,
            config.backend,
            config.engine_path,
            config.use_cuda,
            config.queue_size);

        pipeline.run();
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\n程序异常："
            << e.what()
            << std::endl;

        return 1;
    }

    return 0;
}
