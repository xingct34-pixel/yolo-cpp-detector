#include "int8_calibrator.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

class BuildLogger : public nvinfer1::ILogger
{
public:
    void log(
        Severity severity,
        const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout
                << "[TensorRT] "
                << msg
                << std::endl;
        }
    }
};

struct BuildConfig
{
    std::string onnx_path;
    std::string engine_path;

    std::string precision = "fp32";

    std::string calib_dir;
    std::string calib_cache =
        "calibration.cache";

    int batch_size = 1;
    int input_width = 640;
    int input_height = 640;

    size_t workspace_gb = 2;
};

void printUsage(
    const char* program)
{
    std::cout
        << "\n用法:\n"
        << "  "
        << program
        << " --onnx <model.onnx>"
        << " --engine <model.engine>"
        << " --precision <fp32|fp16|int8>"
        << " [options]\n\n"

        << "参数:\n"
        << "  --onnx <path>\n"
        << "      ONNX 模型路径\n\n"

        << "  --engine <path>\n"
        << "      输出 TensorRT Engine 路径\n\n"

        << "  --precision <mode>\n"
        << "      fp32 / fp16 / int8\n\n"

        << "  --calib-dir <path>\n"
        << "      INT8 校准图片目录\n\n"

        << "  --calib-cache <path>\n"
        << "      INT8 Calibration Cache\n\n"

        << "  --batch-size <N>\n"
        << "      INT8 校准 Batch Size\n\n"

        << "  --workspace-gb <N>\n"
        << "      TensorRT Workspace 大小\n\n";
}

BuildConfig parseArguments(
    int argc,
    char* argv[])
{
    BuildConfig config;

    for (int i = 1;
         i < argc;
         ++i)
    {
        const std::string arg =
            argv[i];

        if (arg == "--help")
        {
            printUsage(argv[0]);
            std::exit(0);
        }

        if (i + 1 >= argc)
        {
            throw std::runtime_error(
                "参数缺少值: " + arg);
        }

        const std::string value =
            argv[++i];

        if (arg == "--onnx")
        {
            config.onnx_path =
                value;
        }
        else if (arg == "--engine")
        {
            config.engine_path =
                value;
        }
        else if (arg == "--precision")
        {
            config.precision =
                value;
        }
        else if (arg == "--calib-dir")
        {
            config.calib_dir =
                value;
        }
        else if (arg == "--calib-cache")
        {
            config.calib_cache =
                value;
        }
        else if (arg == "--batch-size")
        {
            config.batch_size =
                std::stoi(value);
        }
        else if (arg == "--workspace-gb")
        {
            config.workspace_gb =
                static_cast<size_t>(
                    std::stoul(value));
        }
        else
        {
            throw std::runtime_error(
                "未知参数: " + arg);
        }
    }

    if (config.onnx_path.empty() ||
        config.engine_path.empty())
    {
        throw std::runtime_error(
            "--onnx 和 --engine 是必需参数。");
    }

    if (config.precision != "fp32" &&
        config.precision != "fp16" &&
        config.precision != "int8")
    {
        throw std::runtime_error(
            "--precision 必须是 "
            "fp32 / fp16 / int8。");
    }

    if (config.precision == "int8" &&
        config.calib_dir.empty())
    {
        throw std::runtime_error(
            "INT8 模式必须提供 "
            "--calib-dir。");
    }

    return config;
}

void saveEngine(
    const std::string& path,
    nvinfer1::IHostMemory* serialized)
{
    if (!serialized)
    {
        throw std::runtime_error(
            "TensorRT Engine 构建失败。");
    }

    std::ofstream output(
        path,
        std::ios::binary);

    if (!output)
    {
        throw std::runtime_error(
            "无法创建 Engine 文件: " +
            path);
    }

    output.write(
        static_cast<const char*>(
            serialized->data()),
        serialized->size());

    if (!output)
    {
        throw std::runtime_error(
            "写入 Engine 文件失败: " +
            path);
    }
}

int main(
    int argc,
    char* argv[])
{
    try
    {
        const BuildConfig config =
            parseArguments(
                argc,
                argv);

        BuildLogger logger;

        // ==================== Builder 模块 ====================
        // Builder（建厂工程师）

        std::unique_ptr<
            nvinfer1::IBuilder>
            builder(
                nvinfer1::createInferBuilder(
                    logger));

        if (!builder)
        {
            throw std::runtime_error(
                "createInferBuilder failed.");
        }

        // ==================== Network 模块 ====================
        // Network（生产线设计图）

        const uint32_t flags =
            1U <<
            static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::
                    kEXPLICIT_BATCH);

        std::unique_ptr<
            nvinfer1::INetworkDefinition>
            network(
                builder->createNetworkV2(
                    flags));

        if (!network)
        {
            throw std::runtime_error(
                "createNetworkV2 failed.");
        }

        // ==================== ONNX Parser 模块 ====================

        std::unique_ptr<
            nvonnxparser::IParser>
            parser(
                nvonnxparser::createParser(
                    *network,
                    logger));

        if (!parser)
        {
            throw std::runtime_error(
                "createParser failed.");
        }

        std::cout
            << "开始解析 ONNX: "
            << config.onnx_path
            << std::endl;

        if (!parser->parseFromFile(
                config.onnx_path.c_str(),
                static_cast<int>(
                    nvinfer1::ILogger::Severity::
                        kWARNING)))
        {
            for (int i = 0;
                 i < parser->getNbErrors();
                 ++i)
            {
                std::cerr
                    << "[ONNX Parser] "
                    << parser->getError(i)->desc()
                    << std::endl;
            }

            throw std::
