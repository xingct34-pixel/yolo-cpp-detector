
#include "int8_calibrator.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

class BuildLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

struct BuildConfig
{
    std::string onnx_path;
    std::string engine_path;

    std::string precision = "fp32";

    std::string calib_dir;
    std::string calib_cache = "calibration.cache";

    int batch_size = 1;
    int input_width = 640;
    int input_height = 640;

    size_t workspace_gb = 2;
};

static void printUsage(const char* program)
{
    std::cout
        << "\n用法:\n"
        << "  " << program
        << " --onnx <model.onnx>"
        << " --engine <model.engine>"
        << " --precision <fp32|fp16|int8>"
        << " [options]\n\n"
        << "参数:\n"
        << "  --onnx <path>          输入 ONNX 模型\n"
        << "  --engine <path>        输出 TensorRT Engine\n"
        << "  --precision <mode>     fp32 / fp16 / int8\n"
        << "  --calib-dir <path>     INT8 校准图片目录\n"
        << "  --calib-cache <path>   INT8 校准缓存文件\n"
        << "  --batch-size <N>       INT8 校准 Batch Size\n"
        << "  --workspace-gb <N>     Workspace 上限（GB）\n"
        << "  --help                 显示帮助\n\n";
}

static BuildConfig parseArguments(int argc, char* argv[])
{
    BuildConfig config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help")
        {
            printUsage(argv[0]);
            std::exit(0);
        }

        if (i + 1 >= argc)
        {
            throw std::runtime_error("参数缺少值: " + arg);
        }

        const std::string value = argv[++i];

        if (arg == "--onnx")
            config.onnx_path = value;
        else if (arg == "--engine")
            config.engine_path = value;
        else if (arg == "--precision")
            config.precision = value;
        else if (arg == "--calib-dir")
            config.calib_dir = value;
        else if (arg == "--calib-cache")
            config.calib_cache = value;
        else if (arg == "--batch-size")
            config.batch_size = std::stoi(value);
        else if (arg == "--workspace-gb")
            config.workspace_gb =
                static_cast<size_t>(std::stoull(value));
        else
            throw std::runtime_error("未知参数: " + arg);
    }

    if (config.onnx_path.empty() || config.engine_path.empty())
    {
        throw std::runtime_error(
            "--onnx 和 --engine 是必需参数。");
    }

    if (config.precision != "fp32" &&
        config.precision != "fp16" &&
        config.precision != "int8")
    {
        throw std::runtime_error(
            "--precision 必须是 fp32 / fp16 / int8。");
    }

    if (config.batch_size <= 0 || config.workspace_gb == 0)
    {
        throw std::runtime_error(
            "--batch-size 和 --workspace-gb 必须大于 0。");
    }

    if (config.precision == "int8" && config.calib_dir.empty())
    {
        throw std::runtime_error(
            "INT8 模式必须提供 --calib-dir。");
    }

    return config;
}

// 将构建后的 Engine（二进制推理执行计划）保存到磁盘。
static void saveEngine(
    const std::string& path,
    const nvinfer1::IHostMemory& serialized)
{
    const std::filesystem::path output_path(path);

    if (!output_path.parent_path().empty())
    {
        std::filesystem::create_directories(
            output_path.parent_path());
    }

    std::ofstream output(path, std::ios::binary);

    if (!output)
    {
        throw std::runtime_error("无法创建 Engine 文件: " + path);
    }

    output.write(
        static_cast<const char*>(serialized.data()),
        static_cast<std::streamsize>(serialized.size()));

    if (!output)
    {
        throw std::runtime_error("写入 Engine 文件失败: " + path);
    }
}

int main(int argc, char* argv[])
{
    try
    {
        const BuildConfig config = parseArguments(argc, argv);

        if (!std::filesystem::exists(config.onnx_path))
        {
            throw std::runtime_error(
                "ONNX 文件不存在: " + config.onnx_path);
        }

        BuildLogger logger;

        std::cout << "========== Engine 构建配置 ==========\n"
                  << "ONNX:       " << config.onnx_path << '\n'
                  << "Engine:     " << config.engine_path << '\n'
                  << "Precision:  " << config.precision << '\n'
                  << "Workspace:  " << config.workspace_gb << " GB\n";

        // 1. Builder：建厂工程师，负责创建推理执行计划。
        std::unique_ptr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(logger));

        if (!builder)
        {
            throw std::runtime_error("createInferBuilder 失败。");
        }

        // 2. Network：生产线设计图，接收 ONNX 解析出的网络结构。
        const uint32_t flags =
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        std::unique_ptr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(flags));

        if (!network)
        {
            throw std::runtime_error("createNetworkV2 失败。");
        }

        // 3. Parser：读取 ONNX 文件，并将其转换为 TensorRT 网络定义。
        std::unique_ptr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger));

        if (!parser)
        {
            throw std::runtime_error("createParser 失败。");
        }

        std::cout << "\n开始解析 ONNX: "
                  << config.onnx_path << '\n';

        if (!parser->parseFromFile(
                config.onnx_path.c_str(),
                static_cast<int>(
                    nvinfer1::ILogger::Severity::kWARNING)))
        {
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                const auto* error = parser->getError(i);
                if (error)
                {
                    std::cerr << "[ONNX Parser] "
                              << error->desc() << '\n';
                }
            }

            throw std::runtime_error("ONNX 解析失败。");
        }

        std::cout << "ONNX 解析成功。\n";

        // 4. BuilderConfig：设置精度、显存工作区等构建选项。
        std::unique_ptr<nvinfer1::IBuilderConfig> builder_config(
            builder->createBuilderConfig());

        if (!builder_config)
        {
            throw std::runtime_error("createBuilderConfig 失败。");
        }

        const size_t workspace_bytes =
            config.workspace_gb * 1024ULL * 1024ULL * 1024ULL;

        builder_config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            workspace_bytes);

        // 5. 按指定精度配置构建选项。
        std::unique_ptr<Int8Calibrator> calibrator;

        if (config.precision == "fp32")
        {
            std::cout << "使用 FP32 精度。\n";
        }
        else if (config.precision == "fp16")
        {
            if (!builder->platformHasFastFp16())
            {
                std::cerr
                    << "警告：设备未报告快速 FP16 支持，"
                    << "仍尝试设置 FP16 标志。\n";
            }

            builder_config->setFlag(
                nvinfer1::BuilderFlag::kFP16);

            std::cout << "已启用 FP16 构建选项。\n";
        }
        else if (config.precision == "int8")
        {
            if (!builder->platformHasFastInt8())
            {
                std::cerr
                    << "警告：设备未报告快速 INT8 支持，"
                    << "仍尝试设置 INT8 标志。\n";
            }

            builder_config->setFlag(
                nvinfer1::BuilderFlag::kINT8);

            // Int8Calibrator 必须与 int8_calibrator.h/.cpp 的接口一致。
            calibrator = std::make_unique<Int8Calibrator>(
                config.batch_size,
                config.input_width,
                config.input_height,
                config.calib_dir,
                config.calib_cache);

            builder_config->setInt8Calibrator(calibrator.get());

            std::cout << "已启用 INT8 构建选项。\n"
                      << "校准图片目录: " << config.calib_dir << '\n'
                      << "校准缓存文件: " << config.calib_cache << '\n';
        }

        // 6. 构建并序列化 Engine。
        // Engine（已构建的生产线）：供推理程序加载执行。
        std::cout << "\n开始构建 Engine，耗时取决于模型和硬件...\n";

        std::unique_ptr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(
                *network,
                *builder_config));

        if (!serialized)
        {
            throw std::runtime_error(
                "buildSerializedNetwork 失败，未生成 Engine。");
        }

        // 7. 保存序列化结果。
        saveEngine(config.engine_path, *serialized);

        std::cout << "\nEngine 构建成功。\n"
                  << "输出路径: " << config.engine_path << '\n'
                  << "文件大小: "
                  << serialized->size() / (1024.0 * 1024.0)
                  << " MB\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n[BuildEngine Error] "
                  << e.what() << '\n';
        return 1;
    }
}
/*
1. Parser 干什么？

nvonnxparser::createParser(*network, logger) 创建的 Parser 读取 .onnx 文件，把 Conv、Concat、Sigmoid 等算子翻译成 TensorRT 的层，填进 network。失败时你的代码用 getNbErrors() 和 getError(i)->desc() 打印原因，常见的是算子不支持或 opset 不兼容。

2. Network 是什么？

INetworkDefinition 是网络结构的内存表示：有哪些层、张量怎么连、输入输出是什么。它只描述"算什么"，还没决定"怎么在 GPU 上算最快"。kEXPLICIT_BATCH 表示 batch 维写在张量形状里。TensorRT 10 里这个标志已弃用（显式 batch 成为默认），旧写法会有警告，但一般不影响构建。

3. Builder 是什么？

IBuilder 是构建的总入口，负责层融合、对多种 CUDA kernel 试跑计时并挑最快的，最终产出优化后的执行计划。这一步很慢，所以只在离线做一次。

4. BuilderConfig 是什么？

builder->createBuilderConfig() 创建的 IBuilderConfig 保存构建规则：精度标志、workspace 上限、INT8 校准器、optimization profile 等。你的代码在这里设置了 workspace，并在后面按精度追加 flag。

5. FP32 / FP16 / INT8 在哪里选择？

在第 5 步的 if / else if 分支里，通过 BuilderConfig 的 flag 选择：

cpp
builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);

FP32 是默认值，不用设置。设了 FP16 或 INT8 后，TensorRT 会逐层选更快的低精度 kernel，不支持的层自动回退到更高精度。platformHasFastFp16() 和 platformHasFastInt8() 只是检查硬件是否支持，你的代码只给警告、不阻止构建。

6. Workspace 是什么？

构建阶段尝试各种 kernel 实现所需的临时 GPU 显存上限：

cpp
builder_config->setMemoryPoolLimit(
    nvinfer1::MemoryPoolType::kWORKSPACE,
    workspace_bytes);

你用 workspace_gb * 1024ULL * 1024ULL * 1024ULL 换算成字节，写法正确。上限太小会让部分更快的算法无法使用，影响性能甚至构建失败。

7. Calibration 在哪里接入？

在 INT8 分支里：先 setFlag(kINT8)，再创建校准器，最后 setInt8Calibrator(calibrator.get())。校准器放在 main 作用域内的 unique_ptr 里，生命周期覆盖了 buildSerializedNetwork，这点是对的。构建时 TensorRT 会回调 readCalibrationCache，有缓存就直接用，没有就反复调用 getBatch 统计，最后调用 writeCalibrationCache。

8. 为什么 Engine 生成后运行时可以直接加载？

buildSerializedNetwork 返回的 IHostMemory 已包含优化后的网络、选定的 kernel 和权重，saveEngine 把它写成文件。运行时用 deserializeCudaEngine 反序列化即可，不需要再解析 ONNX，也不需要再耗时选 kernel。Engine 与 GPU 型号、TensorRT 和 CUDA 版本绑定，换环境要重新构建。

代码里需要注意的问题

1. 校准器类名和构造函数与前面那个文件不一致（会导致编译错误）。
你之前贴的 int8_calibrator.cpp 里，类名是 YoloInt8Calibrator，构造参数顺序是 (image_dir, cache_file, batch_size, input_width, input_height)。而这里用的是 Int8Calibrator，参数顺序是 (batch_size, input_width, input_height, calib_dir, calib_cache)。需要统一，例如改成：

cpp
std::unique_ptr<YoloInt8Calibrator> calibrator;

calibrator = std::make_unique<YoloInt8Calibrator>(
    config.calib_dir,
    config.calib_cache,
    config.batch_size,
    config.input_width,
    config.input_height);

2. 没有 --input-width / --input-height 参数。
校准尺寸永远是 640×640。如果模型输入不是 640，校准预处理和模型输入就对不上。

3. 校准 batch size 要和模型输入的 batch 维一致。
如果 ONNX 导出时 batch 固定为 1，而 --batch-size 8，校准会出错。如果 ONNX 是动态 batch，还需要设置 optimization profile 和 setCalibrationProfile，当前代码没有处理。

4. 校准图片数量要不少于 batch_size。
getBatch 在剩余图片不足一个 batch 时返回 false，图片太少会导致校准不充分。
*/
