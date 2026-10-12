#include "int8_calibrator.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs =
    std::filesystem;

namespace
{

bool isImageFile(
    const fs::path& path)
{
    if (!path.has_extension())
    {
        return false;
    }

    std::string ext =
        path.extension().string();

    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(
                std::tolower(c));
        });

    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png" ||
           ext == ".bmp";
}

}

YoloInt8Calibrator::YoloInt8Calibrator(
    const std::string& image_dir,
    const std::string& cache_file,
    int batch_size,
    int input_width,
    int input_height)
    : cache_file_(cache_file),
      batch_size_(batch_size),
      input_width_(input_width),
      input_height_(input_height)
{
    if (batch_size_ <= 0)
    {
        throw std::invalid_argument(
            "batch_size 必须大于 0。");
    }

    if (input_width_ <= 0 ||
        input_height_ <= 0)
    {
        throw std::invalid_argument(
            "输入尺寸必须大于 0。");
    }

    image_elements_ =
        static_cast<size_t>(3) *
        static_cast<size_t>(
            input_width_) *
        static_cast<size_t>(
            input_height_);

    batch_elements_ =
        static_cast<size_t>(
            batch_size_) *
        image_elements_;

    host_batch_.resize(
        batch_elements_);

    loadImagePaths(image_dir);

    if (image_paths_.empty())
    {
        throw std::runtime_error(
            "INT8 校准目录中没有找到图片: " +
            image_dir);
    }

    cudaError_t err =
        cudaMalloc(
            &device_input_,
            batch_elements_ *
                sizeof(float));

    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "INT8 calibrator cudaMalloc failed: " +
            std::string(
                cudaGetErrorString(err)));
    }

    std::cout
        << "INT8 校准图片数量: "
        << image_paths_.size()
        << std::endl;
}

YoloInt8Calibrator::~YoloInt8Calibrator()
{
    if (device_input_)
    {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
}

void YoloInt8Calibrator::loadImagePaths(
    const std::string& image_dir)
{
    if (!fs::exists(image_dir))
    {
        throw std::runtime_error(
            "INT8 校准目录不存在: " +
            image_dir);
    }

    for (const auto& entry :
         fs::directory_iterator(image_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (isImageFile(
                entry.path()))
        {
            image_paths_.push_back(
                entry.path().string());
        }
    }

    std::sort(
        image_paths_.begin(),
        image_paths_.end());
}

bool YoloInt8Calibrator::preprocessImage(
    const std::string& image_path,
    float* output)
{
    cv::Mat image =
        cv::imread(
            image_path,
            cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr
            << "无法读取校准图片: "
            << image_path
            << std::endl;

        return false;
    }

    const float scale =
        std::min(
            static_cast<float>(
                input_width_) /
                static_cast<float>(
                    image.cols),
            static_cast<float>(
                input_height_) /
                static_cast<float>(
                    image.rows));

    const int new_width =
        static_cast<int>(
            image.cols * scale);

    const int new_height =
        static_cast<int>(
            image.rows * scale);

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(
            new_width,
            new_height));

    cv::Mat letterbox(
        input_height_,
        input_width_,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114));

    const int pad_x =
        (input_width_ -
         new_width) /
        2;

    const int pad_y =
        (input_height_ -
         new_height) /
        2;

    resized.copyTo(
        letterbox(
            cv::Rect(
                pad_x,
                pad_y,
                new_width,
                new_height)));

    cv::Mat rgb;

    cv::cvtColor(
        letterbox,
        rgb,
        cv::COLOR_BGR2RGB);

    cv::Mat float_image;

    rgb.convertTo(
        float_image,
        CV_32FC3,
        1.0 / 255.0);

    std::vector<cv::Mat>
        channels(3);

    cv::split(
        float_image,
        channels);

    const size_t plane_size =
        static_cast<size_t>(
            input_width_) *
        static_cast<size_t>(
            input_height_);

    for (int c = 0;
         c < 3;
         ++c)
    {
        const float* src =
            channels[c].ptr<float>();

        float* dst =
            output +
            static_cast<size_t>(c) *
                plane_size;

        std::copy(
            src,
            src + plane_size,
            dst);
    }

    return true;
}

int YoloInt8Calibrator::getBatchSize()
    const noexcept
{
    return batch_size_;
}

bool YoloInt8Calibrator::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings) noexcept
{
    if (nbBindings != 1 ||
        bindings == nullptr)
    {
        return false;
    }

    if (current_index_ +
            static_cast<size_t>(
                batch_size_) >
        image_paths_.size())
    {
        return false;
    }

    for (int i = 0;
         i < batch_size_;
         ++i)
    {
        float* destination =
            host_batch_.data() +
            static_cast<size_t>(i) *
                image_elements_;

        if (!preprocessImage(
                image_paths_[
                    current_index_ +
                    static_cast<size_t>(i)],
                destination))
        {
            return false;
        }
    }

    cudaError_t err =
        cudaMemcpy(
            device_input_,
            host_batch_.data(),
            batch_elements_ *
                sizeof(float),
            cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        return false;
    }

    bindings[0] =
        device_input_;

    current_index_ +=
        static_cast<size_t>(
            batch_size_);

    return true;
}

const void*
YoloInt8Calibrator::readCalibrationCache(
    std::size_t& length) noexcept
{
    calibration_cache_.clear();

    std::ifstream input(
        cache_file_,
        std::ios::binary);

    if (!input)
    {
        length = 0;
        return nullptr;
    }

    input.seekg(
        0,
        std::ios::end);

    const std::streamsize size =
        input.tellg();

    input.seekg(
        0,
        std::ios::beg);

    if (size <= 0)
    {
        length = 0;
        return nullptr;
    }

    calibration_cache_.resize(
        static_cast<size_t>(size));

    input.read(
        calibration_cache_.data(),
        size);

    length =
        calibration_cache_.size();

    std::cout
        << "读取 INT8 Calibration Cache: "
        << cache_file_
        << std::endl;

    return calibration_cache_.data();
}

void YoloInt8Calibrator::writeCalibrationCache(
    const void* cache,
    std::size_t length) noexcept
{
    if (cache == nullptr ||
        length == 0)
    {
        return;
    }

    std::ofstream output(
        cache_file_,
        std::ios::binary);

    if (!output)
    {
        std::cerr
            << "无法写入 Calibration Cache: "
            << cache_file_
            << std::endl;

        return;
    }

    output.write(
        static_cast<const char*>(
            cache),
        static_cast<std::streamsize>(
            length));

    std::cout
        << "写入 INT8 Calibration Cache: "
        << cache_file_
        << std::endl;
}
/*
⑭ 为什么 INT8 不能简单地把 FP32 改成 INT8？

FP32 有约 ±3.4×10³⁸ 的动态范围和很高的精度，INT8 只有 256 个离散值（-128~127）。要把浮点数映射到整数，必须先知道"这一层的值大概落在多大范围内"：

q = clamp(round(x / scale), -128, 127)
scale ≈ 激活值范围 / 127
权重是固定的，范围可以直接统计。
激活值取决于输入数据，只有拿真实数据跑一遍才知道范围。
范围取小了，大值被截断；取大了，小值全被量化成同一个整数，精度损失严重。

所以流程是：校准数据 → 前向推理 → 统计每层激活分布 → 选出 scale（TensorRT 默认用 entropy 校准，通过 KL 散度选截断阈值）→ 生成 INT8 Engine。

⑮ 对应你代码的 6 个问题

1. Calibration Dataset 有什么作用？
它是 TensorRT 统计各层激活分布的"样本"。loadImagePaths 收集目录里的图片，getBatch 每次喂一个 batch，TensorRT 在构建 engine 时用这些数据跑 FP32 前向并记录每层的数值分布。

2. 为什么需要真实、有代表性的图片？
scale 完全由这批数据的分布决定。如果校准图和实际部署场景差异大（光照、目标大小、背景不同），算出的范围就会偏，部署后精度明显下降。随机噪声图或全黑图会让 scale 严重失真。一般用几百张覆盖实际场景的图即可。

另外，preprocessImage 里的 letterbox、BGR→RGB、除以 255、HWC→CHW，必须和推理时的预处理完全一致，否则统计的是另一种分布。

3. 为什么 GPU 上需要 calibration buffer？
TensorRT 在 GPU 上执行校准用的前向推理，要求输入张量在显存中。device_input_ 就是这块缓冲区，在构造函数里用 cudaMalloc 一次性分配 batch_size × 3 × H × W × sizeof(float)，每个 batch 复用，避免反复申请释放。bindings[0] = device_input_ 就是把这块显存地址告诉 TensorRT。

4. 为什么需要 H2D？
OpenCV 读图和预处理都在 CPU 内存里（host_batch_），而 GPU 不能直接访问它。cudaMemcpy(..., cudaMemcpyHostToDevice) 把一个 batch 的数据从主机拷到 device_input_，TensorRT 才能读到。

5. Calibration Cache 是干什么的？
它保存校准的结果，即每个张量对应的量化 scale。writeCalibrationCache 在校准完成后由 TensorRT 回调写入文件，readCalibrationCache 在构建时回调读取。

6. 为什么下次可以直接读取缓存？
校准是最耗时的一步（要读图、预处理、跑很多个 batch）。scale 只取决于网络结构和校准数据，这两者不变，结果就不变。readCalibrationCache 若读到有效缓存，返回指针和长度，TensorRT 就跳过 getBatch，直接用缓存里的 scale 构建 INT8 engine，几秒内完成。

需要注意：如果更换了模型、修改了网络结构、预处理方式或校准数据，要删除旧缓存重新校准，否则会沿用过期的 scale。
*/
