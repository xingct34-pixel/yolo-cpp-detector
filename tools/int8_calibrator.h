#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

class YoloInt8Calibrator
    : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    YoloInt8Calibrator(
        const std::string& image_dir,
        const std::string& cache_file,
        int batch_size = 1,
        int input_width = 640,
        int input_height = 640);

    ~YoloInt8Calibrator() override;

    int getBatchSize() const noexcept override;

    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings) noexcept override;

    const void* readCalibrationCache(
        std::size_t& length) noexcept override;

    void writeCalibrationCache(
        const void* cache,
        std::size_t length) noexcept override;

private:
    void loadImagePaths(
        const std::string& image_dir);

    bool preprocessImage(
        const std::string& image_path,
        float* output);

private:
    std::vector<std::string>
        image_paths_;

    std::string cache_file_;

    std::vector<char>
        calibration_cache_;

    std::vector<float>
        host_batch_;

    void* device_input_ = nullptr;

    size_t current_index_ = 0;

    int batch_size_ = 1;
    int input_width_ = 640;
    int input_height_ = 640;

    size_t image_elements_ = 0;
    size_t batch_elements_ = 0;
};
