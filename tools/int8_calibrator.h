// YOLO 检测部署 | tools/int8_calibrator.h
// TensorRT 构建工具通过这个接口向校准器索取成批输入，也可以读写校准缓存。
// 构建器决定何时要下一批，具体怎样找图、预处理和送入显存在同名 .cpp 中。
//
// 阅读地图（按源码顺序）
//   1 校准器接口


#pragma once

#include <NvInfer.h>  // TensorRT 构建网络和执行推理的类型
#include <cuda_runtime_api.h>  // 操作 CUDA 缓冲区、异步复制和执行流

#include <cstddef>
#include <string>
#include <vector>  // 元素连续存放，适合装浮点张量或一组检测记录


// 模块 1：校准器接口
// 功能：为TensorRT提供输入批次及校准缓存（实验路径）


class YoloInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    YoloInt8Calibrator(const std::string& image_dir,
                       const std::string& cache_file,
                       int batch_size = 1,
                       int input_width = 640,
                       int input_height = 640);

    ~YoloInt8Calibrator() override;

    int getBatchSize() const noexcept override;

    // TensorRT 向 getBatch 索取下一批。返回 true 时，bindings 要能指到这批输入的设备内存。
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    // length 是引用：返回缓存地址的同时，也要把这个地址后面有多少字节写回调用者。
    const void* readCalibrationCache(std::size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, std::size_t length) noexcept override;

private:
    void loadImagePaths(const std::string& image_dir);

    bool preprocessImage(const std::string& image_path, float* output);

private:
    std::vector<std::string> image_paths_;

    std::string cache_file_;

    std::vector<char> calibration_cache_;

    std::vector<float> host_batch_;

    void* device_input_ = nullptr;

    size_t current_index_ = 0;

    int batch_size_ = 1;
    int input_width_ = 640;
    int input_height_ = 640;

    size_t image_elements_ = 0;
    size_t batch_elements_ = 0;
};
