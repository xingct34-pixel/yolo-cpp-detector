#pragma once

#include "benchmark.h"
#include "detector.h"
#include "thread_safe_queue.h"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <string>

// ==================== Pipeline 数据结构模块 ====================

struct FrameData
{
    int frame_id = 0;
    cv::Mat frame;
};

struct ResultData
{
    int frame_id = 0;
    cv::Mat frame;
};

// ==================== Pipeline 主流程模块 ====================

class Pipeline
{
public:
    Pipeline(
        const std::string& video_path,
        const std::string& model_path,
        const std::string& classes_path,
        Detector::BackendType backend_type,
        const std::string& engine_path = "",
        bool use_cuda = false,
        size_t queue_size = 3);

    ~Pipeline() = default;

    // ==================== Pipeline 启动模块 ====================

    void run();

private:
    // ==================== 读取线程模块 ====================

    void readLoop();

    // ==================== 推理线程模块 ====================

    void inferenceLoop();

    // ==================== 显示模块 ====================

    void display();

private:
    // ==================== Pipeline 数据通道 ====================

    ThreadSafeQueue<FrameData>
        frame_queue_;

    ThreadSafeQueue<ResultData>
        result_queue_;

    // ==================== 模型推理模块 ====================

    Detector detector_;

    // ==================== 视频配置模块 ====================

    std::string video_path_;

    // ==================== 停止控制模块 ====================
    //
    // true：
    // Pipeline 收到停止信号。
    //
    // false：
    // Pipeline 正常运行。

    std::atomic<bool> stop_requested_{
        false
    };

    // ==================== 帧计数模块 ====================

    std::atomic<long long> read_count_{
        0
    };

    std::atomic<long long> inference_count_{
        0
    };

    std::atomic<long long> display_count_{
        0
    };

    // ==================== 性能统计模块 ====================

    Benchmark benchmark_;
};
