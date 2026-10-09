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


    void run();

private:

    void readLoop();


    void inferenceLoop();


    void display();

private:
    // ==================== Pipeline 数据通道 ====================

    ThreadSafeQueue<FrameData>
        frame_queue_;

    ThreadSafeQueue<ResultData>
        result_queue_;

/*为什么 Detector 是 Pipeline 的成员?

模型加载很慢,必须只做一次,放在构造阶段,而不是每帧加载。
它的生命周期应该和 Pipeline 一致:Pipeline 存在,模型就在;Pipeline 析构,资源由 RAII 自动释放。
构造函数里的 model_path、backend_type、engine_path、use_cuda 等参数都是转交给 Detector 的,模型加载失败在 run() 之前就能暴露。
它只被推理线程使用,不需要对外共享。
*/



    Detector detector_;


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
