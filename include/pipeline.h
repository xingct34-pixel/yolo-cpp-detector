#pragma once

#include "benchmark.h"
#include "detector.h"
#include "thread_safe_queue.h"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <exception>
#include <mutex>
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

    void requestStop(std::exception_ptr error = nullptr);

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
,模型加载失败在 run() 之前就能暴露。
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
     /*
     8. stop_requested_ 是干什么的?
      它是跨线程的停止信号:视频读完、用户按键退出或出错时置为 true,各线程循环检查它然后退出。用 std::atomic<bool> 是为了避免多线程读写同一个普通 bool 带来的数据竞争。
      它和队列的 close() 配合使用:标志告诉线程"该停了",close() 负责唤醒正在队列上阻塞等待的线程。
     */

    // 保存第一个工作线程异常，主线程 join 后统一重新抛出。
    std::mutex worker_exception_mutex_;
    std::exception_ptr worker_exception_;

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

    // 模型运行时和各阶段的独立统计。
    Benchmark benchmark_;
    Benchmark image_clone_benchmark_;
    Benchmark preprocess_benchmark_;
    Benchmark resize_benchmark_;
    Benchmark padding_benchmark_;
    Benchmark tensor_prepare_benchmark_;
    Benchmark backend_call_benchmark_;
    Benchmark postprocess_benchmark_;
};
