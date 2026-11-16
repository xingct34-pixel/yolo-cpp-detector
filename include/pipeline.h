// YOLO 检测部署 | include/pipeline.h
// 功能：声明视频处理的三个角色和它们传递的数据。帧编号、图片与采集时间一起向下游走，
// 这样显示结果时还能算出这张图从采集完成到输出经历了多久。
// 两个 ThreadSafeQueue 分别连接读取与推理、推理与结果消费。执行顺序见 pipeline.cpp。
// 目的：把选项和跨线程传递的数据定义在一起，阅读实现前先辨清每条队列里装的类型。


#pragma once
#include "benchmark.h"  // 把多次计时样本汇总成均值和分位数
#include "detector.h"  // 单帧检测入口，调用它即可走完预处理、推理和绘图
#include "thread_safe_queue.h"  // 线程之间通过这个有界队列传递数据
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>


// 模块：运行配置（PipelineOptions）
// 功能：选择显示、预热、队列策略及外部停止标志
struct PipelineOptions {
    bool show = false, sequential = false;
    int max_frames = 0, warmup = 5;
    std::string report;
    QueuePolicy policy = QueuePolicy::Block;
    std::atomic<bool>* external_stop = nullptr;
};


// 模块：输入记录（FrameData）
// 功能：携带帧编号、独立图片和采集完成时间
// 只有图片本身还算不出等待时间。把采集时刻和图片放在同一个 FrameData 中，推理端取到这帧时才知道该减哪个时刻；frame_id 用来认出它是哪一帧。
// 读取线程记录捕获时刻。传给推理线程的不仅有像素，还有之后计算等待时间所需的时钟值。
struct FrameData {
    int frame_id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured;
};


// 模块：输出记录（ResultData）
// 功能：携带检测图片并保留原始帧时间
// 结果保留原帧编号。若允许丢帧，编号仍能说明画面来自哪一帧。
struct ResultData {
    int frame_id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured;
};


// 模块：Pipeline接口
// 功能：组织读取、推理和结果消费三个角色
class Pipeline {
public:
    Pipeline(const std::string&,
             const std::string&,
             const std::string&,
             Detector::BackendType,
             const std::string& engine = "",
             bool cuda = false,
             size_t queue_size = 3,
             PipelineOptions options = {});
    void run();

private:
    void requestStop(std::exception_ptr error = nullptr);
    void readLoop();
    void inferenceLoop();
    void display();
    void collect();
    void consume(ResultData&);


    // 内部状态
    // 功能：队列、检测器、退出控制和阶段统计
    ThreadSafeQueue<FrameData> frame_queue_;
    ThreadSafeQueue<ResultData> result_queue_;
    Detector detector_;
    std::string video_path_;
    PipelineOptions options_;
    std::atomic<bool> stop_requested_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;
    long long read_count_ = 0, inference_count_ = 0, display_count_ = 0;
    Benchmark inference_, preprocess_, backend_, postprocess_, queue_wait_, end_to_end_;
};
