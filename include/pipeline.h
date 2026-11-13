// YOLO 检测部署 | include/pipeline.h
// 声明视频处理的三个角色和它们传递的数据。帧编号、图片与采集时间一起向下游走，
// 这样显示结果时还能算出这张图从采集完成到输出经历了多久。
// 两个 ThreadSafeQueue 分别连接读取与推理、推理与结果消费。执行顺序见 pipeline.cpp。


#pragma once
#include "benchmark.h"
#include "detector.h"
#include "thread_safe_queue.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>


// 模块：运行配置：选择显示、预热、队列策略及外部停止标志


struct PipelineOptions {
    bool show = false, sequential = false;
    int max_frames = 0, warmup = 5;
    std::string report;
    QueuePolicy policy = QueuePolicy::Block;
    std::atomic<bool>* external_stop = nullptr;
};


// 模块：输入记录：携带帧编号、独立图片和采集完成时间


struct FrameData {
    int frame_id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured;
};


// 模块：输出记录：携带检测图片并保留原始帧时间


struct ResultData {
    int frame_id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured;
};


// 模块：Pipeline接口：组织读取、推理和结果消费三个角色


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


    // 模块：内部状态：队列、检测器、退出控制和阶段统计


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
