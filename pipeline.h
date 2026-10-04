#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "detector.h"
#include "thread_safe_queue.h"

// 读线程产生的数据
struct FrameData {
    int frame_id;
    cv::Mat frame;
};

// 推理线程产生的数据
struct ResultData {
    int frame_id;
    cv::Mat frame;
};

class Pipeline {
public:
    Pipeline(const std::string& video_path,
             const std::string& model_path,
             const std::string& classes_path);

    void run();

private:
    // 三个线程分别负责三个阶段
    void readLoop();
    void inferenceLoop();
    void display();

private:
    long long read_count_ = 0;
    long long inference_count_ = 0;
    long long display_count_ = 0;
    // 帧队列：读线程 → 推理线程
    ThreadSafeQueue<FrameData> frame_queue_;

    // 结果队列：推理线程 → 显示线程
    ThreadSafeQueue<ResultData> result_queue_;

    // 真正负责 YOLO 推理
    Detector detector_;

    // 输入视频
    std::string video_path_;
};
