#pragma once

#include <atomic>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "detector.h"
#include "thread_safe_queue.h"

/*
 * ============================================================
 * Pipeline
 *
 * 功能：
 *
 * 将视频推理拆成多个阶段：
 *
 * ┌────────────┐
 * │ Read       │ 读取视频
 * └─────┬──────┘
 *       ↓
 * ┌────────────┐
 * │ FrameQueue │
 * └─────┬──────┘
 *       ↓
 * ┌────────────┐
 * │ Inference  │ 推理
 * └─────┬──────┘
 *       ↓
 * ┌────────────┐
 * │ ResultQueue│
 * └─────┬──────┘
 *       ↓
 * ┌────────────┐
 * │ Display    │ 显示结果
 * └────────────┘
 *
 * 通过流水线并行，提高整个系统的吞吐能力。
 * ============================================================
 */

// ==================== 帧数据结构 ====================

struct FrameData
{
    int frame_id;
    cv::Mat frame;
};

// ==================== 推理结果数据结构 ====================

struct ResultData
{
    int frame_id;
    cv::Mat frame;
};

class Pipeline
{
public:

    // ==================== Pipeline 初始化 ====================

    Pipeline(
        const std::string& video_path,
        const std::string& model_path,
        const std::string& classes_path
    );

    // ==================== Pipeline 主控制 ====================

    void run();

private:

    // ==================== 读取线程 ====================

    void readLoop();

    // ==================== 推理线程 ====================

    void inferenceLoop();

    // ==================== 显示模块 ====================

    void display();

private:

    // ==================== Pipeline 数据流 ====================

    ThreadSafeQueue<FrameData> frame_queue_;

    ThreadSafeQueue<ResultData> result_queue_;

    // ==================== 推理模块 ====================

    Detector detector_;

    // ==================== 输入数据 ====================

    std::string video_path_;

    // ==================== 性能统计 ====================

    std::atomic<long long> read_count_{0};

    std::atomic<long long> inference_count_{0};

    std::atomic<long long> display_count_{0};
};
