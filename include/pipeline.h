#pragma once
#include "benchmark.h"
#include "detector.h"
#include "thread_safe_queue.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
struct PipelineOptions {
    bool show=false,sequential=false;
    int max_frames=0,warmup=5;
    std::string report;
    QueuePolicy policy=QueuePolicy::Block;
    std::atomic<bool>* external_stop=nullptr;
};
struct FrameData {int frame_id=0;cv::Mat frame;std::chrono::steady_clock::time_point captured;};
struct ResultData {int frame_id=0;cv::Mat frame;std::chrono::steady_clock::time_point captured;};
class Pipeline {
public:
    Pipeline(const std::string&,const std::string&,const std::string&,Detector::BackendType,
        const std::string& engine="",bool cuda=false,size_t queue_size=3,PipelineOptions options={});
    void run();
private:
    void requestStop(std::exception_ptr error=nullptr);
    void readLoop();void inferenceLoop();void display();void collect();void consume(ResultData&);
    ThreadSafeQueue<FrameData> frame_queue_;
    ThreadSafeQueue<ResultData> result_queue_;
    Detector detector_;
    std::string video_path_;
    PipelineOptions options_;
    std::atomic<bool> stop_requested_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;
    long long read_count_=0,inference_count_=0,display_count_=0;
    Benchmark inference_,preprocess_,backend_,postprocess_,queue_wait_,end_to_end_;
};
