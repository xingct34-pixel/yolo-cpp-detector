// YOLO 检测部署 | src/pipeline.cpp
// 每帧视频在这里经历读取、检测、输出。读取端复制当前画面送入第一条队列；
// 推理端取出后调用 Detector，结果进入第二条队列，消费端取走结果做显示或统计。
// 读并发路径时跟着同一帧看两次 push/pop，再看视频读完后 close 怎样一路传下去。


#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <stdexcept>


// 模块：统计辅助：计算毫秒差并写入阶段统计


namespace {
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void stats(cv::FileStorage& fs, const char* name, const Benchmark& b) {
    fs << name << "{" << "samples" << double(b.count()) << "mean_ms" << b.average() << "p50_ms"
       << b.p50() << "p95_ms" << b.p95() << "}";
}
}


// 模块：流水线初始化：创建两个有界队列和唯一检测器
// 数据流：读取线程 -> frame_queue_ -> 单推理线程 -> result_queue_ -> 主线程消费。
// detector_在Pipeline构造时加载一次；容量限制待处理帧数，不会提高最慢阶段的处理速度。


Pipeline::Pipeline(const std::string& video,
                   const std::string& model,
                   const std::string& classes,
                   Detector::BackendType type,
                   const std::string& engine,
                   bool cuda,
                   size_t capacity,
                   PipelineOptions options)
    : frame_queue_(capacity, options.policy), result_queue_(capacity, options.policy),
      detector_(model, classes, type, engine, cuda), video_path_(video),
      options_(std::move(options)) {
}


// 模块：异常与取消：保存错误，关闭队列并唤醒等待者
// 异常/取消走快速停止：先记住首个异常、关闭并唤醒两个队列，再由run()汇合线程。
// 与正常EOF逐级排空不同，此路径不保证处理完所有已读帧。


void Pipeline::requestStop(std::exception_ptr error) {
    if (error) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_)
            error_ = error;
    }
    stop_requested_ = true;
    frame_queue_.close();
    result_queue_.close();
}


// 模块：读取阶段：解码视频，深拷贝当前帧后交给输入队列


void Pipeline::readLoop() {
    cv::VideoCapture cap(video_path_);
    if (!cap.isOpened())
        throw std::runtime_error("视频打开失败：" + video_path_);
    cv::Mat image;
    int id = 0;
    while (!stop_requested_ && (!options_.max_frames || id < options_.max_frames) &&
           cap.read(image)) {
        // VideoCapture会复用image缓冲，clone使入队帧拥有独立像素；move只转交对象，不替代深拷贝。
        FrameData data{id++, image.clone(), Clock::now()};
        ++read_count_;
        if (!frame_queue_.push(std::move(data)))
            break;
    }
    // 正常读完只关闭输入队列：拒绝新帧，但推理线程仍可取走队列里的剩余帧。
    frame_queue_.close();
}


// 模块：阶段采样：收集检测器最近一帧的耗时


void Pipeline::collect() {
    inference_.record(detector_.last_inference_ms());
    preprocess_.record(detector_.last_preprocess_ms());
    backend_.record(detector_.last_backend_call_ms());
    postprocess_.record(detector_.last_postprocess_ms());
}


// 模块：推理阶段：出队后在锁外检测，再投递结果


void Pipeline::inferenceLoop() {
    FrameData data;
    // pop返回前已释放队列锁，耗时detect在锁外执行，读取线程可继续入队。
    while (!stop_requested_ && frame_queue_.pop(data)) {
        queue_wait_.record(ms(data.captured, Clock::now()));
        auto image = detector_.detect(data.frame);
        ++inference_count_;
        collect();
        if (!result_queue_.push({data.frame_id, std::move(image), data.captured}))
            break;
    }
    // 输入排空后关闭结果队列，让主线程消费完最后的结果并退出。
    result_queue_.close();
}


// 模块：结果消费：可选显示图片并记录端到端耗时


void Pipeline::consume(ResultData& result) {
    if (options_.show) {
        cv::imshow("YOLO Result", result.frame);
        if (cv::waitKey(1) == 27)
            requestStop();
    }
    // 端到端计时从解码并clone完成后的captured开始，含排队、推理及消费，不含此前读取。
    ++display_count_;
    end_to_end_.record(ms(result.captured, Clock::now()));
}


// 模块：消费循环：读取结果队列直到结束或取消


void Pipeline::display() {
    ResultData result;
    while (!stop_requested_ && result_queue_.pop(result))
        consume(result);
}


// 模块：运行入口：预热，组织顺序模式或多线程模式


void Pipeline::run() {
    if (options_.warmup < 0 || options_.max_frames < 0)
        throw std::invalid_argument("invalid pipeline limits");


    // 模块：1. 预热：初始化计算路径，不纳入正式统计
    // 预热复用同一模型，但不record到阶段统计，也不计入下面的整体运行时间。


    for (int i = 0; i < options_.warmup; ++i)
        (void)detector_.detect(cv::Mat(640, 640, CV_8UC3, cv::Scalar{}));


    // 模块：2. 正式运行：启动监视线程并选择执行模式


    auto begin = Clock::now();
    std::thread reader, infer, monitor;
    std::atomic<bool> finished{false};
    try {
        if (options_.external_stop)
            monitor = std::thread([&] {
                while (!finished) {
                    if (options_.external_stop->load()) {
                        requestStop();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            });
        if (options_.sequential) {
            cv::VideoCapture cap(video_path_);
            if (!cap.isOpened())
                throw std::runtime_error("视频打开失败：" + video_path_);
            cv::Mat frame;
            int id = 0;
            while (!stop_requested_ && (!options_.max_frames || id < options_.max_frames) &&
                   cap.read(frame)) {
                auto captured = Clock::now();
                ++read_count_;
                auto result = detector_.detect(frame);
                ++inference_count_;
                collect();
                queue_wait_.record(0);
                ResultData data{id++, std::move(result), captured};
                consume(data);
            }
        } else {
            reader = std::thread([&] {
                try {
                    readLoop();
                } catch (...) {
                    requestStop(std::current_exception());
                }
            });
            infer = std::thread([&] {
                try {
                    inferenceLoop();
                } catch (...) {
                    requestStop(std::current_exception());
                }
            });
            display();
        }
    } catch (...) {
        requestStop(std::current_exception());
    }


    // 模块：3. 线程回收：等待退出并重新抛出首个异常
    // join只等待线程结束，不会主动停止线程；异常时前面必须先关闭队列解除等待。


    if (reader.joinable())
        reader.join();
    if (infer.joinable())
        infer.join();
    finished = true;
    if (monitor.joinable())
        monitor.join();
    if (options_.show)
        cv::destroyAllWindows();
    if (error_)
        std::rethrow_exception(error_);


    // 模块：4. 汇总报告：计算实际输出FPS和各阶段指标
    // 实际输出FPS=输出帧数/整段运行秒数；与1000/平均Run毫秒数的计时范围不同。


    double seconds = ms(begin, Clock::now()) / 1000.;
    std::cout << "读取帧数：" << read_count_ << "\n推理帧数：" << inference_count_ << "\n输出帧数："
              << display_count_ << "\nPipeline FPS："
              << (seconds > 0 ? display_count_ / seconds : 0) << "\n丢弃输入/结果帧："
              << frame_queue_.dropped() << "/" << result_queue_.dropped() << '\n';
    inference_.print(detector_.backend_name().c_str());
    if (!options_.report.empty()) {
        cv::FileStorage fs(options_.report, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened())
            throw std::runtime_error("cannot write report: " + options_.report);
        fs << "schema_version" << 1 << "mode" << (options_.sequential ? "sequential" : "pipeline")
           << "queue_policy" << (options_.policy == QueuePolicy::Block ? "block" : "drop-oldest")
           << "backend" << detector_.backend_name() << "show" << int(options_.show) << "warmup"
           << options_.warmup << "stopped" << int(stop_requested_.load()) << "read_frames"
           << double(read_count_) << "inferred_frames" << double(inference_count_)
           << "output_frames" << double(display_count_) << "input_dropped"
           << double(frame_queue_.dropped()) << "result_dropped" << double(result_queue_.dropped())
           << "input_queue_peak" << double(frame_queue_.peak()) << "result_queue_peak"
           << double(result_queue_.peak()) << "elapsed_seconds" << seconds << "output_fps"
           << (seconds > 0 ? display_count_ / seconds : 0);
        stats(fs, "inference", inference_);
        stats(fs, "preprocess", preprocess_);
        stats(fs, "backend_api", backend_);
        stats(fs, "postprocess", postprocess_);
        stats(fs, "queue_wait", queue_wait_);
        stats(fs, "end_to_end", end_to_end_);
    }
}
