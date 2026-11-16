// YOLO 检测部署 | src/pipeline.cpp
// 功能：把视频逐帧交给 Detector，再显示或统计。并发模式走下面两条队列，串行模式在一个循环内处理。
// frame_queue_：读取线程放帧，推理线程取帧；读完视频后由读取线程关闭，推理端可取完剩余帧。
// result_queue_：推理线程放结果，主线程取结果；推理循环结束后关闭，主线程可取完剩余结果。
// 两条队列都在空且未关闭时等数据。满时按同一策略等空位或丢最旧项。异常停止会关闭两队列并置停止标志，循环可能提前退出。
// 目的：让读取下一帧和检测当前帧可以重叠，同时给等待中的帧数设上限。
//
// 阅读顺序：初始化与停止条件 → 读取帧 → 推理 → 消费结果 → run 串起各阶段并汇总。


#include "pipeline.h"  // 视频帧、处理选项和各线程之间的交接接口
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <stdexcept>


// 模块：统计辅助
// 功能：计算毫秒差并写入阶段统计
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


// 模块：流水线初始化（Pipeline::Pipeline）
// 功能：创建两个有界队列和唯一检测器
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


// 模块：异常与取消（Pipeline::requestStop）
// 功能：保存错误，关闭队列并唤醒等待者
// 异常/取消走快速停止：先记住首个异常、关闭并唤醒两个队列，再由run()汇合线程。
// 与正常EOF逐级排空不同，此路径不保证处理完所有已读帧。
void Pipeline::requestStop(std::exception_ptr error) {
    if (error) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        // 只保留第一个异常。后面线程因为队列已关闭而退出时，不会盖掉真正的起因。
        if (!error_)
            error_ = error;
    }
    stop_requested_ = true;
    frame_queue_.close();
    result_queue_.close();
}


// 模块：读取阶段（Pipeline::readLoop）
// 功能：解码视频，深拷贝当前帧后交给输入队列
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
        // data 移入输入队列后，由推理线程取走。push 返回 false 表示交接失败，读取方应停止继续投递。
        if (!frame_queue_.push(std::move(data)))
            break;
    }
    // 正常读完只关闭输入队列：拒绝新帧，但推理线程仍可取走队列里的剩余帧。
    frame_queue_.close();
}


// 模块：阶段采样（Pipeline::collect）
// 功能：收集检测器最近一帧的耗时
// collect 在一次 detect 结束后读取该帧各阶段的计时。这能排查“模型慢还是前后处理慢”；想知道帧在进模型之前等了多久，要看 inferenceLoop 记录的队列等待时间。
void Pipeline::collect() {
    inference_.record(detector_.last_inference_ms());
    preprocess_.record(detector_.last_preprocess_ms());
    backend_.record(detector_.last_backend_call_ms());
    postprocess_.record(detector_.last_postprocess_ms());
}


// 模块：推理阶段（Pipeline::inferenceLoop）
// 功能：出队后在锁外检测，再投递结果
void Pipeline::inferenceLoop() {
    FrameData data;
    // pop返回前已释放队列锁，耗时detect在锁外执行，读取线程可继续入队。
    while (!stop_requested_ && frame_queue_.pop(data)) {
        queue_wait_.record(ms(data.captured, Clock::now()));
        auto image = detector_.detect(data.frame);
        ++inference_count_;
        collect();
        // 同一帧的 captured 一路传到输出端。这样端到端耗时会包括中间排队的时间。
        if (!result_queue_.push({data.frame_id, std::move(image), data.captured}))
            break;
    }
    // 输入排空后关闭结果队列，让主线程消费完最后的结果并退出。
    result_queue_.close();
}


// 模块：结果消费（Pipeline::consume）
// 功能：可选显示图片并记录端到端耗时
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


// 模块：消费循环（Pipeline::display）
// 功能：读取结果队列直到结束或取消
void Pipeline::display() {
    ResultData result;
    while (!stop_requested_ && result_queue_.pop(result))
        consume(result);
}


// 模块：运行入口（Pipeline::run）
// 功能：预热，组织顺序模式或多线程模式
void Pipeline::run() {
    if (options_.warmup < 0 || options_.max_frames < 0)
        throw std::invalid_argument("invalid pipeline limits");


    // 预热
    // 功能：初始化计算路径，不纳入正式统计
    // 预热复用同一模型，但不record到阶段统计，也不计入下面的整体运行时间。
    for (int i = 0; i < options_.warmup; ++i)
        (void)detector_.detect(cv::Mat(640, 640, CV_8UC3, cv::Scalar{}));


    // 正式运行
    // 功能：启动监视线程并选择执行模式
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
            // [&] 让线程函数访问当前作用域里的对象。run 返回前要等线程退出，避免线程继续使用已离开作用域的数据。
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


    // 线程回收
    // 功能：等待退出并重新抛出首个异常
    // join只等待线程结束，不会主动停止线程；异常时前面必须先关闭队列解除等待。
    // 假设读线程已经退出，结果队列里还剩一帧：读线程已结束只说明不会再读新图。正常结束时仍要让后面的阶段消费这一帧，才能汇总最终数量。
    // joinable 表示这个对象还有线程需要回收，线程即使已经跑完也可能为真。join 等该线程结束并收回它。
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


    // 汇总报告
    // 功能：计算实际输出FPS和各阶段指标
    // 实际输出FPS=输出帧数/整段运行秒数；与1000/平均Run毫秒数的计时范围不同。
    // 真实输出 FPS 的分子是消费端实际处理的帧数；如果旧帧被丢弃，就不能拿读入帧数代替。
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
