#include "pipeline.h"

#include <chrono>
#include <iostream>
#include <thread>

// ==================== 构造模块 ====================

Pipeline::Pipeline(
    const std::string& video_path,
    const std::string& model_path,
    const std::string& classes_path,
    Detector::BackendType backend_type,
    const std::string& engine_path,
    bool use_cuda,
    size_t queue_size)
    : frame_queue_(queue_size),
      result_queue_(queue_size),
      detector_(
          model_path,
          classes_path,
          backend_type,
          engine_path,
          use_cuda),
      video_path_(video_path)
{
}

// ==================== 读取线程模块 ====================

void Pipeline::readLoop()
{
    cv::VideoCapture cap(
        video_path_);

    if (!cap.isOpened())
    {
        std::cerr
            << "视频打开失败："
            << video_path_
            << std::endl;

        frame_queue_.close();

        return;
    }

    int frame_id = 0;

    cv::Mat frame;

    // ==================== 视频读取循环 ====================

    while (!stop_requested_ &&
           cap.read(frame))
    {
        FrameData data;

        data.frame_id =
            frame_id++;

        // clone：
        // 给队列中的数据建立独立图像内存，
        // 避免下一次 cap.read() 修改同一块数据。

        data.frame =
            frame.clone();

        ++read_count_;

        if (!frame_queue_.push(
                std::move(data)))
        {
            break;
        }
    }

    // ==================== 通知消费者 ====================

    frame_queue_.close();
}

// ==================== 推理线程模块 ====================

void Pipeline::inferenceLoop()
{
    while (!stop_requested_)
    {
        FrameData data;

        // ==================== 从输入队列取数据 ====================

        if (!frame_queue_.pop(data))
        {
            break;
        }

        // ==================== Detector 推理 ====================

        cv::Mat result =
            detector_.detect(
                data.frame);

        ++inference_count_;

        // ==================== Benchmark ====================

        benchmark_.record(
            detector_.last_inference_ms());

        // ==================== 结果入队 ====================

        ResultData result_data;

        result_data.frame_id =
            data.frame_id;

        result_data.frame =
            std::move(result);

        if (!result_queue_.push(
                std::move(result_data)))
        {
            break;
        }
    }

    // ==================== 通知显示线程 ====================

    result_queue_.close();
}

// ==================== 显示模块 ====================

void Pipeline::display()
{
    int last_displayed_id = -1;

    while (!stop_requested_)
    {
        ResultData result;

        if (!result_queue_.pop(result))
        {
            break;
        }

        // ==================== 帧序号检查模块 ====================
        //
        // frame_id：
        // 每一帧的身份证。
        //
        // 后续如果扩展多个推理线程，
        // 可以利用它判断帧顺序。

        if (result.frame_id <=
            last_displayed_id)
        {
            continue;
        }

        cv::imshow(
            "YOLO Result",
            result.frame);

        ++display_count_;

        last_displayed_id =
            result.frame_id;

        // ESC：
        // 请求整个 Pipeline 停止。

        const int key =
            cv::waitKey(1);

        if (key == 27)
        {
            stop_requested_ = true;

            frame_queue_.close();
            result_queue_.close();

            break;
        }
    }

    cv::destroyAllWindows();
}

// ==================== Pipeline 主控制模块 ====================

void Pipeline::run()
{
    const auto start =
        std::chrono::steady_clock::now();

    // ==================== 创建读取线程 ====================

    std::thread read_thread(
        &Pipeline::readLoop,
        this);

    // ==================== 创建推理线程 ====================

    std::thread inference_thread(
        &Pipeline::inferenceLoop,
        this);

    // ==================== 主线程负责显示 ====================
    //
    // GUI（图形界面）通常要求在主线程处理。
    // 因此这里不再创建 display thread。

    display();

    // ==================== 等待线程结束 ====================
    //
    // join：
    // 主线程等待子线程完成。
    //
    // 可以理解成：
    // “等两个工人把手上的任务做完再关厂。”

    if (read_thread.joinable())
    {
        read_thread.join();
    }

    if (inference_thread.joinable())
    {
        inference_thread.join();
    }

    const auto end =
        std::chrono::steady_clock::now();

    const double elapsed =
        std::chrono::duration<
            double>(
                end - start)
            .count();

    // ==================== Pipeline 统计模块 ====================

    std::cout
        << "\n========== Pipeline ==========\n";

    std::cout
        << "读取帧数："
        << read_count_
        << '\n';

    std::cout
        << "推理帧数："
        << inference_count_
        << '\n';

    std::cout
        << "显示帧数："
        << display_count_
        << '\n';

    std::cout
        << "Pipeline 总耗时："
        << elapsed
        << " 秒\n";

    if (elapsed > 0.0)
    {
        std::cout
            << "Pipeline FPS："
            << static_cast<double>(
                   inference_count_) /
                   elapsed
            << '\n';
    }

    std::cout
        << "==============================\n";

    // ==================== 模型性能统计 ====================

    benchmark_.print(
        detector_.backend_name());
}
