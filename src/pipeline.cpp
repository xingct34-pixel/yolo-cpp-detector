#include "pipeline.h"

#include <chrono>
#include <iostream>
#include <thread>

/*
1. 为什么 Pipeline 要分三个阶段?
读取(解码)、推理、显示的耗时差别很大,尤其是推理最慢。串行时每帧耗时是三者之和,分成三个阶段并行后,吞吐取决于最慢的那一段,读取和显示的耗时基本被推理时间"藏"起来了。分阶段还让各模块职责单一
    */

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

    /*
    7. frame_id 是干什么的?
    给每帧一个全局递增的序号,用途有三个:

    对齐结果和原始帧:ResultData 带着同一个 frame_id,可以知道这个结果对应第几帧。
    检测丢帧:序号不连续就说明中间的帧被丢了。
    排序和调试:多个推理线程时,结果可能乱序到达,用它可以恢复顺序,日志中也能定位具体某一帧。
    */


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
        
    /*
    2. 为什么 readLoop() 要 clone()?
    cv::Mat 赋值或拷贝只复制头部,像素数据是引用计数共享的。cv::VideoCapture::read(frame) 会反复复用同一块缓冲区,如果直接把 frame 放进队列,
    下一次 read 会覆盖队列里还没处理的那一帧,推理线程就会读到被篡改或画面撕裂的数据,这是数据竞争。clone() 做一次深拷贝,让每个 FrameData 拥有独立的像素内存。
    */

        
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
        
        /*
        6. 为什么使用 pop(),不能直接访问 queue_.front()?
        front() 加 pop() 是两步操作,如果分别加锁,两步之间锁已经释放,别的线程可能插入或取走元素,拿到的引用就可能失效,而且 front() 返回的引用在锁外使用本身就不安全。
        封装成一个 pop(),在同一把锁内完成"等待、取出、移除",再把元素按值或移动返回,保证原子性。封装还能让调用方不用关心锁和条件变量,并且能把"队列已关闭且为空"通过返回值告知调用方。
        */   
        
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

/*
5. condition_variable 在这里解决什么问题?
解决"队列为空时消费者怎么等"。没有它,消费者只能在循环里不断检查队列(忙等),白白占满 CPU。有了它,消费者在队列为空时睡眠,生产者 push 之后调用 notify 唤醒它,等待期间不消耗 CPU,数据一到几乎立刻响应。
    */

/*
8. 为什么 display() 放在主线程?
OpenCV 的 imshow / waitKey 这类 GUI 函数在很多平台(尤其是 macOS、部分 Linux 窗口系统)上要求在主线程调用,放在子线程容易出现窗口不刷新或崩溃。
    另外主线程要负责启动子线程并等待它们结束,把显示放在这里,既利用了主线程,又可以顺便处理按键退出,再通过 stop_requested_ 通知其他线程。
    */


/*
10. close() 是怎么让消费者退出的?
典型实现是:close() 在锁内把 closed_ 置为 true,然后对条件变量调用 notify_all()。消费者的等待条件是"队列非空或已关闭",被唤醒后如果队列为空且已关闭,pop() 就返回"失败"(比如 false 或空 optional),消费者循环据此退出。如果没有 notify_all(),阻塞在空队列上的消费者会永远睡下去,程序无法正常结束。通常还会让消费者先把队列里剩余的数据处理完再退出。

11. run() 为什么需要 join()?

保证线程结束后 run() 才返回,否则 run() 返回、Pipeline 析构时子线程还在访问成员(队列、detector_),会访问已销毁的对象,属于未定义行为。
一个仍可 join 的 std::thread 被销毁时会调用 std::terminate,程序直接崩溃。
需要等推理线程把已有数据处理完,统计数据才完整。

12. 如果把一个 inference thread 改成两个,会发生什么?

吞吐可能提高,前提是推理有可并行的资源,比如 GPU 有空闲算力或 CPU 核数足够。
结果可能乱序:两个线程各自耗时不同,帧 5 可能比帧 4 先进入 result_queue_,显示会出现画面跳动或倒退,需要靠 frame_id 做重排序,或者在显示端缓存后按序输出。
Detector 不一定线程安全:两个线程共用同一个 detector_,内部的会话、缓冲区、TensorRT 执行上下文都可能有共享状态,会引发数据竞争。解决办法是每个线程一个 Detector 实例(显存和内存翻倍),或者在 detector_ 内部加锁(此时第二个线程基本无收益)。
inference_count_ 是原子变量,不受影响;队列本身是线程安全的,两个消费者可以同时 pop。
如果推理已经占满 GPU,第二个线程只会增加竞争和延迟。
*/


/*
condition_variable
核心就是一个等待队列 + 一把锁，wait 和 notify 的具体过程如下。

等待方调用 cv.wait(lk) 时：

线程此时持有锁，先检查条件，不满足才进入等待。
把自己放进条件变量内部的等待队列，同时释放锁。这两步是原子的。
线程进入睡眠，由操作系统挂起，不占 CPU。

通知方调用 cv.notify_one() 时：

通知方先加锁修改条件（ready = true），再解锁。
notify_one 从等待队列里取出一个线程，把它标记为可运行。
被唤醒的线程先重新抢锁，抢到后 wait 才返回，然后重新检查条件。

为什么"入队 + 释放锁"必须原子：
如果拆成两步，先释放锁、再去睡，中间这个空档里线程B可能刚好修改条件并发出通知。此时A还没进队列，通知就丢了，A随后睡下去就再也醒不来（lost wakeup）。原子操作保证了：要么A在B修改条件之前已经入队，要么A检查条件时已经能看到 ready 为 true。

为什么醒来后要重新检查条件：
被唤醒到抢到锁之间，别的线程可能已经把条件改回去了（比如另一个消费者先取走了数据），加上系统可能发生虚假唤醒，所以要用 while 或带谓词的 wait 反复确认。
