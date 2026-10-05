#include "pipeline.h"

#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

Pipeline::Pipeline(const string& video_path,
                   const string& model_path,
                   const string& classes_path)

    : frame_queue_(3),
      // 结果队列同理,最多容纳 3 个
      result_queue_(3),
      // 用模型路径和类别文件路径初始化检测器
      detector_(model_path, classes_path),

      video_path_(video_path)
{
    // 函数体为空:所有初始化工作都已经在上面的初始化列表里完成
}

// 读线程
void Pipeline::readLoop()
{
    // VideoCapture 是 OpenCV 的视频读取类
    VideoCapture cap(video_path_);

    if (!cap.isOpened())
    {
        cerr << "视频打开失败！" << endl;
        frame_queue_.close();
        return;
    }

    int frame_id = 0;
    Mat frame;

    while (cap.read(frame))
    {
        FrameData data;
      
        data.frame_id = frame_id++;                 // frame_id++ 是"先使用当前值,再加 1",所以第一帧编号是 0
        // clone() 是深拷贝,复制一份独立的图像数据
        // 必须拷贝:cap.read 每次会覆盖 frame 的内容,
        // 如果不拷贝,队列里的图像会被下一次读取改掉
        data.frame = frame.clone();
        // 已读取帧数加 1(只有读线程修改它)
        read_count_++;
        // std::move 把 data "移动"进队列而不是复制,避免多拷贝一次图像
        // 如果队列已满,push 会阻塞等待,直到推理线程取走数据腾出位置(依据队列实现推断)
        frame_queue_.push(std::move(data));
    }

    frame_queue_.close();
}

// 推理线程:从 frame_queue_ 取帧,做 YOLO 检测,结果送入 result_queue_
void Pipeline::inferenceLoop()
{
    while (true)
    {
        FrameData data;

        if (!frame_queue_.pop(data))
        {
            break;
        }


        Mat result = detector_.detect(data.frame);
        inference_count_++;
        ResultData result_data;

        // 帧编号沿用原来的编号,这样显示线程能知道这是第几帧
        result_data.frame_id = data.frame_id;
        // 放入画好框的图像
        result_data.frame = result;

        // 移动进结果队列,等待显示线程取走
        result_queue_.push(std::move(result_data));
    }

    result_queue_.close();
}

void Pipeline::display()
{
    // 记录上一次显示的帧编号,初始为 -1 表示还没显示过任何帧
    int last_displayed_id = -1;

    while (true)
    {
        ResultData result;

        // 空
        if (!result_queue_.pop(result))
        {
            break;
        }

        // 防御性判断:如果这一帧的编号不比上次显示的大(乱序或重复),就跳过
        if (result.frame_id <= last_displayed_id)
        {
            continue;
        }

        // 在名为 "YOLO Result" 的窗口中显示图像
        imshow("YOLO Result", result.frame);1
        display_count_++;
        last_displayed_id = result.frame_id;

        // waitKey(1) 等待 1 毫秒看有没有按键                               ？
        // 它同时还负责让窗口刷新画面,没有它 imshow 的窗口不会正常显示
        // 27 是 ESC 键的编码,按下 ESC 就退出显示循环
        if (waitKey(1) == 27)
        {
            break;
        }
    }
}

// 对外入口:启动整个流水线并统计耗时
void Pipeline::run()
{
    // 记录开始时间,steady_clock 是单调时钟,不受系统改时间影响,适合计时

    auto start = chrono::steady_clock::now();

    // 创建读线程:第一个参数是要执行的成员函数的地址,第二个参数 this 指明在当前对象上调用
    thread read_thread(&Pipeline::readLoop, this);
    // 创建推理线程
    thread inference_thread(&Pipeline::inferenceLoop, this);

    // 显示放在当前(主)线程执行,因为 OpenCV 的窗口操作通常要求在主线程里做
    // 这个函数会一直运行到显示结束
    display();

    // join() 表示等待该线程执行完毕,主线程在这里阻塞,确保线程都结束后才继续
    read_thread.join();
    inference_thread.join();

    // 记录结束时间
    auto end = chrono::steady_clock::now();

    // 两个时间点相减得到时间段,转成以秒为单位的 double,count() 取出具体数值
    double elapsed =
        chrono::duration<double>(end - start).count();
        // 打印三个阶段各处理的帧数,方便观察是否有丢帧
        cout << "读取帧数：" << read_count_ << endl;
        cout << "推理帧数：" << inference_count_ << endl;
        cout << "显示帧数：" << display_count_ << endl;
        // 打印总耗时
        cout << "Pipeline 总耗时：" << elapsed << " 秒" << endl;

        // 耗时大于 0 才计算,避免除以 0
        if (elapsed > 0)
         {
           // FPS = 每秒处理的帧数 = 推理帧数 / 总耗时
           cout << "实际推理 FPS："
          << inference_count_ / elapsed << endl;
         }
}
