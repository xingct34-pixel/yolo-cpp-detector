#include <iostream>
#include <thread>
#include <chrono>
#include "detector.h"
#include "thread_safe_queue.h"
using namespace std;
using namespace cv;

// 读帧线程：只负责从视频里读帧，push进frame_queue
void read_frames(VideoCapture& cap, ThreadSafeQueue<Mat>& frame_queue) {
    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;   // 视频读完了，退出循环
        frame_queue.push(frame);
    }
    // 读完之后，push一个空Mat作为"结束信号"，通知推理线程没有更多帧了
    frame_queue.push(Mat());
}

// 推理线程：从frame_queue里pop，调用detect()，结果push进result_queue
void infer_frames(Detector& detector, ThreadSafeQueue<Mat>& frame_queue, ThreadSafeQueue<Mat>& result_queue) {
    while (true) {
        Mat frame = frame_queue.pop();
        if (frame.empty()) break;   // 收到结束信号，退出循环
        Mat result = detector.detect(frame);
        result_queue.push(result);
    }
    // 同样push一个空Mat，通知显示线程没有更多结果了
    result_queue.push(Mat());
}

// 显示线程：从result_queue里pop，保存/显示
void show_results(ThreadSafeQueue<Mat>& result_queue) {
    int count = 0;
    while (true) {
        Mat result = result_queue.pop();
        if (result.empty()) break;  // 收到结束信号，退出循环
        count++;
        if (count % 10 == 0) {
            imwrite("frame_" + to_string(count) + ".jpg", result);
            cout << "第" << count << "帧处理完成" << endl;
        }
    }
}

int main() {
    Detector detector(
        "/home/xct/cpp_projects/yolo11n.onnx",
        "/home/xct/cpp_projects/coco.txt"
    );

    VideoCapture cap("/home/xct/cpp_projects/test.mp4");
    if (!cap.isOpened()) {
        cout << "视频打开失败" << endl;
        return -1;
    }

    // 创建两个队列，容量都设为5（经验值，后续可调）
    ThreadSafeQueue<Mat> frame_queue(5);
    ThreadSafeQueue<Mat> result_queue(5);

    // 记录整个Pipeline开始的时间点
    auto start = chrono::high_resolution_clock::now();

    // 创建三个线程，分别执行读帧、推理、显示
    thread t1(read_frames, std::ref(cap), std::ref(frame_queue));
    thread t2(infer_frames, std::ref(detector), std::ref(frame_queue), std::ref(result_queue));
    thread t3(show_results, std::ref(result_queue));

    // 等待三个线程都执行完毕，主线程才能退出
    t1.join();
    t2.join();
    t3.join();

    // 记录结束时间点，计算总耗时
    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end - start).count();
    cout << "总处理时间：" << total_time << " 秒" << endl;

    cap.release();
    cout << "处理完成" << endl;
    return 0;
}
