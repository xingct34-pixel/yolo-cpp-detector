#include <iostream>
#include <thread>
#include <chrono>
#include "detector.h"
#include "thread_safe_queue.h"
using namespace std;
using namespace cv;

// 用来在"显示线程"和"保存线程"之间传递数据：图片本身 + 要保存的文件名
struct SaveTask {
    Mat image;
    string filename;
};

// 读帧线程：只负责从视频里读帧，push进frame_queue
void read_frames(VideoCapture& cap, ThreadSafeQueue<Mat>& frame_queue) {
    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frame_queue.push(frame);
    }
    frame_queue.push(Mat());
}

// 推理线程：从frame_queue里pop，调用detect()，结果push进result_queue
void infer_frames(Detector& detector, ThreadSafeQueue<Mat>& frame_queue, ThreadSafeQueue<Mat>& result_queue) {
    while (true) {
        Mat frame = frame_queue.pop();
        if (frame.empty()) break;
        Mat result = detector.detect(frame);
        result_queue.push(result);
    }
    result_queue.push(Mat());
}

// 显示线程：从result_queue里pop，每10帧生成一个保存任务，push进save_queue（不直接写文件，不卡住自己）
void show_results(ThreadSafeQueue<Mat>& result_queue, ThreadSafeQueue<SaveTask>& save_queue) {
    int count = 0;
    while (true) {
        Mat result = result_queue.pop();
        if (result.empty()) break;
        count++;
        if (count % 10 == 0) {
            SaveTask task;
            task.image = result;
            task.filename = "frame_" + to_string(count) + ".jpg";
            save_queue.push(task);   // 只是丢进队列，不等真正写完
            cout << "第" << count << "帧处理完成" << endl;
        }
    }
    // 通知保存线程：没有更多要保存的了
    save_queue.push(SaveTask{Mat(), ""});
}

// 保存线程：专门负责把图片写入磁盘，慢是它自己的事，不影响别的线程
void save_results(ThreadSafeQueue<SaveTask>& save_queue) {
    while (true) {
        SaveTask task = save_queue.pop();
        if (task.image.empty()) break;   // 收到结束信号
        imwrite(task.filename, task.image);
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

    ThreadSafeQueue<Mat> frame_queue(5);
    ThreadSafeQueue<Mat> result_queue(5);
    ThreadSafeQueue<SaveTask> save_queue(5);   // 新增：保存任务队列

    auto start = chrono::high_resolution_clock::now();

    thread t1(read_frames, std::ref(cap), std::ref(frame_queue));
    thread t2(infer_frames, std::ref(detector), std::ref(frame_queue), std::ref(result_queue));
    thread t3(show_results, std::ref(result_queue), std::ref(save_queue));
    thread t4(save_results, std::ref(save_queue));   // 新增：保存线程

    t1.join();
    t2.join();
    t3.join();
    t4.join();   // 新增：也要等保存线程跑完

    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end - start).count();
    cout << "总处理时间：" << total_time << " 秒" << endl;

    cap.release();
    cout << "处理完成" << endl;
    return 0;
}
