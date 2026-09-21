#include <iostream>
#include <chrono>
#include "detector.h"
using namespace std;
using namespace cv;

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

    auto start = chrono::high_resolution_clock::now();

    Mat frame;
    int count = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        Mat result = detector.detect(frame);

        count++;
        if (count % 10 == 0) {
            imwrite("frame_" + to_string(count) + ".jpg", result);
            cout << "第" << count << "帧处理完成" << endl;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end - start).count();
    cout << "总处理时间：" << total_time << " 秒" << endl;
    cout << "总帧数：" << count << endl;

    cap.release();
    cout << "处理完成" << endl;
    return 0;
}
