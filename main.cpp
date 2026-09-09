#include <iostream>
#include "detector.h"
using namespace std;
using namespace cv;

int main() {
    // 初始化检测器
    Detector detector(
        "/home/xct/cpp_projects/yolo11n.onnx",
        "/home/xct/cpp_projects/coco.txt"
    );

    // 打开视频
    VideoCapture cap("/home/xct/cpp_projects/test.mp4");
    if (!cap.isOpened()) {
        cout << "视频打开失败" << endl;
        return -1;
    }

    Mat frame;
    int count = 0;
    while (true) {
        cap >> frame;          // 读取一帧
        if (frame.empty()) break;

        Mat result = detector.detect(frame);   // 检测

        count++;
        if (count % 10 == 0) {
            // 每10帧保存一张结果
            imwrite("frame_" + to_string(count) + ".jpg", result);
            cout << "第" << count << "帧处理完成" << endl;
        }
        if (count >= 50) break;
    }

    cap.release();
    cout << "处理完成" << endl;
    return 0;
}
