// 头文件保护:防止这个头文件被重复包含导致"重复定义"的编译错误
#pragma once


#include <opencv2/opencv.hpp>            // OpenCV 的总头文件,引入后才能使用 cv::Mat、cv::VideoCapture 等

#include <string>                       // C++ 标准库的字符串类型 std::string

#include <thread>                      // C++ 标准库的线程支持 std::thread,流水线用多线程并行工作


#include "detector.h"             // 自己写的头文件,用双引号表示"项目内的文件"(尖括号一般用于系统/第三方库)

#include "thread_safe_queue.h"         // thread_safe_queue.h:线程安全队列,多个线程同时存取数据不会出错



struct FrameData {            // FrameData 表示"读线程交给推理线程的一份数据"

    int frame_id;

    // cv::Mat frame 拆开理解:
    // cv   是 OpenCV 的命名空间(namespace),相当于一个"姓氏",
    //      说明后面的名字属于 OpenCV 这个库,避免和别的库里的同名东西冲突。
    // ::   是作用域运算符,意思是"进入 cv 里面去找",cv::Mat 读作"cv 里的 Mat"。
    // Mat  是 OpenCV 里的类(Matrix 的缩写,矩阵),用来存储图像。
    //      一张彩色图片本质上就是一个 高 x 宽 x 3(蓝绿红三个通道)的数字矩阵,
    //      Mat 负责保存这些像素数据,并提供缩放、画框、转换颜色等操作。
    // frame 是变量名,是我们给这个 Mat 对象起的名字,这里代表"一帧画面"。
    // 合起来:声明一个名叫 frame 的变量,类型是 OpenCV 的图像矩阵 Mat。
    cv::Mat frame;
};


struct ResultData {          // ResultData 表示"推理线程交给显示线程的一份数据"

    int frame_id;

    cv::Mat frame;        // 已经画上检测框和类别名的图像,类型同样是 cv::Mat
};


class Pipeline {            // class 是"类",把数据和操作数据的函数封装在一起

public:

    // const std::string& 拆开理解:
    //   const        表示函数内部不会修改它
    //   &            表示"引用",传的是原对象本身而不是复制一份,避免拷贝字符串的开销
    Pipeline(const std::string& video_path,
             const std::string& model_path,
             const std::string& classes_path);

    void run();         // void 表示没有返回值

private:


    void readLoop();         // 读线程:循环读取视频的每一帧,打包成 FrameData 放进 frame_queue_


    void inferenceLoop();       // 推理线程:循环从 frame_queue_ 取帧,交给 detector_ 检测,结果放进 result_queue_

    void display();          // 显示线程:循环从 result_queue_ 取结果并显示出来           为什么显示不是循环？

private:

    long long read_count_ = 0;        // long long:比 int 范围更大的整数类型,计数很大时不会溢出

    long long inference_count_ = 0;      // inference_count_:已经完成推理的帧数

    long long display_count_ = 0;        // display_count_:已经显示的帧数

    // 作用:读线程往里放,推理线程从里取,中间用队列解耦,两个线程互不等待对方
    ThreadSafeQueue<FrameData> frame_queue_;

    // 结果队列:推理线程放入,显示线程取出,里面装的是 ResultData
    ThreadSafeQueue<ResultData> result_queue_;


    Detector detector_;

    // 输入视频路径,保存下来供 readLoop() 打开视频时使用
    std::string video_path_;
};
