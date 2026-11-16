// YOLO 检测部署 | include/yolo_processing.h
// 功能：定义预处理和后处理之间要共享的几何信息、检测记录和计时结构。
// 缩放时记下比例和补边，解码时用它们把模型坐标还原到原图。
// 实现位于 src/yolo_processing.cpp，Detector 和精度评估工具共用这些接口。
// 目的：让画面变形前后的坐标能够相互换算，让显示和导出使用同一份检测记录。


#pragma once
#include <opencv2/core.hpp>
#include <vector>


// 模块：几何记录（LetterboxInfo）
// 功能：连接预处理缩放与后处理坐标还原
// scale是理想比例，scale_x/y是尺寸取整后的实际比例；pad_x/y只记录左/上补边。
// 例如原图 1280×960 被缩成 640×480，上边补 80 行。模型坐标 y=180 换回原图是 (180−80)/0.5=200；所以解码除了拿到框，还得拿到这张图自己的 LetterboxInfo。
struct LetterboxInfo {
    // 一个 info 描述一张图的几何变换。按原尺寸算的理想比例和按取整结果算的实际比例要区分。
    float scale = 1, scale_x = 1, scale_y = 1;
    int pad_x = 0, pad_y = 0, input_width = 640, input_height = 640;
};


// 模块：检测记录（Detection）
// 功能：保存原图矩形、类别分数和类别编号
// box已经还原到原图，格式为{左上角x, 左上角y, 宽, 高}；class_id对应名称列表下标。
struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};


// 模块：耗时记录（PreprocessTimes）
// 功能：区分缩放、补边和张量准备
struct PreprocessTimes {
    double resize_ms = 0, padding_ms = 0, tensor_ms = 0;
};


// 模块：处理接口（prepare_yolo）
// 功能：由yolo_processing.cpp实现，供Detector及测试调用
// 纯处理函数不加载模型，便于用固定数据测试。默认分数门槛0.5、NMS IoU阈值0.45是两回事。
// 第二、三个参数带 &，prepare_yolo 在返回浮点数组的同时，还能把补边和计时写回调用方。
std::vector<float> prepare_yolo(const cv::Mat&, LetterboxInfo&, PreprocessTimes&);
std::vector<Detection> decode_yolo(const std::vector<float>&,
                                   cv::Size,
                                   const LetterboxInfo&,
                                   float confidence = 0.5f,
                                   float iou = 0.45f);
void draw_detections(cv::Mat&, const std::vector<Detection>&, const std::vector<std::string>&);
