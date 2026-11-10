#pragma once
#include <opencv2/core.hpp>
#include <vector>

// scale是理想比例，scale_x/y是尺寸取整后的实际比例；pad_x/y只记录左/上补边。
struct LetterboxInfo {
    float scale=1, scale_x=1, scale_y=1;
    int pad_x=0, pad_y=0, input_width=640, input_height=640;
};
// box已经还原到原图，格式为{左上角x, 左上角y, 宽, 高}；class_id对应名称列表下标。
struct Detection { cv::Rect box; float score; int class_id; };
struct PreprocessTimes { double resize_ms=0, padding_ms=0, tensor_ms=0; };
// 纯处理函数不加载模型，便于用固定数据测试。默认分数门槛0.5、NMS IoU阈值0.45是两回事。
std::vector<float> prepare_yolo(const cv::Mat&, LetterboxInfo&, PreprocessTimes&);
std::vector<Detection> decode_yolo(const std::vector<float>&, cv::Size,
    const LetterboxInfo&, float confidence=0.5f, float iou=0.45f);
void draw_detections(cv::Mat&,const std::vector<Detection>&,const std::vector<std::string>&);
