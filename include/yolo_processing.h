#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct LetterboxInfo {
    float scale=1, scale_x=1, scale_y=1;
    int pad_x=0, pad_y=0, input_width=640, input_height=640;
};
struct Detection { cv::Rect box; float score; int class_id; };
struct PreprocessTimes { double resize_ms=0, padding_ms=0, tensor_ms=0; };
// Pure processing functions allow deterministic tests without model weights.
std::vector<float> prepare_yolo(const cv::Mat&, LetterboxInfo&, PreprocessTimes&);
std::vector<Detection> decode_yolo(const std::vector<float>&, cv::Size,
    const LetterboxInfo&, float confidence=0.5f, float iou=0.45f);
void draw_detections(cv::Mat&,const std::vector<Detection>&,const std::vector<std::string>&);
