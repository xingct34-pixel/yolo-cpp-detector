#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class Detector {
public:
    Detector(const std::string& model_path, const std::string& classes_path);
    cv::Mat detect(cv::Mat& img);
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;              // session_必须在env_后面
    std::vector<std::string> class_names_;
    std::vector<float> preprocess(cv::Mat& img, int& img_w, int& img_h);
    void postprocess(cv::Mat& img, float* data, int img_w, int img_h);};
