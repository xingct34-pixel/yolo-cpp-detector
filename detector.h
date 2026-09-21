#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class Detector {
public:
    Detector(const std::string& model_path, const std::string& classes_path);             //加载模型路径和类别路径
    cv::Mat detect(cv::Mat& img);
private:
    Ort::Env env_;                                                         //加载ONNX Runtime环境
    Ort::SessionOptions session_options_;                                  //session配置选项
    Ort::Session session_;                                        // session_必须在env_后面，ONNX模型会话，模型加载、推理
    std::vector<std::string> class_names_;
    std::vector<float> preprocess(cv::Mat& img, int& img_w, int& img_h);    //预处理，输入原图，输出归一化、转通道、resize后的模型输入浮点数组
    void postprocess(cv::Mat& img, float* data, int img_w, int img_h);      // 后处理函数
         static Ort::SessionOptions make_session_options();   // 新增这一行
};
