#pragma once
#include "inference_backend.h"
#include "yolo_processing.h"
#include <memory>
#include <string>
class Detector {
public:
    enum class BackendType { ORT,TensorRT };
    Detector(const std::string&,const std::string&,BackendType,const std::string& engine_path="",bool use_cuda=false);
    cv::Mat detect(const cv::Mat&);
    const std::vector<Detection>& detections() const{return detections_;}
    double last_inference_ms() const{return backend_->last_latency_ms();}
    double last_image_clone_ms() const{return clone_ms_;}
    double last_preprocess_ms() const{return preprocess_ms_;}
    double last_resize_ms() const{return times_.resize_ms;}
    double last_padding_ms() const{return times_.padding_ms;}
    double last_tensor_prepare_ms() const{return times_.tensor_ms;}
    double last_backend_call_ms() const{return backend_ms_;}
    double last_postprocess_ms() const{return postprocess_ms_;}
    std::string backend_name() const{return backend_->name();}
private:
    std::vector<std::string> names_;
    std::unique_ptr<InferenceBackend> backend_;
    std::vector<Detection> detections_;
    PreprocessTimes times_;
    double clone_ms_=0,preprocess_ms_=0,backend_ms_=0,postprocess_ms_=0;
};
