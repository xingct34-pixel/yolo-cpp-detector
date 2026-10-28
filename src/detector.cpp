#include "detector.h"
#include "ort_backend.h"
#ifdef ENABLE_TENSORRT
#include "trt_backend.h"
#endif
#include <chrono>
#include <fstream>
#include <stdexcept>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
}
Detector::Detector(const std::string& model,const std::string& classes,BackendType type,const std::string& engine,bool cuda){
    std::ifstream file(classes);if(!file)throw std::runtime_error("cannot open classes: "+classes);
    std::string name;while(std::getline(file,name)){if(!name.empty()&&name.back()=='\r')name.pop_back();if(!name.empty())names_.push_back(name);}
    if(names_.size()!=80)throw std::runtime_error("this YOLO model requires exactly 80 class names");
    if(type==BackendType::ORT)backend_=std::make_unique<OrtBackend>(model,cuda);
    else {
#ifdef ENABLE_TENSORRT
        backend_=std::make_unique<TrtBackend>(engine);
#else
        (void)engine;throw std::runtime_error("TensorRT is not enabled in this build");
#endif
    }
}
cv::Mat Detector::detect(const cv::Mat& image){
    auto begin=Clock::now();auto result=image.clone();auto cloned=Clock::now();
    LetterboxInfo info;auto input=prepare_yolo(image,info,times_);auto prepared=Clock::now();
    auto output=backend_->infer(input);auto inferred=Clock::now();
    detections_=decode_yolo(output,image.size(),info);draw_detections(result,detections_,names_);auto end=Clock::now();
    clone_ms_=ms(begin,cloned);preprocess_ms_=ms(cloned,prepared);backend_ms_=ms(prepared,inferred);postprocess_ms_=ms(inferred,end);
    return result;
}
