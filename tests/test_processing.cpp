#include "yolo_processing.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
int main(){
    LetterboxInfo info;PreprocessTimes times;
    auto tensor=prepare_yolo(cv::Mat(333,1000,CV_8UC3,cv::Scalar(0,0,255)),info,times);
    assert(tensor.size()==3u*640*640);assert(info.pad_y==213);assert(info.scale_y!=info.scale_x);
    auto at=213u*640+20;assert(std::abs(tensor[at]-1)<1e-6);assert(tensor[640u*640+at]==0);
    assert(std::abs(tensor[0]-114.f/255)<1e-6);
    std::vector<float> raw(84*8400,0);
    auto box=[&](int n,int cls,float score){raw[n]=320;raw[8400+n]=320;raw[16800+n]=320;raw[25200+n]=100;raw[(4+cls)*8400+n]=score;};
    box(0,0,.9);box(1,0,.8);box(2,1,.85); // same class suppressed; different class kept
    auto detections=decode_yolo(raw,{1000,333},info);assert(detections.size()==2);assert(detections[0].box.x==250);
    raw[0]=std::numeric_limits<float>::quiet_NaN();
    detections=decode_yolo(raw,{1000,333},info);assert(detections.size()==2); // second class-0 survives
    raw[1]=std::numeric_limits<float>::max();detections=decode_yolo(raw,{1000,333},info);assert(detections.size()==1);
    bool rejected=false;try{prepare_yolo(cv::Mat(),info,times);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    rejected=false;try{decode_yolo({}, {1,1},info);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    prepare_yolo(cv::Mat(1,20000,CV_8UC3,cv::Scalar{}),info,times);assert(info.scale_y>0);
    std::cout<<"letterbox RGB/CHW, class-aware NMS, NaN, clipping and invalid input passed\n";
}
