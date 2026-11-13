// YOLO 检测部署 | tests/test_processing.cpp
// 手工构造像素和模型输出，直接调用 prepare_yolo、decode_yolo，不依赖模型权重。
// 例如把两个同类框和一个异类框放在相同位置，就能检查 NMS 是否只抑制同类重复框。


#include "yolo_processing.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
int main(){
    // Regression: external-buffer packing must match the former blob + copy
    // path for nonuniform pixels, portrait/landscape and rounding boundaries.
    for (const auto size : {cv::Size(1000,333), cv::Size(333,1000), cv::Size(640,640), cv::Size(17,31)}) {
        cv::Mat image(size,CV_8UC3);
        cv::randu(image,0,256);
        LetterboxInfo geometry; PreprocessTimes measured;
        auto packed=prepare_yolo(image,geometry,measured);
        const int width=std::clamp(int(std::round(image.cols*geometry.scale)),1,640);
        const int height=std::clamp(int(std::round(image.rows*geometry.scale)),1,640);
        cv::Mat resized,padded;
        cv::resize(image,resized,{width,height});
        cv::copyMakeBorder(resized,padded,geometry.pad_y,640-height-geometry.pad_y,
            geometry.pad_x,640-width-geometry.pad_x,cv::BORDER_CONSTANT,{114,114,114});
        auto reference=cv::dnn::blobFromImage(padded,1./255.,{}, {},true,false,CV_32F);
        assert(packed.size()==reference.total());
        for(size_t i=0;i<packed.size();++i) assert(packed[i]==reference.ptr<float>()[i]);
    }
    LetterboxInfo info;PreprocessTimes times;
    auto tensor=prepare_yolo(cv::Mat(333,1000,CV_8UC3,cv::Scalar(0,0,255)),info,times);
    assert(tensor.size()==3u*640*640);assert(info.pad_y==213);assert(info.scale_y!=info.scale_x);
    auto at=213u*640+20;assert(std::abs(tensor[at]-1)<1e-6);assert(tensor[640u*640+at]==0);
    assert(std::abs(tensor[0]-114.f/255)<1e-6);


    // 模块：制造模型输出
    // 直接填入框和类别分数，后处理的正确答案可以手算，不需要运行模型。


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
