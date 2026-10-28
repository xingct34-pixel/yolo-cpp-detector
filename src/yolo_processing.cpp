#include "yolo_processing.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
}
std::vector<float> prepare_yolo(const cv::Mat& image,LetterboxInfo& info,PreprocessTimes& times){
    if(image.empty()||image.type()!=CV_8UC3) throw std::invalid_argument("input must be a nonempty BGR uint8 image");
    info=LetterboxInfo{};
    info.scale=std::min(640.f/image.cols,640.f/image.rows);
    const int w=std::clamp(int(std::round(image.cols*info.scale)),1,640);
    const int h=std::clamp(int(std::round(image.rows*info.scale)),1,640);
    info.scale_x=float(w)/image.cols;info.scale_y=float(h)/image.rows;
    info.pad_x=(640-w)/2;info.pad_y=(640-h)/2;
    auto begin=Clock::now();cv::Mat resized,padded,blob;
    cv::resize(image,resized,{w,h});auto resized_at=Clock::now();
    cv::copyMakeBorder(resized,padded,info.pad_y,640-h-info.pad_y,
        info.pad_x,640-w-info.pad_x,cv::BORDER_CONSTANT,{114,114,114});
    auto padded_at=Clock::now();
    cv::dnn::blobFromImage(padded,blob,1./255.,{}, {},true,false,CV_32F);
    std::vector<float> data(blob.ptr<float>(),blob.ptr<float>()+blob.total());
    times={ms(begin,resized_at),ms(resized_at,padded_at),ms(padded_at,Clock::now())};return data;
}
std::vector<Detection> decode_yolo(const std::vector<float>& output,cv::Size size,
    const LetterboxInfo& info,float confidence,float iou){
    constexpr int count=8400,classes=80;
    if(output.size()!=84u*count||size.width<=0||size.height<=0||
       !std::isfinite(info.scale_x)||!std::isfinite(info.scale_y)||info.scale_x<=0||info.scale_y<=0||
       !std::isfinite(confidence)||!std::isfinite(iou)||confidence<0||confidence>1||iou<0||iou>1)
        throw std::invalid_argument("invalid YOLO output, geometry or thresholds");
    std::map<int,std::vector<Detection>> groups;
    for(int n=0;n<count;++n){
        float score=confidence;int id=-1;
        for(int c=0;c<classes;++c){float s=output[(4+c)*count+n];
            if(std::isfinite(s)&&s>=score&&s<=1){score=s;id=c;}}
        if(id<0)continue;
        float x=output[n],y=output[count+n],w=output[2*count+n],h=output[3*count+n];
        if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0)continue;
        // Clamp while still floating point, before conversion to an integer.
        auto bound=[](double v,int upper){return int(std::round(std::clamp(v,0.,double(upper))));};
        int left=bound((double(x)-w/2.-info.pad_x)/info.scale_x,size.width);
        int top=bound((double(y)-h/2.-info.pad_y)/info.scale_y,size.height);
        int right=bound((double(x)+w/2.-info.pad_x)/info.scale_x,size.width);
        int bottom=bound((double(y)+h/2.-info.pad_y)/info.scale_y,size.height);
        if(right>left&&bottom>top)groups[id].push_back({{left,top,right-left,bottom-top},score,id});
    }
    std::vector<Detection> result;
    // Per-class NMS avoids fixed coordinate-offset assumptions on large images.
    for(auto& entry:groups){std::vector<cv::Rect> boxes;std::vector<float> scores;std::vector<int> keep;
        for(const auto& d:entry.second){boxes.push_back(d.box);scores.push_back(d.score);}
        cv::dnn::NMSBoxes(boxes,scores,std::max(0.f,confidence-1e-6f),iou,keep);
        for(int k:keep)result.push_back(entry.second.at(k));}
    std::stable_sort(result.begin(),result.end(),[](const auto&a,const auto&b){return a.score>b.score;});return result;
}
void draw_detections(cv::Mat& image,const std::vector<Detection>& detections,const std::vector<std::string>& names){
    for(const auto& d:detections){cv::rectangle(image,d.box,{0,255,0},2);
        std::string label=names.at(d.class_id)+" "+std::to_string(d.score).substr(0,4);
        cv::putText(image,label,{d.box.x,std::max(15,d.box.y)},cv::FONT_HERSHEY_SIMPLEX,0.5,{0,180,0},1);}
}
