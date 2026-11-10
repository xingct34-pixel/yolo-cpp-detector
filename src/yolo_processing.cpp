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
// 预处理：BGR原图 -> 等比例缩放/补边 -> RGB、0~1、NCHW浮点数组。
// 返回值交给模型；info通过引用保存几何变换，后处理还原坐标时必须使用同一份记录。
std::vector<float> prepare_yolo(const cv::Mat& image,LetterboxInfo& info,PreprocessTimes& times){
    if(image.empty()||image.type()!=CV_8UC3) throw std::invalid_argument("input must be a nonempty BGR uint8 image");
    info=LetterboxInfo{};
    // 宽、高使用同一比例，取较小值才能让两边都放进640x640，避免拉伸变形。
    info.scale=std::min(640.f/image.cols,640.f/image.rows);
    const int w=std::clamp(int(std::round(image.cols*info.scale)),1,640);
    const int h=std::clamp(int(std::round(image.rows*info.scale)),1,640);
    // w/h已经取整，实际横纵比例可能略有差异，反变换不能只用理想scale。
    info.scale_x=float(w)/image.cols;info.scale_y=float(h)/image.rows;
    info.pad_x=(640-w)/2;info.pad_y=(640-h)/2;
    auto begin=Clock::now();cv::Mat resized,padded;
    cv::resize(image,resized,{w,h});auto resized_at=Clock::now();
    // pad_x/pad_y记录左、上补边；右、下接收余量，避免奇数补边少补一个像素。
    // 例如1280x720 -> 640x360，上下各补140；114是当前预处理使用的灰色填充值。
    cv::copyMakeBorder(resized,padded,info.pad_y,640-h-info.pad_y,
        info.pad_x,640-w-info.pad_x,cv::BORDER_CONSTANT,{114,114,114});
    auto padded_at=Clock::now();
    // [N,C,H,W]=[1,3,640,640]：1张图、RGB三通道；内存依次放整幅R、G、B。
    // data拥有内存，blob只借用data.data()，转换完成前不能让data销毁或扩容。
    // 直接写入最终vector，省去一份约4.69 MiB的中间张量及完整复制。
    std::vector<float> data(3u * 640 * 640);
    const int shape[] = {1, 3, 640, 640};
    cv::Mat blob(4, shape, CV_32F, data.data());
    // 1/255将像素缩放到0~1，true将BGR换成RGB，CV_32F指定float32。
    cv::dnn::blobFromImage(padded,blob,1./255.,{}, {},true,false,CV_32F);
    // 若OpenCV另分配了缓冲区，返回data就拿不到转换结果，因此检查首地址。
    if (blob.ptr<float>() != data.data()) {
        throw std::runtime_error("OpenCV replaced the external tensor buffer");
    }
    times={ms(begin,resized_at),ms(resized_at,padded_at),ms(padded_at,Clock::now())};return data;
}
std::vector<Detection> decode_yolo(const std::vector<float>& output,cv::Size size,
    const LetterboxInfo& info,float confidence,float iou){
    // 本项目固定为640输入的YOLO11 COCO导出模型，不是所有YOLO都能套用这些值。
    // 检测头步长8/16/32，对应80x80、40x40、20x20网格，共8400个候选位置。
    // 输出[1,84,8400]中84=中心x/y、宽/高这4项+80类分数，不另含objectness一项。
    constexpr int count=8400,classes=80;
    if(output.size()!=84u*count||size.width<=0||size.height<=0||
       !std::isfinite(info.scale_x)||!std::isfinite(info.scale_y)||info.scale_x<=0||info.scale_y<=0||
       !std::isfinite(confidence)||!std::isfinite(iou)||confidence<0||confidence>1||iou<0||iou>1)
        throw std::invalid_argument("invalid YOLO output, geometry or thresholds");
    std::map<int,std::vector<Detection>> groups;
    // 一个候选框选一个最高分类分数；id=-1表示没有类别达到confidence门槛。
    for(int n=0;n<count;++n){
        float score=confidence;int id=-1;
        // 扁平数组按组存放：所有x、所有y、所有w、所有h、各类分数，每组8400个。
        // (4+c)*count+n先跳过坐标组及之前的类别组，再取第n个框的第c类分数。
        for(int c=0;c<classes;++c){float s=output[(4+c)*count+n];
            if(std::isfinite(s)&&s>=score&&s<=1){score=s;id=c;}}
        if(id<0)continue;
        float x=output[n],y=output[count+n],w=output[2*count+n],h=output[3*count+n];
        if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0)continue;
        // 先限制浮点坐标到原图边界，再四舍五入转整数，避免越界及过早整数转换。
        auto bound=[](double v,int upper){return int(std::round(std::clamp(v,0.,double(upper))));};
        // 中心坐标加/减半个宽高得到四边；先减缩小图上的补边，再除实际缩放比例。
        // 例如原图1280x720、scale_y=0.5、pad_y=140，y=300/h=80时top=(300-40-140)/0.5=240。
        int left=bound((double(x)-w/2.-info.pad_x)/info.scale_x,size.width);
        int top=bound((double(y)-h/2.-info.pad_y)/info.scale_y,size.height);
        int right=bound((double(x)+w/2.-info.pad_x)/info.scale_x,size.width);
        int bottom=bound((double(y)+h/2.-info.pad_y)/info.scale_y,size.height);
        // cv::Rect需要左上角+宽高，而不是右下角；裁剪后零面积的框直接丢弃。
        if(right>left&&bottom>top)groups[id].push_back({{left,top,right-left,bottom-top},score,id});
    }
    std::vector<Detection> result;
    // 按类别分别NMS：IoU=交集面积/并集面积，分数高的先保留，重叠超过阈值的同类框被抑制。
    // 不同类别即使完全重合也不互相抑制；同分时不额外定义业务优先级。
    for(auto& entry:groups){std::vector<cv::Rect> boxes;std::vector<float> scores;std::vector<int> keep;
        for(const auto& d:entry.second){boxes.push_back(d.box);scores.push_back(d.score);}
        // boxes/scores下标一一对应；keep返回当前类别列表的下标，不是类别编号。
        // 正分数门槛向下微调且不低于0，使边界与前面的>=筛选衔接。
        cv::dnn::NMSBoxes(boxes,scores,std::max(0.f,confidence-1e-6f),iou,keep);
        for(int k:keep)result.push_back(entry.second.at(k));}
    std::stable_sort(result.begin(),result.end(),[](const auto&a,const auto&b){return a.score>b.score;});return result;
}
// image是可修改引用，直接在调用方的图片上绘制；另两个const引用只读取检测记录和名称。
void draw_detections(cv::Mat& image,const std::vector<Detection>& detections,const std::vector<std::string>& names){
    for(const auto& d:detections){cv::rectangle(image,d.box,{0,255,0},2);
        std::string label=names.at(d.class_id)+" "+std::to_string(d.score).substr(0,4);
        cv::putText(image,label,{d.box.x,std::max(15,d.box.y)},cv::FONT_HERSHEY_SIMPLEX,0.5,{0,180,0},1);}
}
