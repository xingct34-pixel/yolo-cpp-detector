#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <stdexcept>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
void stats(cv::FileStorage& fs,const char* name,const Benchmark& b){
    fs<<name<<"{"<<"samples"<<double(b.count())<<"mean_ms"<<b.average()<<"p50_ms"<<b.p50()<<"p95_ms"<<b.p95()<<"}";}
}
Pipeline::Pipeline(const std::string& video,const std::string& model,const std::string& classes,
    Detector::BackendType type,const std::string& engine,bool cuda,size_t capacity,PipelineOptions options):
    frame_queue_(capacity,options.policy),result_queue_(capacity,options.policy),
    detector_(model,classes,type,engine,cuda),video_path_(video),options_(std::move(options)){}
void Pipeline::requestStop(std::exception_ptr error){
    if(error){std::lock_guard<std::mutex> lock(error_mutex_);if(!error_)error_=error;}
    stop_requested_=true;frame_queue_.close();result_queue_.close();
}
void Pipeline::readLoop(){
    cv::VideoCapture cap(video_path_);if(!cap.isOpened())throw std::runtime_error("视频打开失败："+video_path_);
    cv::Mat image;int id=0;
    while(!stop_requested_ && (!options_.max_frames||id<options_.max_frames) && cap.read(image)){
        FrameData data{id++,image.clone(),Clock::now()};++read_count_;
        if(!frame_queue_.push(std::move(data)))break;
    }
    frame_queue_.close();
}
void Pipeline::collect(){
    inference_.record(detector_.last_inference_ms());preprocess_.record(detector_.last_preprocess_ms());
    backend_.record(detector_.last_backend_call_ms());postprocess_.record(detector_.last_postprocess_ms());
}
void Pipeline::inferenceLoop(){
    FrameData data;
    while(!stop_requested_&&frame_queue_.pop(data)){
        queue_wait_.record(ms(data.captured,Clock::now()));
        auto image=detector_.detect(data.frame);++inference_count_;collect();
        if(!result_queue_.push({data.frame_id,std::move(image),data.captured}))break;
    }
    result_queue_.close();
}
void Pipeline::consume(ResultData& result){
    if(options_.show){cv::imshow("YOLO Result",result.frame);if(cv::waitKey(1)==27)requestStop();}
    ++display_count_;end_to_end_.record(ms(result.captured,Clock::now()));
}
void Pipeline::display(){ResultData result;while(!stop_requested_&&result_queue_.pop(result))consume(result);}
void Pipeline::run(){
    if(options_.warmup<0||options_.max_frames<0)throw std::invalid_argument("invalid pipeline limits");
    for(int i=0;i<options_.warmup;++i)(void)detector_.detect(cv::Mat(640,640,CV_8UC3,cv::Scalar{}));
    auto begin=Clock::now();std::thread reader,infer,monitor;std::atomic<bool> finished{false};
    try {
        if(options_.external_stop)monitor=std::thread([&]{while(!finished){
            if(options_.external_stop->load()){requestStop();break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));}});
        if(options_.sequential){
            cv::VideoCapture cap(video_path_);if(!cap.isOpened())throw std::runtime_error("视频打开失败："+video_path_);
            cv::Mat frame;int id=0;
            while(!stop_requested_&&(!options_.max_frames||id<options_.max_frames)&&cap.read(frame)){
                auto captured=Clock::now();++read_count_;auto result=detector_.detect(frame);++inference_count_;collect();
                queue_wait_.record(0);ResultData data{id++,std::move(result),captured};consume(data);
            }
        }else{
            reader=std::thread([&]{try{readLoop();}catch(...){requestStop(std::current_exception());}});
            infer=std::thread([&]{try{inferenceLoop();}catch(...){requestStop(std::current_exception());}});
            display();
        }
    }catch(...){requestStop(std::current_exception());}
    if(reader.joinable())reader.join();if(infer.joinable())infer.join();finished=true;
    if(monitor.joinable())monitor.join();if(options_.show)cv::destroyAllWindows();
    if(error_)std::rethrow_exception(error_);
    double seconds=ms(begin,Clock::now())/1000.;
    std::cout<<"读取帧数："<<read_count_<<"\n推理帧数："<<inference_count_<<"\n输出帧数："<<display_count_
        <<"\nPipeline FPS："<<(seconds>0?display_count_/seconds:0)<<"\n丢弃输入/结果帧："<<frame_queue_.dropped()<<"/"<<result_queue_.dropped()<<'\n';
    inference_.print(detector_.backend_name().c_str());
    if(!options_.report.empty()){
        cv::FileStorage fs(options_.report,cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
        if(!fs.isOpened())throw std::runtime_error("cannot write report: "+options_.report);
        fs<<"schema_version"<<1<<"mode"<<(options_.sequential?"sequential":"pipeline")
            <<"queue_policy"<<(options_.policy==QueuePolicy::Block?"block":"drop-oldest")<<"backend"<<detector_.backend_name()
            <<"show"<<int(options_.show)<<"warmup"<<options_.warmup<<"stopped"<<int(stop_requested_.load())
            <<"read_frames"<<double(read_count_)<<"inferred_frames"<<double(inference_count_)<<"output_frames"<<double(display_count_)
            <<"input_dropped"<<double(frame_queue_.dropped())<<"result_dropped"<<double(result_queue_.dropped())
            <<"input_queue_peak"<<double(frame_queue_.peak())<<"result_queue_peak"<<double(result_queue_.peak())
            <<"elapsed_seconds"<<seconds<<"output_fps"<<(seconds>0?display_count_/seconds:0);
        stats(fs,"inference",inference_);stats(fs,"preprocess",preprocess_);stats(fs,"backend_api",backend_);
        stats(fs,"postprocess",postprocess_);stats(fs,"queue_wait",queue_wait_);stats(fs,"end_to_end",end_to_end_);
    }
}
