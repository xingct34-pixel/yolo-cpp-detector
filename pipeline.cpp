#include "pipeline.h"

#include <iostream>

using namespace std;
using namespace cv;

Pipeline::Pipeline(const string& video_path,
                   const string& model_path,
                   const string& classes_path)
    : frame_queue_(3),
      result_queue_(3),
      detector_(model_path, classes_path),
      video_path_(video_path)
{
}

void Pipeline::readLoop()
{
    VideoCapture cap(video_path_);

    if (!cap.isOpened())
    {
        cerr << "视频打开失败！" << endl;
        frame_queue_.close();
        return;
    }

    int frame_id = 0;
    Mat frame;

    while (cap.read(frame))
    {
        FrameData data;

        data.frame_id = frame_id++;
        data.frame = frame.clone();

        frame_queue_.push(std::move(data));
    }

    frame_queue_.close();
}

void Pipeline::inferenceLoop()
{
    while (true)
    {
        FrameData data;

        if (!frame_queue_.pop(data))
        {
            break;
        }

        Mat result = detector_.detect(data.frame);

        ResultData result_data;

        result_data.frame_id = data.frame_id;
        result_data.frame = result;

        result_queue_.push(std::move(result_data));
    }

    result_queue_.close();
}

void Pipeline::displayLoop()
{
    int last_displayed_id = -1;

    while (true)
    {
        ResultData result;

        if (!result_queue_.pop(result))
        {
            break;
        }

        if (result.frame_id <= last_displayed_id)
        {
            continue;
        }

        imshow("YOLO Result", result.frame);

        last_displayed_id = result.frame_id;

        if (waitKey(1) == 27)
        {
            break;
        }
    }
}

void Pipeline::run()
{
    thread read_thread(&Pipeline::readLoop, this);
    thread inference_thread(&Pipeline::inferenceLoop, this);
    thread display_thread(&Pipeline::displayLoop, this);

    read_thread.join();
    inference_thread.join();
    display_thread.join();
}
