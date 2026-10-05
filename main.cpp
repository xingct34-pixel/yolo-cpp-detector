#include "pipeline.h"

int main()
{
    Pipeline pipeline(
        "/home/xct/yolo-cpp-detector/test.mp4",    //视频路径
        "/home/xct/yolo-cpp-detector/yolo11n.onnx",  //onnx模型路径
        "/home/xct/yolo-cpp-detector/coco.txt"    //coco类别名文件路径
    );

    pipeline.run();   //开始读帧

    return 0;
}
