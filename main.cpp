#include "pipeline.h"

int main()
{
    Pipeline pipeline(
        "/home/xct/yolo-cpp-detector/test.mp4",
        "/home/xct/yolo-cpp-detector/yolo11n.onnx",
        "/home/xct/yolo-cpp-detector/coco.txt"
    );

    pipeline.run();

    return 0;
}
