#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

int main()
{
    Logger logger;

    IBuilder* builder = createInferBuilder(logger);

    return 0;
}
