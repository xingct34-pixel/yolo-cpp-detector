#include "ort_backend.h"

#include <stdexcept>

Ort::SessionOptions OrtBackend::make_session_options()
{
    Ort::SessionOptions options;

    options.SetIntraOpNumThreads(1);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    options.AppendExecutionProvider_CUDA(cuda_options);

    return options;
}

OrtBackend::OrtBackend(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session_(env_, model_path.c_str(), make_session_options())
{
}

std::vector<float> OrtBackend::infer(
    const std::vector<float>& input_data)
{
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {
        1, 3, 640, 640
    };

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    float* output_data =
        output_tensors[0].GetTensorMutableData<float>();

    size_t output_size =
        output_tensors[0]
            .GetTensorTypeAndShapeInfo()
            .GetElementCount();

    return std::vector<float>(
        output_data,
        output_data + output_size);
}
