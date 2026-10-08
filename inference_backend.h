#pragma once

#include <vector>

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual std::vector<float> infer(
        const std::vector<float>& input_data) = 0;
};
