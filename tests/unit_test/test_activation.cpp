#include <gtest/gtest.h>
#include <demonn_core.h>
#include "common.hpp"

namespace d = demonn;
namespace dc = demonn_core;

TEST(activation, relu_forward) {
    float input[] = { 1.0f, -1.0f, 0.0f, -0.5f, 0.5f, 0.0f };
    float answer[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f };
    float output[6];
    dc::relu_forward(input, 3, 2, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, relu_backward) {
    float before_relu[] = {1.0f, -1.0f, 0.5f, -0.5f, -0.42f, 0.42f};
    float grad_input[] = { 3.0f, -3.0f, -3.0f, 3.0f, -3.0f, -3.0f };
    float answer[] = {3.0f, 0.0f, -3.0f, 0.0f, 0.0f, -3.0f};
    float output[6];
    dc::relu_backward(before_relu, 2, 3, grad_input, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, softmax_forward) {
    float input[12];
    float answer[12];
    float output[12];
    dc::normal_distribution(input, 12, 0.0f, 0.1f);
    dc::softmax_forward(input, 2, 6, output);
    for (int bi = 0; bi < 2; bi++) {
        float sum = 0.0f;
        for (int i = 0; i < 6; i++)
            sum += expf(input[bi * 6 + i]);
        for (int i = 0; i < 6; i++)
            answer[bi * 6 + i] = expf(input[bi * 6 + i]) / sum;
    }
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, softmax_backward) {
    //TODO: gradient check
}