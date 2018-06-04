#include <gtest/gtest.h>
#include <demonn.h>
#include "common.hpp"

TEST(activation, relu_direct_forward_cpu_cpp) {
    float input[] = { 1.0f, -1.0f, 0.0f, -0.5f, 0.5f, 0.0f };
    float answer[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f };
    float output[6];
    d::relu_direct_forward_cpu_cpp(3, 2, input, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, relu_direct_backward_cpu_cpp) {
    float input[] = {1.0f, -1.0f, 0.5f, -0.5f, -0.42f, 0.42f};
    float grad_output[] = { 3.0f, -3.0f, -3.0f, 3.0f, -3.0f, -3.0f };
    float answer[] = {3.0f, 0.0f, -3.0f, 0.0f, 0.0f, -3.0f};
    float grad_input[6];
    d::relu_direct_backward_cpu_cpp(2, 3, input, grad_output, grad_input);
    TestCommon::check_float_array(answer, grad_input, 6);
}

TEST(activation, softmax_stable_forward_cpu_mkl) {
    float input[12];
    float answer[12];
    float output[12];
    d::fill_normal_distribution(0.0f, 0.1f, 12, input);
    d::softmax_stable_forward_cpu_mkl(2, 6, input, output);
    for (int bi = 0; bi < 2; bi++) {
        float sum = 0.0f;
        for (int i = 0; i < 6; i++)
            sum += expf(input[bi * 6 + i]);
        for (int i = 0; i < 6; i++)
            answer[bi * 6 + i] = expf(input[bi * 6 + i]) / sum;
    }
    TestCommon::check_float_array(answer, output, 6);
}