#include <gtest/gtest.h>
#include <demonn_core.h>
#include "common.hpp"

namespace d = demonn;
namespace dc = demonn_core;

TEST(activation, relu_forward) {
    float input[] = { 1.0F, -1.0F, 0.0F, -0.5F, 0.5F, 0.0F };
    float answer[] = { 1.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F };
    float output[6];
    dc::relu_forward(input, 3, 2, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, relu_backward) {
    float before_relu[] = {1.0F, -1.0F, 0.5F, -0.5F, -0.42F, 0.42F};
    float grad_input[] = { 3.0F, -3.0F, -3.0F, 3.0F, -3.0F, -3.0F };
    float answer[] = {3.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.42F};
    float output[6];
    dc::relu_backward(before_relu, 2, 3, grad_input, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, softmax_forward) {
    float input[] = { 1.0F };
    float answer[] = { 1.0F };
    float output[6];
    dc::softmax_forward(input, 2, 3, output);
    TestCommon::check_float_array(answer, output, 6);
}

TEST(activation, softmax_backward) {

}