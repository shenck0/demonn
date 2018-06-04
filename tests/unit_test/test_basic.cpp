#include <gtest/gtest.h>
#include <demonn.h>
#include "common.hpp"

TEST(basic, argmax_direct_forward_cpu_cpp) {
    float arr[] = { 5.0f, 1.0f, 3.0f, 4.0f, 2.0f };
    int ret;
    d::argmax_direct_forward_cpu_cpp(1, 5, arr, &ret);
    EXPECT_EQ(0, ret);
}

TEST(basic, fill_onehot) {
    int labels[] = { 1, 2, 0 };
    float outputs[9];
    float answer[] = {
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
    };
    d::fill_onehot(3, 3, labels, outputs);
    TestCommon::check_float_array(outputs, answer, 9);
}

TEST(basic, mean_direct_forward_cpu_cpp) {
    float input[] = { -1.0f, 1.0f, 3.0f };
    float output;
    d::mean_direct_forward_cpu_cpp(3, input, &output);
    EXPECT_FLOAT_EQ(1.0f, output);
}