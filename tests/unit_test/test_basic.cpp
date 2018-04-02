#include <gtest/gtest.h>
#include <demonn_core.h>
#include "common.hpp"

namespace d = demonn;
namespace dc = demonn_core;

TEST(basic, argmax) {
    float arr[] = { 5.0f, 1.0f, 3.0f, 4.0f, 2.0f };
    EXPECT_EQ(0, dc::argmax(arr, 5));
}

TEST(basic, onehot) {
    int labels[] = { 1, 2, 0 };
    float outputs[9];
    float answer[] = {
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
    };
    dc::onehot(labels, 3, 3, outputs);
    TestCommon::check_float_array(outputs, answer, 9);
}

TEST(basic, mean) {
    float input[] = { -1.0f, 1.0f, 3.0f };
    EXPECT_FLOAT_EQ(1.0f, dc::mean(input, 3));
}