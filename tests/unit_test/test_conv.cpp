#include <gtest/gtest.h>
#include <demonn_core.h>
#include "common.hpp"

namespace d = demonn;
namespace dc = demonn_core;

TEST(conv, create_desc_1) {
    auto desc = dc::create_conv2d_descriptor(1, 3, 3, 2, 2, 1);
    EXPECT_EQ(2, desc.filter_height);
    EXPECT_EQ(2, desc.output_height);
    EXPECT_EQ(2, desc.output_width);
    EXPECT_NE(nullptr, desc.im2col_index);
    EXPECT_NE(nullptr, desc.im2col_buffer);
}

TEST(conv, simple_forward) {
    auto desc = dc::create_conv2d_descriptor(1, 3, 3, 2, 2, 1);
    float input[] = {
        1, 2, 3,
        4, 0, 4,
        3, 2, 1
    };
    float filter[] = {
        -1, 1,
         0, 2
    };
    float bias[] = {
        1,
    };
    float output[4];
    auto bias_multiplier = TestCommon::get_bias_multiplier(4);
    dc::conv2d_forward(desc, input, 1, filter, bias, bias_multiplier.get(), output);
    float output_check[4] = {
        2, 10,
        1, 7
    };
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(output_check[i], output[i]);
    }

}

TEST(conv, simple_forward_stride) {

}

TEST(conv, simple_forward_padding) {

}