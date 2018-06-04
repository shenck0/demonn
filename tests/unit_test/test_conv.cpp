#include <gtest/gtest.h>
#include <demonn.h>
#include "common.hpp"

namespace d = demonn;

TEST(conv, conv2d_descriptor) {
    d::conv2d_descriptor desc(3, 3, 1, 3, 2, 2, 0, 0, 0, 0, 1, 1);
    EXPECT_EQ(2, desc.output_height);
    EXPECT_EQ(2, desc.output_width);
}

TEST(conv, conv2d_im2col_forward_cpu_mkl) {
    d::conv2d_descriptor desc(3, 3, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1);
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
    d::conv2d_im2col_res res(desc);
    d::conv2d_im2col_forward_cpu_mkl(desc, res, 1, input, filter, bias, output);
    float output_check[4] = {
        2, 10,
        1, 7
    };
    TestCommon::check_float_array(output_check, output, 4);
}