#pragma once
#include "common.h"

namespace demonn_core {

    // convolution forward
    EXPORT_SYMBOL void conv2d_forward(
        const float* input, int batch_size, int input_channel, int height, int width, // input:(batch_size, input_channel, height, width)
        const float* filter, int filter_height, int filter_width, // filter:(filter_height, filter_width)
        const float* bias, // bias:(output_channel)
        int pad_top, int pad_bottom, int pad_left, int pad_right,
        int stride_x, int stride_y,
        float* output, int output_channel
    );

}