#pragma once
#include "common.h"

namespace demonn_core {

    struct conv2d_descriptor
    {
        int input_channel, input_height, input_width;
        int filter_height, filter_width;
        int pad_top, pad_bottom, pad_left, pad_right;
        int stride_x, stride_y;
        int output_channel;

        int output_height, output_width;
        int* im2col_index;
        float* im2col_buffer;
        int* im2col_index_pad;
        int im2col_buffer_length;
        int im2col_index_pad_length;
    };

    EXPORT_SYMBOL conv2d_descriptor create_conv2d_descriptor(
        int input_channel, int input_height, int input_width,
        int filter_height, int filter_width,
        int output_channel,
        int pad_top = 0, int pad_bottom = 0, int pad_left = 0, int pad_right = 0,
        int stride_x = 1, int stride_y = 1
    );

    EXPORT_SYMBOL void dispose_conv2d_descriptor(
        conv2d_descriptor& desc
    );

    // convolution forward
    EXPORT_SYMBOL void conv2d_forward(
        conv2d_descriptor& desc,
        const float* input, int batch_size, // input:(batch_size, input_channel, input_height, input_width)
        const float* filter, const float* bias, // filter:(filter_height, filter_width), bias:(output_channel)
        const float* bias_multiplier, // bias_multiplier:(output_height*output_width,1)
        float* output
    );

}