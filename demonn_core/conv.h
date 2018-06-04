#pragma once
#include "common.h"

namespace demonn {

    struct export_symbol conv2d_descriptor {
        int input_height, input_width;
        int input_channel, output_channel;
        int filter_height, filter_width;
        int pad_top, pad_bottom, pad_left, pad_right;
        int stride_x, stride_y;
        // calculated
        int output_height, output_width;

        conv2d_descriptor(
            int input_height, int input_width,
            int input_channel, int output_channel,
            int filter_height, int filter_width,
            int pad_top, int pad_bottom, int pad_left, int pad_right,
            int stride_x, int stride_y
        );
    };

    struct export_symbol conv2d_im2col_res {
        float* im2col_buffer; // (filter_height*filter_width*input_channel, output_height*output_width)
        int im2col_buffer_length;
        int* im2col_index; // same size as im2col_buffer
        int* im2col_index_pad;
        int im2col_index_pad_length;
        float* bias_multiplier; // at least: (output_height*output_width,)

        conv2d_im2col_res(const conv2d_descriptor & desc);
        ~conv2d_im2col_res();
    };
    
    export_symbol void op(conv2d, im2col, forward, cpu, mkl)(
        const conv2d_descriptor & desc,
        const conv2d_im2col_res & res,
        int batch_size,
        const float* input, // (batch_size, input_channel, input_height, input_width)
        const float* filter, // (output_channel, input_channel, filter_height, filter_width)
        const float* bias, // (output_channel)
        float* output // (batch_size, output_channel, output_height, output_width)
    );

    export_symbol void op(conv2d, im2col, backward, cpu, mkl)(
        const conv2d_descriptor & desc,
        const conv2d_im2col_res & res,
        int batch_size,
        const float* input, // (batch_size, input_channel, input_height, input_width)
        const float* filter, // (output_channel, input_channel, filter_height, filter_width)
        float* grad_output, // (batch_size, output_channel, output_height, output_width) // in-out
        float* grad_filter, // (output_channel, input_channel, filter_height, filter_width)
        float* grad_bias, // (output_channel)
        float* grad_input // (batch_size, input_channel, input_height, input_width)
    );

}