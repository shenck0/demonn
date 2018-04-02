#include <cstring>
#include <cstdlib>
#include "conv.h"
#include "blas.hpp"

namespace demonn_core {

    conv2d_descriptor create_conv2d_descriptor(
        int input_channel, int input_height, int input_width,
        int filter_height, int filter_width,
        int output_channel,
        int pad_top, int pad_bottom, int pad_left, int pad_right,
        int stride_x, int stride_y
    ) {
        conv2d_descriptor ret;
        ret.input_channel = input_channel, ret.input_height = input_height, ret.input_width = input_width;
        ret.filter_height = filter_height, ret.filter_width = filter_width;
        ret.output_channel = output_channel;
        ret.pad_top = pad_top, ret.pad_bottom = pad_bottom, ret.pad_left = pad_left, ret.pad_right = pad_right;
        ret.stride_x = stride_x, ret.stride_y = stride_y;
    
        // initialize im2col buffers
        //TODO: implement pad \ stride
        ret.output_height = input_height - filter_height + 1;
        ret.output_width = input_width - filter_width + 1;

        int buffer_width = ret.output_width * ret.output_height;
        int buffer_height = filter_height * filter_width * input_channel;
        ret.im2col_buffer_length = buffer_width * buffer_height;
        ret.im2col_index = (int*)malloc(sizeof(int) * ret.im2col_buffer_length);
        ret.im2col_buffer = (float*)malloc(sizeof(float) * ret.im2col_buffer_length);
        ret.im2col_index_pad = (int*)malloc(sizeof(int) * ret.im2col_buffer_length);
        for (int i = 0; i < ret.im2col_buffer_length; i++)
            ret.im2col_index[i] = -1;

        for (int by = 0; by < ret.output_height; by++) {
            for (int bx = 0; bx < ret.output_width; bx++) {
                for (int ci = 0; ci < input_channel; ci++) {
                    for (int fy = 0; fy < filter_height; fy++) {
                        for (int fx = 0; fx < filter_width; fx++) {
                            int buffer_row = ci * filter_height * filter_width + fy * filter_width + fx;
                            int buffer_col = by * ret.output_width + bx;
                            ret.im2col_index[buffer_row * buffer_width + buffer_col] =
                                input_height * input_width * ci + input_width * (by + fy) + (bx + fx);
                        }
                    }
                }
            }
        }
        int last_index = 0;
        ret.im2col_index_pad_length = 0;
        for (int i = 0; i < ret.im2col_buffer_length; i++) {
            if (ret.im2col_index[i] < 0) {
                ret.im2col_index[i] = last_index;
                ret.im2col_index_pad[ret.im2col_index_pad_length++] = i;
            } else {
                last_index = ret.im2col_index[i];
            }
        }
        return ret;
    }

    void dispose_conv2d_descriptor(
        conv2d_descriptor& desc
    ) {
        free_and_clear(desc.im2col_index);
        free_and_clear(desc.im2col_buffer);
        free_and_clear(desc.im2col_index_pad);
        memset(&desc, 0, sizeof(conv2d_descriptor));
    }

    void conv2d_forward(
        conv2d_descriptor& desc,
        const float* input, int batch_size, // input:(batch_size, input_channel, input_height, input_width)
        const float* filter, const float* bias, // filter:(filter_height, filter_width), bias:(output_channel)
        const float* bias_multiplier, // bias_multiplier:(output_height*output_width,1)
        float* output
    ) {
        for (int bi = 0; bi < batch_size; bi++) {
            const float* cur_input = input + bi * (desc.input_channel * desc.input_height * desc.input_width);
            float* cur_output = output + bi * (desc.output_channel * desc.output_height * desc.output_width);
            
            // do im2col:
            gather(cur_input, desc.im2col_index, desc.im2col_buffer, desc.im2col_buffer_length);
            for (int ti = 0; ti < desc.im2col_index_pad_length; ti++)
                desc.im2col_buffer[desc.im2col_index_pad[ti]] = 0;

            // do convolution
            //         filter         .           im2col          =       output
            // (n(out), area(f)n(in)) . (area(f)n(in), area(out)) = (n(out), area(out))
            gemm(
                filter, 
                desc.im2col_buffer, 
                cur_output, 
                desc.output_channel, 
                desc.filter_height * desc.filter_width * desc.input_channel,
                desc.output_height * desc.output_width
            );

            // add bias
            //  bias       . bias_multiplier -->      output
            // (n(out), 1) . (1, area(out))  --> (n(out), area(out))
            gemm(
                bias,
                bias_multiplier,
                output,
                desc.output_channel,
                1,
                desc.output_height * desc.output_width,
                1.0f
            );
        }
    }

}