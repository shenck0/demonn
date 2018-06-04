#include <cstring>
#include <cstdlib>
#include "conv.h"
#include "blas.h"

namespace demonn {

    conv2d_descriptor::conv2d_descriptor(
        int input_height, int input_width,
        int input_channel, int output_channel,
        int filter_height, int filter_width,
        int pad_top, int pad_bottom, int pad_left, int pad_right,
        int stride_x, int stride_y
    ) {
        this->input_channel = input_channel, this->input_height = input_height, this->input_width = input_width;
        this->filter_height = filter_height, this->filter_width = filter_width;
        this->output_channel = output_channel;
        this->pad_top = pad_top, this->pad_bottom = pad_bottom, this->pad_left = pad_left, this->pad_right = pad_right;
        this->stride_x = stride_x, this->stride_y = stride_y;

        // initialize im2col buffers
        //TODO: implement pad \ stride
        this->output_height = input_height - filter_height + 1;
        this->output_width = input_width - filter_width + 1;
    }

    conv2d_im2col_res::conv2d_im2col_res(const conv2d_descriptor & desc) {
        int buffer_width = desc.output_width * desc.output_height;
        int buffer_height = desc.filter_height * desc.filter_width * desc.input_channel;

        im2col_buffer_length = buffer_width * buffer_height;
        im2col_index = (int*)malloc(sizeof(int) * im2col_buffer_length);
        im2col_buffer = (float*)malloc(sizeof(float) * im2col_buffer_length);
        im2col_index_pad = (int*)malloc(sizeof(int) * im2col_buffer_length);
        bias_multiplier = (float*)malloc(sizeof(float) * desc.output_height * desc.output_width);
        for (int i = 0; i < desc.output_height*desc.output_width; i++)
            bias_multiplier[i] = 1.0f;
        for (int i = 0; i < im2col_buffer_length; i++)
            im2col_index[i] = -1;

        for (int by = 0; by < desc.output_height; by++) {
            for (int bx = 0; bx < desc.output_width; bx++) {
                for (int ci = 0; ci < desc.input_channel; ci++) {
                    for (int fy = 0; fy < desc.filter_height; fy++) {
                        for (int fx = 0; fx < desc.filter_width; fx++) {
                            int buffer_row = ci * desc.filter_height * desc.filter_width + fy * desc.filter_width + fx;
                            int buffer_col = by * desc.output_width + bx;
                            im2col_index[buffer_row * buffer_width + buffer_col] =
                                desc.input_height * desc.input_width * ci + desc.input_width * (by + fy) + (bx + fx);
                        }
                    }
                }
            }
        }
        int last_index = 0;
        im2col_index_pad_length = 0;
        for (int i = 0; i < im2col_buffer_length; i++) {
            if (im2col_index[i] < 0) {
                im2col_index[i] = last_index;
                im2col_index_pad[im2col_index_pad_length++] = i;
            } else {
                last_index = im2col_index[i];
            }
        }
    }

    conv2d_im2col_res::~conv2d_im2col_res() {
        free_and_clear(im2col_index);
        free_and_clear(im2col_buffer);
        free_and_clear(im2col_index_pad);
        free_and_clear(bias_multiplier);
    }

    void _im2col(
        const conv2d_im2col_res & impl,
        const float* input, // (input_channel, input_height, input_width)
        float* output
    ) {
        gather_mkl(impl.im2col_buffer_length, input, impl.im2col_index, output);
        for (int ti = 0; ti < impl.im2col_index_pad_length; ti++)
            output[impl.im2col_index_pad[ti]] = 0;
    }

    void op(conv2d, im2col, forward, cpu, mkl)(
        const conv2d_descriptor & desc,
        const conv2d_im2col_res & res,
        int batch_size,
        const float* input, // (batch_size, input_channel, input_height, input_width)
        const float* filter, // (output_channel, input_channel, filter_height, filter_width)
        const float* bias, // (output_channel)
        float* output // (batch_size, output_channel, output_height, output_width)
    ) {
        for (int bi = 0; bi < batch_size; bi++) {
            const float* cur_input = input + bi * (desc.input_channel * desc.input_height * desc.input_width);
            float* cur_output = output + bi * (desc.output_channel * desc.output_height * desc.output_width);
            
            // do im2col:
            _im2col(res, cur_input, res.im2col_buffer);

            // do convolution
            gemm_mkl(
                filter, false,
                res.im2col_buffer, false,
                desc.output_channel, desc.output_height * desc.output_width, desc.filter_height * desc.filter_width * desc.input_channel,
                1.0f, 0.0f,
                cur_output
            );

            // add bias
            gemm_mkl(
                bias, false,
                res.bias_multiplier, false,
                desc.output_channel, desc.output_height * desc.output_width, 1,
                1.0f, 1.0f,
                cur_output
            );
        }
    }

    void op(conv2d, im2col, backward, cpu, mkl)(
        const conv2d_descriptor & desc,
        const conv2d_im2col_res & res,
        int batch_size,
        const float* input, // (batch_size, input_channel, input_height, input_width)
        const float* filter, // (output_channel, input_channel, filter_height, filter_width)
        float* grad_output, // (batch_size, output_channel, output_height, output_width) // in-out
        float* grad_filter, // (output_channel, input_channel, filter_height, filter_width)
        float* grad_bias, // (output_channel)
        float* grad_input // (batch_size, input_channel, input_height, input_width)
    ) {
        memset(grad_bias, 0, sizeof(float) * desc.output_channel);
        memset(grad_filter, 0, sizeof(float) * desc.output_channel * desc.input_channel * desc.filter_height * desc.filter_width);
        const int output_sample_length = desc.output_channel * desc.output_height * desc.output_width;
        const int input_sample_length = desc.input_channel * desc.input_height * desc.input_width;
        for (int bi = 0; bi < batch_size; bi++) {
            float* cur_grad_output = grad_output + bi * output_sample_length;
            const float* cur_input = input + bi * input_sample_length;
            float* cur_grad_input = grad_input + bi * input_sample_length;
            // graident: filter
            // im2col_buffer of current sample
            _im2col(desc, cur_input, res.im2col_buffer);

            // grad_filter += grad_output[bi] . im2col'
            gemm_mkl(
                cur_grad_output, false,
                res.im2col_buffer, true,
                desc.output_channel, desc.input_channel * desc.filter_height * desc.filter_width, desc.output_height * desc.output_width,
                1.0f, 1.0f,
                grad_filter
            );

            // gradient: bias
            // grad_bias += grad_output[bi] . bias_multiplier
            gemm_mkl(
                cur_grad_output, false,
                res.bias_multiplier, false,
                desc.output_channel, 1, desc.output_height * desc.output_width,
                1.0f, 1.0f,
                grad_bias
            );
            
            // gradient: input
            // gradient of current im2col_buffer
            gemm_mkl(
                filter, true,
                cur_grad_output, false,
                desc.filter_height * desc.filter_width * desc.input_channel,
                desc.output_height * desc.output_width,
                desc.output_channel,
                1.0f, 0.0f,
                res.im2col_buffer
            );

            // gradient of input
            for (int i = 0; i < res.im2col_index_pad_length; i++) {
                res.im2col_buffer[res.im2col_index_pad[i]] = 0.0f;
            }
            for (int i = 0; i < res.im2col_buffer_length; i++) {
                cur_grad_input[res.im2col_index[i]] += res.im2col_buffer[i];
            }
        }

    }

}