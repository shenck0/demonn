#include "conv.h"

namespace demonn_core {

    void conv2d_forward(
        const float* input, int batch_size, int input_channel, int height, int width, // input:(batch_size, input_channel, height, width)
        const float* filter, int filter_height, int filter_width, // filter:(filter_height, filter_width)
        const float* bias, // bias:(output_channel)
        void* workspace, //workspace:(???)
        int pad_top, int pad_bottom, int pad_left, int pad_right,
        int stride_x, int stride_y,
        float* output, int output_channel
    ) {
        //TODO: implement
        // im2col
        //         filter       .       im2col
        // (n(out) x a(f)a(in)) . (a(f)a(in) x a(out))
        for (int bi = 0; bi < batch_size; bi++) {
            
        }

        // do convolution


    }

}