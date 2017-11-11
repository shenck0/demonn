#pragma once
#include "common.h"

namespace demonn_core {

    EXPORT_SYMBOL void relu_forward(
        const float* input, int batch_size, int input_neuron,
        float* output
    );

    EXPORT_SYMBOL void relu_backward(
        const float* before_relu, int batch_size, int neuron_num,
        const float* grad_input,
        float* grad_output
    );

    EXPORT_SYMBOL void softmax_forward(
        const float* input, int batch_size, int class_num,
        float* output
    );

    EXPORT_SYMBOL void softmax_backward(
        const float* softmax_output, int batch_size, int class_num, // softmax_output:(batch_size, class_num)
        const float* grad_input, // grad_input:(batch_size, class_num)
        float* grad_output // grad_output:(batch_size, class_num)
    );

}