#pragma once
#include "common.h"

namespace demonn_core {

    EXPORT_SYMBOL void fully_connected_forward(
        const float* input, int batch_size, int input_neuron, // input:(batch_size, input_neuron)
        const float* weight, const float* bias, // weight:(input_neuron, output_neuron)
        const float* bias_multiplier, // bias_multiplier:(batch_size)
        float* output, int output_neuron // output:(batch_size, output_neuron)
    );

    EXPORT_SYMBOL void fully_connected_backward(
        const float* fc_input, int batch_size, int input_neuron, int output_neuron, // fc_input:(batch_size, input_neuron)
        const float* weight, // weight:(input_neuron, output_neuron)
        const float* bias_multiplier, // bias_multiplier:(batch_size)
        float* bias_grad, // bias_grad:(output_neuron)
        float* grad_weight, // grad_weight:(input_neuron, output_neuron
        const float* grad_input, // grad_input:(batch_size, output_neuron)
        float* grad_output // grad_output:(batch_size, input_neuron)
    );

}

namespace demonn_core {

    EXPORT_SYMBOL int argmax(float* arr, int n);

    EXPORT_SYMBOL void onehot(const int* labels, int batch_size, int class_num,
        float* output);

    EXPORT_SYMBOL float mean(const float* arr, int count);

    EXPORT_SYMBOL void set_array(float* arr, float value, int count);

}