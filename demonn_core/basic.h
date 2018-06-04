#pragma once
#include "common.h"

namespace demonn {

    export_symbol void op(fully_connected, mm, forward, cpu, mkl)(
        int batch_size,
        int input_neuron,
        const float* input, // (batch_size, input_neuron)
        int output_neuron,
        const float* weight, // (input_neuron, output_neuron)
        const float* bias, // (output_neuron,)
        const float* bias_multiplier, // (batch_size,)
        float* output // (batch_size, output_neuron)
    );

    export_symbol void op(fully_connected, mm, backward, cpu, mkl)(
        int batch_size,
        int input_neuron,
        int output_neuron,
        const float* input, // (batch_size, input_neuron)
        const float* weight, // (input_neuron, output_neuron)
        const float* bias_multiplier, // at least:(batch_size,)
        const float* grad_output, // (batch_size, output_neuron)
        float* grad_bias, // (output_neuron,)
        float* grad_weight, // (input_neuron, output_neuron)
        float* grad_input // (batch_size, input_neuron)
    );

    export_symbol void op(mean, direct, forward, cpu, cpp)(
        int count,
        const float* input, // (count,)
        float* output // (1,)
    );

    export_symbol void op(mean, direct, backward, cpu, cpp)(
        int count,
        const float* grad_output, // (1,)
        float* grad_input // (count,)
    );

    export_symbol void op(argmax, direct, forward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, n)
        int* output// (batch_size,)
    );

}