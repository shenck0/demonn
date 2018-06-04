#pragma once
#include "common.h"

namespace demonn {

    export_symbol void op(relu, direct, forward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, neuron_num)
        float* output // (batch_size, neuron_num)
    );

    export_symbol void op(relu, direct, backward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, neuron_num)
        const float* grad_output, // (batch_size, neuron_num)
        float* grad_input // (batch_size, neuron_num)
    );

    export_symbol void op(softmax, stable, forward, cpu, mkl)(
        int batch_size,
        int class_num,
        const float* input, // (batch_size, class_num)
        float* output // (batch_size, class_num)
    );

    export_symbol void op(softmax, stable, backward, cpu, mkl)(
        int batch_size,
        int class_num,
        const float* output, // (batch_size, class_num)
        const float* grad_output, // (batch_size, class_num)
        float* grad_input // (batch_size, class_num)
    );

}