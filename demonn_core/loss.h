#pragma once
#include "common.h"

namespace demonn {

    export_symbol void op(cross_entropy, direct, forward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, class_num)
        const float* label, // (batch_size, class_num)
        float* output // (batch_size,) 
    );

    export_symbol void op(cross_entropy, direct, backward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, class_num)
        const float* label, // (batch_size, class_num)
        const float* grad_output, // (batch_size,)
        float* grad_input // (batch_size, class_num)
    );

}