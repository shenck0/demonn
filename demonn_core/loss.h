#pragma once
#include "common.h"

namespace demonn_core {

    EXPORT_SYMBOL void cross_entropy_forward(
        const float* predict, int batch_size, int class_num,
        const float* label,
        float* output_loss // average loss of batch
    );

    EXPORT_SYMBOL void cross_entropy_backward(
        const float* predict, int batch_size, int class_num, // predict:(batch_size, class_num)
        const float* label, // label:(batch_size, class_num)
        float* grad_output // grad_output:(batch_size, class_num)
    );

}