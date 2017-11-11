#pragma once
#include "common.h"

namespace demonn_core {

    EXPORT_SYMBOL void stochastic_gradient_descent(
        float* weight, int count,
        const float* grad,
        float learning_rate
    );

}