#pragma once
#include "common.h"

namespace demonn {

    export_symbol void op(sgd, plain, none, cpu, mkl)(
        int n,
        const float* grad_weight, // (n,)
        float learning_rate,
        float* weight // (n,)
    );

}