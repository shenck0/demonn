#pragma once
#include "common.h"

namespace demonn {

    export_symbol void fill_onehot(
        int batch_size,
        int n,
        const int* label, // (batch_size,)
        float* output // (batch_size, n)
    );

    export_symbol void fill_constant(
        int count,
        float value,
        float* output // (count,)
    );

    export_symbol void fill_normal_distribution(
        float mean, float std_dev,
        int n,
        float* output // (n,)
    );

}