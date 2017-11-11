#include "training.h"
#include <mkl_cblas.h>

namespace demonn_core {

    void stochastic_gradient_descent(
        float* weight, int count,
        const float* grad,
        float learning_rate
    ) {
        cblas_saxpy(count, -learning_rate, grad, 1, weight, 1);
    }

}