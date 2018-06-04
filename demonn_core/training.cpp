#include "training.h"
#ifdef compile_implement_mkl
    #include <mkl_cblas.h>
#endif

namespace demonn {

    void op(sgd, plain, none, cpu, mkl)(
        int n,
        const float* grad_weight, // (n,)
        float learning_rate, 
        float* weight // (n,)
    ) {
#ifdef compile_implement_mkl
        cblas_saxpy(n, -learning_rate, grad_weight, 1, weight, 1);
#endif
    }

}