#pragma once
#include "common.h"

namespace demonn {

    // C = alpha*op(A)*op(B) + beta*C, do matrix multiply: (M,K) x (K,N)
    export_symbol void gemm_mkl(
        const float * A, bool transpose_A,
        const float * B, bool transpose_B,
        int M, int N, int K,
        float alpha, float beta,
        float * C
    );

    // compressed[i] = full_storage[indices[i]] i=0,1,...,(n-1)
    export_symbol void gather_mkl(
        int n,
        const float* full_storage, // at least (max(indices[0..n))+1,)
        const int* indices, // (n,)
        float* compressed // (n,)
    );

} 


