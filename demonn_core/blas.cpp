#include "blas.h"
#ifdef compile_implement_mkl
    #include <mkl.h>
    #include <mkl_cblas.h>
#endif

namespace demonn {

    // C = alpha*op(A)*op(B) + beta*C, do matrix multiply: (M,K) x (K,N)
    void gemm_mkl(
        const float * A, bool transpose_A,
        const float * B, bool transpose_B,
        int M, int N, int K,
        float alpha, float beta,
        float * C
    ) {
#ifdef compile_implement_mkl
        // https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
        cblas_sgemm(
            CblasRowMajor,
            transpose_A ? CblasTrans : CblasNoTrans,
            transpose_B ? CblasTrans : CblasNoTrans,
            M, N, K,
            alpha,
            A, /*lda*/ transpose_A ? M : K,
            B, /*ldb*/ transpose_B ? K : N,
            beta,
            C, /*ldc*/ N
        );
#endif
    }

    // compressed[i] = full_storage[indices[i]] i=0,1,...,(n-1)
    void gather_mkl(
        int n,
        const float* full_storage, // at least (max(indices[0..n))+1,)
        const int* indices, // (n,)
        float* compressed // (n,)
    ) {
#ifdef compile_implement_mkl
        // https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gthr
        cblas_sgthr(n, full_storage, compressed, indices);
#endif
    }

}