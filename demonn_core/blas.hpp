#pragma once
#include <mkl.h>

// Matrix multiply: C=beta*C + A*B, A:(row_a, col_a), B:(col_a, col_b)
inline void gemm(const float* A, const float* B, float* C, int row_a, int col_a, int col_b, float beta=0.0F) {
    // https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_a, col_b, col_a, 1.0F, A, col_a, B, col_b, beta, C, col_b);
}

// Matrix multiply: C=A*B', A:(row_a, col_a), B:(col_b, col_a)
inline void gemm_trans_b(const float* A, const float* B, float* C, int row_a, int col_a, int col_b) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, row_a, col_b, col_a, 1.0F, A, col_a, B, col_a, 0.0F, C, col_b);
}

// Matrix multiply: C=A'*B, A:(col_a, row_a), B:(col_a, col_b)
inline void gemm_trans_a(const float* A, const float* B, float* C, int row_a, int col_a, int col_b) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, row_a, col_b, col_a, 1.0F, A, row_a, B, col_b, 0.0F, C, col_b);
}

// compressed[i] = full_storage[indices[i]] i=0,1,...,(n-1)
inline void gather(const float* full_storage, const int* indices, float* compressed, int n) {
    // https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gthr
    cblas_sgthr(n, full_storage, compressed, indices);
}

// B=exp(A) A:(count)
inline void exp(const float* A, float* B, int count) {
    // https://software.intel.com/en-us/mkl-developer-reference-c-v-exp
    vsExp(count, A, B);
}