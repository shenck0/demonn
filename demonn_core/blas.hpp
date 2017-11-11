#pragma once
#include <mkl.h>

inline void gemm(const float* A, const float* B, float* C, int row_a, int col_a, int col_b, float beta=0.0F) {
    // https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_a, col_b, col_a, 1.0F, A, col_a, B, col_b, beta, C, col_b);
}

inline void gemm_trans_b(const float* A, const float* B, float* C, int row_a, int col_a, int col_b) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, row_a, col_b, col_a, 1.0F, A, col_a, B, col_a, 0.0F, C, col_b);
}

inline void gemm_trans_a(const float* A, const float* B, float* C, int row_a, int col_a, int col_b) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, row_a, col_b, col_a, 1.0F, A, row_a, B, col_b, 0.0F, C, col_b);
}

inline void exp(const float* A, float* B, int count) {
    // https://software.intel.com/en-us/mkl-developer-reference-c-v-exp
    vsExp(count, A, B);
}