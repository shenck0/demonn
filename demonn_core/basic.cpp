#include <cstring>
#include <algorithm>
#include "basic.h"
#include "blas.h"

namespace demonn {

    void op(fully_connected, mm, forward, cpu, mkl)(
        int batch_size, 
        int input_neuron,
        const float* input, // (batch_size, input_neuron)
        int output_neuron,
        const float* weight, // (input_neuron, output_neuron)
        const float* bias, // (output_neuron,)
        const float* bias_multiplier, // (batch_size,)
        float* output // (batch_size, output_neuron)
    ) {
        gemm_mkl(
            bias_multiplier, false,
            bias, false,
            batch_size, output_neuron, 1,
            1.0f, 0.0f,
            output
        );
        gemm_mkl(
            input, false,
            weight, false,
            batch_size, output_neuron, input_neuron,
            1.0f, 0.0f,
            output
        );
    }

    void op(fully_connected, mm, backward, cpu, mkl)(
        int batch_size, 
        int input_neuron,
        int output_neuron,
        const float* input, // (batch_size, input_neuron)
        const float* weight, // (input_neuron, output_neuron)
        const float* bias_multiplier, // at least:(batch_size,)
        const float* grad_output, // (batch_size, output_neuron)
        float* grad_bias, // (output_neuron,)
        float* grad_weight, // (input_neuron, output_neuron)
        float* grad_input // (batch_size, input_neuron)
    ) {
        gemm_mkl(
            grad_output, true,
            bias_multiplier, false,
            output_neuron, 1, batch_size,
            1.0f, 0.0f,
            grad_bias
        );
        gemm_mkl(
            grad_output, false,
            weight, true,
            batch_size, input_neuron, output_neuron,
            1.0f, 0.0f,
            grad_input
        );
        gemm_mkl(
            input, true,
            grad_output, false,
            input_neuron, output_neuron, batch_size,
            1.0f, 0.0f,
            grad_weight
        );
    }

    void op(mean, direct, forward, cpu, cpp)(
        int count,
        const float* input, // (count,)
        float* output // (1,)
    ) {
        *output = 0.0f;
        for (int i = 0; i < count; i++)
            *output += input[i];
        *output /= count;
    }

    void op(mean, direct, backward, cpu, cpp)(
        int count,
        const float* grad_output, // (1,)
        float* grad_input // (count,)
    ) {
        const float scale = 1.0f / count;
        for (int i = 0; i < count; i++) {
            grad_input[i] = grad_output[0] * scale;
        }
    }

    void op(argmax, direct, forward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, n)
        int* output// (batch_size,)
    ) {
        for (int bi = 0; bi < batch_size; bi++) {
            const float* cur_arr = input + bi * n;
            int & ret = output[bi] = 0;
            float val = cur_arr[0];
            for (int i = 1; i < n; i++) {
                if (cur_arr[i] > val) {
                    val = cur_arr[i];
                    ret = i;
                }
            }
        }
    }

}