#include <cstring>
#include <algorithm>
#include "basic.h"
#include "blas.hpp"

namespace demonn_core {

    void fully_connected_forward(
        const float* input, int batch_size, int input_neuron,
        const float* weight, const float* bias,
        const float* bias_multiplier,
        float* output, int output_neuron
    ) {
        check(bias_multiplier != NULL || batch_size == 1);
        if (batch_size == 1)
            memcpy(output, bias, sizeof(float) * output_neuron);
        else
            gemm(bias_multiplier, bias, output, batch_size, 1, output_neuron);
        gemm(input, weight, output, batch_size, input_neuron, output_neuron, 1.0f);
    }

    void fully_connected_backward(
        const float* fc_input, int batch_size, int input_neuron, int output_neuron,
        const float* weight,
        const float* bias_multiplier,
        float* bias_grad,
        float* grad_weight,
        const float* grad_input,
        float* grad_output
    ) {
        gemm_trans_a(grad_input, bias_multiplier, bias_grad, output_neuron, batch_size, 1); // bias_grad = (batch_size x output_neuron)' * (batch_size x 1)
        gemm_trans_b(grad_input, weight, grad_output, batch_size, output_neuron, input_neuron); // grad_output = (batch_size x output_neuron) * (input_neuron x output_neuron)'
        gemm_trans_a(fc_input, grad_input, grad_weight, input_neuron, batch_size, output_neuron); // grad_weight = (batch_size x input_num)' * (batch_size x output_num)
    }

}

namespace demonn_core {

    int argmax(float* arr, int n) {
        int ret = 0;
        float val = arr[0];
        for (int i = 1; i < n; i++) {
            if (arr[i] > val) {
                val = arr[i];
                ret = i;
            }
        }
        return ret;
    }

    void onehot(const int* labels, int batch_size, int class_num,
        float* output) {
        memset(output, 0, sizeof(float) * batch_size * class_num);
        for (int i = 0; i < batch_size; i++)
            output[class_num * i + labels[i]] = 1.0f;
    }

    float mean(const float* arr, int count) {
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
            sum += arr[i];
        return sum / count;
    }

    void set_array(float* arr, float value, int count) {
        if (value == 0.0f) {
            memset(arr, 0, sizeof(float) * count);
        } else {
            for (int i = 0; i < count; i++)
                arr[i] = value;
        }
    }

}