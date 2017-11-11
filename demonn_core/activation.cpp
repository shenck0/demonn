#include <algorithm>
#include "activation.h"
#include "blas.hpp"

namespace demonn_core {

    void relu_forward(const float* input, int batch_size, int input_neuron,
        float* output) {
        for (int i = 0; i < batch_size*input_neuron; i++)
            output[i] = std::max(input[i], 0.0F);
    }

    void relu_backward(
        const float* before_relu, int batch_size, int neuron_num,
        const float* grad_input,
        float* grad_output
    ) {
        for (int i = 0; i < batch_size*neuron_num; i++) {
            grad_output[i] = (before_relu[i] >= 0.0F) ? grad_input[i] : 0.0F;
        }
    }

    void softmax_forward(
        const float* input, int batch_size, int class_num,
        float* output
    ) {
        for (int i = 0; i < batch_size; i++) {
            const float* cur_batch = input + class_num * i;
            float* cur_batch_output = output + class_num * i;
            float max_val = cur_batch[0];
            for (int ci = 1; ci < class_num; ci++) {
                max_val = std::max(max_val, cur_batch[ci]);
            }
            for (int ci = 0; ci < class_num; ci++) {
                cur_batch_output[ci] = cur_batch[ci] - max_val;
            }
        }
        exp(output, output, batch_size * class_num);
        for (int i = 0; i < batch_size; i++) {
            float* out = output + class_num * i;
            float scale = 1.0F / cblas_sasum(class_num, out, 1);
            cblas_sscal(class_num, scale, out, 1);
        }
    }

    void softmax_backward(
        const float* softmax_output, int batch_size, int class_num,
        const float* grad_input,
        float* grad_output
    ) {
        for (int bi = 0; bi < batch_size; bi++) {
            const int batch_offset = bi * class_num;
            const float* cur_softmax_output = softmax_output + batch_offset;
            const float* cur_grad_input = grad_input + batch_offset;
            float* cur_grad_output = grad_output + batch_offset;
            for (int i = 0; i < class_num; i++) {
                float sum = 0.0F;
                for (int j = 0; j < class_num; j++) {
                    sum += cur_grad_input[j] * (i == j ? (cur_softmax_output[i] * (1 - cur_softmax_output[j])) : (-cur_softmax_output[i] * cur_softmax_output[j]));
                }
                cur_grad_output[i] = sum;
            }
        }
    }

}