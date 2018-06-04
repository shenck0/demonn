#include <algorithm>
#include "activation.h"
#include "blas.h"
#ifdef compile_implement_mkl
    #include <mkl.h>
#endif

namespace demonn {

    void op(relu, direct, forward, cpu, cpp)(
        int batch_size,
        int n,
        const float* input, // (batch_size, neuron_num)
        float* output // (batch_size, neuron_num)
    ) {
        for (int i = 0; i < batch_size*n; i++)
            output[i] = std::max(input[i], 0.0f);
    }

    void op(relu, direct, backward, cpu, cpp)(
        int batch_size, 
        int n,
        const float* input, // (batch_size, neuron_num)
        const float* grad_output, // (batch_size, neuron_num)
        float* grad_input // (batch_size, neuron_num)
    ) {
        for (int i = 0; i < batch_size*n; i++) {
            grad_input[i] = (input[i] >= 0.0f) ? grad_output[i] : 0.0f;
        }
    }

    void op(softmax, stable, forward, cpu, mkl)(
        int batch_size, 
        int class_num,
        const float* input, // (batch_size, class_num)
        float* output // (batch_size, class_num)
    ) {
        /*
            \text{output}_i=\frac{e^{\text{input}_i}}{\sum_je^{\text{input}_j}}
        */
#ifdef compile_implement_mkl
        for (int i = 0; i < batch_size; i++) {
            const float* cur_input = input + class_num * i;
            float* cur_output = output + class_num * i;
            float max_val = cur_input[0];
            for (int ci = 1; ci < class_num; ci++) {
                max_val = std::max(max_val, cur_input[ci]);
            }
            for (int ci = 0; ci < class_num; ci++) {
                cur_output[ci] = cur_input[ci] - max_val;
            }
        }
        vsExp(batch_size * class_num, output, output);
        for (int i = 0; i < batch_size; i++) {
            float* cur_putput = output + class_num * i;
            float scale = 1.0f / cblas_sasum(class_num, cur_putput, 1);
            cblas_sscal(class_num, scale, cur_putput, 1);
        }
#endif
    }

    void op(softmax, stable, backward, cpu, mkl)(
        int batch_size, 
        int class_num,
        const float* output, // (batch_size, class_num)
        const float* grad_output, // (batch_size, class_num)
        float* grad_input // (batch_size, class_num)
    ) {
        /*
            \text{grad_input}_i = \sum_j \text{grad_output}_j \frac{\partial \text{output}_j}{\partial \text{input}_i} \\
            \frac{\partial \text{output}_j}{\partial \text{input}_i} = \begin{cases} \text{output}_i(1-\text{output}_i) & i=j \\ -\text{output}_i\text{output}_j & i\neq j \end{cases}
        */
#ifdef compile_implement_mkl
        memset(grad_input, 0, sizeof(float) * batch_size * class_num);
        for (int bi = 0; bi < batch_size; bi++) {
            const int batch_offset = bi * class_num;
            const float* cur_output = output + batch_offset;
            const float* cur_grad_output = grad_output + batch_offset;
            float* cur_grad_input = grad_input  + batch_offset;
            for (int i = 0; i < class_num; i++) {
                for (int j = 0; j < class_num; j++) {
                    cur_grad_input[i] += cur_grad_output[j] * (i == j ? (cur_output[i] * (1 - cur_output[j])) : (-cur_output[i] * cur_output[j]));
                }
            }
        }
#endif
    }

}
