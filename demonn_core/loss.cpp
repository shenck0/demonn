#include "loss.h"
#include <cmath>

namespace demonn {

    void op(cross_entropy, direct, forward, cpu, cpp)(
        int batch_size, 
        int n,
        const float* input, // (batch_size, class_num)
        const float* label, // (batch_size, class_num)
        float* output // (batch_size,) 
    ) {
        /*
            \text{output}=-\sum_{i=1}^{n}{\text{label}_i\log{\text{input}_i}}
        */
        for (int bi = 0; bi < batch_size; bi++) {
            float cur_loss = 0.0f;
            const float* cur_label = label + bi * n;
            const float* cur_input = input + bi * n;
            for (int ci = 0; ci < n; ci++) {
                if (cur_label[ci] != 0.0f)
                    cur_loss -= cur_label[ci] * logf(cur_input[ci]);
            }
            output[bi] = cur_loss;
        }
    }

    void op(cross_entropy, direct, backward, cpu, cpp)(
        int batch_size, 
        int n,
        const float* input, // (batch_size, class_num)
        const float* label, // (batch_size, class_num)
        const float* grad_output, // (batch_size,)
        float* grad_input // (batch_size, class_num)
    ) {
        /*
            \text{grad_input}_i=-\text{grad_output}\frac{\text{label}_i}{\text{input}_i}
        */
        for (int i = 0; i < batch_size * n; i++) {
            grad_input[i] = -grad_output[i / n] * label[i] / input[i];
        }
    }

}