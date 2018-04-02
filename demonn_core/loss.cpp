#include "loss.h"
#include <cmath>

namespace demonn_core {

    void cross_entropy_forward(
        const float* predict, int batch_size, int class_num,
        const float* label,
        float* output_loss
    ) {
        *output_loss = 0.0f;
        for (int bi = 0; bi < batch_size; bi++) {
            const float* cur_label = label + bi * class_num;
            const float* cur_predict = predict + bi * class_num;
            for (int ci = 0; ci < class_num; ci++) {
                if (cur_label[ci] != 0.0f)
                    *output_loss -= cur_label[ci] * logf(cur_predict[ci]);
            }
        }
        *output_loss /= batch_size;
    }

    void cross_entropy_backward(
        const float* predict, int batch_size, int class_num,
        const float* label,
        float* grad_output
    ) {
        float batch_scale = 1.0f / (float)batch_size;
        for (int i = 0; i < batch_size * class_num; i++) {
            grad_output[i] = -batch_scale * label[i] / predict[i];
        }
    }

}