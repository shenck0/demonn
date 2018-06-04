#include "demonn.h"
#include <ctime>
#include <random>

namespace demonn {

    class _random_res {
    public:
        std::random_device* rd;
        std::mt19937* gen;
        _random_res() {
            rd = new std::random_device{};
            gen = new std::mt19937{ (*rd)() };
        }
        ~_random_res() {
            delete gen;
            delete rd;
        }
    };

    void fill_onehot(
        int batch_size,
        int n,
        const int* label, // (batch_size,)
        float* output // (batch_size, n)
    ) {
        memset(output, 0, sizeof(float) * batch_size * n);
        for (int i = 0; i < batch_size; i++)
            output[n * i + label[i]] = 1.0f;
    }

    void fill_constant(
        int count,
        float value,
        float* output // (count,)
    ) {
        if (value == 0.0f) {
            memset(output, 0, sizeof(float) * count);
        } else {
            for (int i = 0; i < count; i++)
                output[i] = value;
        }
    }

    void fill_normal_distribution(
        float mean, float std_dev,
        int n,
        float* output // (n,)
    ) {
        thread_local static _random_res res;
        std::normal_distribution<float> d{ mean, std_dev };
        for (int i = 0; i < n; i++) {
            output[i] = d(*res.gen);
        }
    }

}