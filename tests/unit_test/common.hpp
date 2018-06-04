#pragma once
#include <memory>

namespace d = demonn;

class TestCommon {
public:
    static std::shared_ptr<float> get_bias_multiplier(int count) {
        float *ret = new float[count];
        for (int i = 0; i < count; i++) {
            ret[i] = 1.0f;
        }
        return std::shared_ptr<float>(ret, std::default_delete<float[]>());
    }

    static void check_float_array(const float* A, const float* B, const int count) {
        for (int i = 0; i < count; i++)
            EXPECT_FLOAT_EQ(A[i], B[i]);
    }

};