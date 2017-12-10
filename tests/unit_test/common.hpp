#pragma once
#include <memory>

class TestCommon {
public:
    static std::shared_ptr<float> get_bias_multiplier(int count) {
        float *ret = new float[count];
        for (int i = 0; i < count; i++) {
            ret[i] = 1.0F;
        }
        return std::shared_ptr<float>(ret, std::default_delete<float[]>());
    }

};