#include "mnist_helper.h"

inline __int32 swap_bytes(__int32 *val) {
    unsigned char* ptr = reinterpret_cast<unsigned char*>(val);
    __int32 ret = 0;
    unsigned char& low8bit = *(reinterpret_cast<unsigned char*>(&ret));
    for (int i = 0; i < 4; i++) {
        ret <<= 8;
        low8bit = ptr[i];
    }
    return ret;
}

int mnist_load_images_file(const char* path, std::vector<auto_float>& dst, bool normailize) {
    dst.clear();

    FILE* fin = fopen(path, "rb");
    __int32 tmp = 0;
    fread(&tmp, 4, 1, fin);
    __int32 magic_number = swap_bytes(&tmp);
    if (magic_number != 2051)
        return 0;
    fread(&tmp, 4, 1, fin);
    int count = (int)swap_bytes(&tmp);
    fread(&tmp, 4, 1, fin);
    __int32 row = swap_bytes(&tmp);
    fread(&tmp, 4, 1, fin);
    __int32 col = swap_bytes(&tmp);
    if (row != 28 || col != 28)
        return 0;
    unsigned char* buf = new unsigned char[MNIST_VECTOR_LEN];
    for (int i = 0; i < count; i++) {
        fread(buf, 1, MNIST_VECTOR_LEN, fin);
        float* next = new float[MNIST_VECTOR_LEN];
        for (int p = 0; p < MNIST_VECTOR_LEN; p++)
            next[p] = (float)buf[p];
        if (normailize) {
            for (int p = 0; p < MNIST_VECTOR_LEN; p++)
                next[p] = (next[p] - 127.5f) * (1.0f / 127.5f);
        }
        dst.emplace_back(next, std::default_delete<float[]>());
    }
    return (int)dst.size();
}

int mnist_load_label_file(const char* path, std::vector<int>& label) {
    label.clear();

    FILE* fin = fopen(path, "rb");
    __int32 tmp = 0;
    fread(&tmp, 4, 1, fin);
    __int32 magic_number = swap_bytes(&tmp);
    if (magic_number != 2049)
        return 0;
    fread(&tmp, 4, 1, fin);
    int count = (int)swap_bytes(&tmp);
    unsigned char* buf = new unsigned char[count];
    fread(buf, 1, count, fin);
    for (int i = 0; i < count; i++) {
        label.push_back((int)buf[i]);
    }
    return (int)label.size();
}