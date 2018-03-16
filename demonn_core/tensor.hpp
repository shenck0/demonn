#pragma once

#include <vector>
#include <memory>
#include <initializer_list>

#include "common.h"

namespace demonn {

    class tensor {
    public:
        void* ptr;
        void* ptr_grad;
        int size;
        int size_in_bytes;
        std::vector<int> shape;

        explicit tensor(std::initializer_list<int> dims, int ele_size) : shape(dims), ptr_grad(NULL) {
            size = 1;
            for (const int& dim : dims)
                size *= dim;
            size_in_bytes = ele_size * size;
            ptr = calloc(size_in_bytes, 1);
        }
        ~tensor() { 
            if (ptr) free(ptr); 
            if (ptr_grad) free(ptr_grad);
        }

        inline float* data() { 
            return reinterpret_cast<float*>(ptr); 
        }
        inline float* grad() { 
            if (!ptr_grad)
                ptr_grad = calloc(size_in_bytes, 1);
            return reinterpret_cast<float*>(ptr_grad); 
        }

        template<typename type>
        inline static std::shared_ptr<tensor> get(std::initializer_list<int> dims) {
            return std::make_shared<tensor>(dims, (int)sizeof(type));
        }

    private:
        tensor(const tensor&) { }
        tensor& operator=(const tensor&) { }
    };

}
