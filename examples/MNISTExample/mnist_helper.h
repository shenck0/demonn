#pragma once
#include <memory>
#include <vector>
#include <stdio.h>

#define MNIST_WIDTH 28
#define MNIST_VECTOR_LEN 784
#define MNIST_CLASS_NUM 10

typedef std::shared_ptr<float> auto_float;
int mnist_load_images_file(const char* path, std::vector<auto_float>& dst, bool normailize);
int mnist_load_label_file(const char* path, std::vector<int>& label);