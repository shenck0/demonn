#include <iostream>
#include <tuple>
#include <random>
#include <algorithm>
#include <demonn_core.h>

#include "mnist_helper.h"

using namespace std;
namespace d = demonn;
namespace dc = demonn_core;

int main_fc_network(int argc, char* argv[])
{
    // Construct fully connected network : 784 - fc1 - 500 - relu1 - 500 - fc2 - 10 - softmax - 10
    // fc1
    auto w1 = d::tensor::get<float>({ 784, 500 });
    auto b1 = d::tensor::get<float>({ 500 });
    // fc2
    auto w2 = d::tensor::get<float>({ 500, MNIST_CLASS_NUM });
    auto b2 = d::tensor::get<float>({ MNIST_CLASS_NUM });
    // initialize weights
    dc::normal_distribution(w1->data(), w1->size, 0.0F, 0.01F);
    dc::normal_distribution(w2->data(), w2->size, 0.0F, 0.001F);
    dc::set_array(b1->data(), 0.0F, b1->size);
    dc::set_array(b2->data(), 0.0F, b2->size);
    
    // Train
    {
        std::default_random_engine e1;
        auto e2 = e1;
        int train_epoch = 10;
        int train_batch_size = 64;
        float learning_rate = 0.01F;
        auto in = d::tensor::get<float>({ train_batch_size, 784 });
        auto fc1 = d::tensor::get<float>({ train_batch_size, 500 });
        auto relu1 = d::tensor::get<float>({ train_batch_size, 500 });
        auto fc2 = d::tensor::get<float>({ train_batch_size, MNIST_CLASS_NUM });
        auto s1 = d::tensor::get<float>({ train_batch_size, MNIST_CLASS_NUM });
        auto label = d::tensor::get<float>({ train_batch_size, MNIST_CLASS_NUM });
        float loss = 0.0F;
        auto bias_multiplier = d::tensor::get<float>({ train_batch_size });
        dc::set_array(bias_multiplier->data(), 1.0F, train_batch_size);

        vector<shared_ptr<float> > train_images;
        vector<int> train_labels;
        int train_count = mnist_load_images_file("D:/Dataset/MNIST/train-images.idx3-ubyte", train_images, true);
        int train_label_count = mnist_load_label_file("D:/Dataset/MNIST/train-labels.idx1-ubyte", train_labels);
        check(train_count > 0 && train_count == train_label_count);
        for (int epoch = 0; epoch < train_epoch; epoch++) {
            std::shuffle(train_images.begin(), train_images.end(), e1); std::shuffle(train_labels.begin(), train_labels.end(), e2); // 按同样方式随机打乱样本&标签
            for (int batch = 0; batch < train_count / train_batch_size; batch++) {
                // Fill current batch
                float *in_ptr = in->data();
                for (int bi = 0; bi < train_batch_size; bi++)
                    memcpy(in_ptr + bi * MNIST_VECTOR_LEN, train_images[batch * train_batch_size + bi].get(), sizeof(float) * MNIST_VECTOR_LEN);
                dc::onehot(&train_labels[batch * train_batch_size], train_batch_size, MNIST_CLASS_NUM, label->data());
                // Forward propagation
                dc::fully_connected_forward(in_ptr, train_batch_size, MNIST_VECTOR_LEN, w1->data(), b1->data(), bias_multiplier->data(), fc1->data(), 500); // fc1
                dc::relu_forward(fc1->data(), train_batch_size, 500, relu1->data()); // relu1
                dc::fully_connected_forward(relu1->data(), train_batch_size, 500, w2->data(), b2->data(), bias_multiplier->data(), fc2->data(), MNIST_CLASS_NUM); // fc2
                dc::softmax_forward(fc2->data(), train_batch_size, MNIST_CLASS_NUM, s1->data()); // softmax
                dc::cross_entropy_forward(s1->data(), train_batch_size, MNIST_CLASS_NUM, label->data(), &loss);
                cout << "epoch=" << epoch << " batch=" << batch << " loss=" << loss << endl;
                // Backward propagation
                dc::cross_entropy_backward(s1->data(), train_batch_size, MNIST_CLASS_NUM, label->data(), s1->grad());
                dc::softmax_backward(s1->data(), train_batch_size, MNIST_CLASS_NUM, s1->grad(), fc2->grad());
                dc::fully_connected_backward(relu1->data(), train_batch_size, 500, MNIST_CLASS_NUM, w2->data(), bias_multiplier->data(), b2->grad(), w2->grad(), fc2->grad(), relu1->grad());
                dc::relu_backward(fc1->data(), train_batch_size, 500, relu1->grad(), fc1->grad());
                dc::fully_connected_backward(in_ptr, train_batch_size, MNIST_VECTOR_LEN, 500, w1->data(), bias_multiplier->data(), b1->grad(), w1->grad(), fc1->grad(), in->grad());
                // Update weights
                dc::stochastic_gradient_descent(w1->data(), w1->size, w1->grad(), learning_rate);
                dc::stochastic_gradient_descent(b1->data(), b1->size, b1->grad(), learning_rate);
                dc::stochastic_gradient_descent(w2->data(), w2->size, w2->grad(), learning_rate);
                dc::stochastic_gradient_descent(b2->data(), b2->size, b2->grad(), learning_rate);
            }
        }
    }
    
    // Test
    {
        auto fc1 = d::tensor::get<float>({ 500 });
        auto fc2 = d::tensor::get<float>({ MNIST_CLASS_NUM });
        auto s1 = d::tensor::get<float>({ MNIST_CLASS_NUM });
        vector<shared_ptr<float> > test_images;
        vector<int> test_labels;
        vector<pair<int, int> > pred_results;
        int test_count = mnist_load_images_file("D:/Dataset/MNIST/t10k-images.idx3-ubyte", test_images, true);
        int label_count = mnist_load_label_file("D:/Dataset/MNIST/t10k-labels.idx1-ubyte", test_labels);
        check(test_count == label_count);
        for (int i = 0; i < test_count; i++) {
            dc::fully_connected_forward(test_images[i].get(), 1, MNIST_VECTOR_LEN, w1->data(), b1->data(), NULL, fc1->data(), 500); // fc1
            dc::relu_forward(fc1->data(), 1, 500, fc1->data()); // relu1
            dc::fully_connected_forward(fc1->data(), 1, 500, w2->data(), b2->data(), NULL, fc2->data(), MNIST_CLASS_NUM); // fc2
            dc::softmax_forward(fc2->data(), 1, MNIST_CLASS_NUM, s1->data()); // softmax
            int cur_pred = dc::argmax(s1->data(), MNIST_CLASS_NUM);
            pred_results.push_back(make_pair(cur_pred, test_labels[i]));
        }
        int correct_count = 0;
        for (pair<int, int> &pred_pair : pred_results) {
            if (pred_pair.first == pred_pair.second)
                correct_count++;
        }
        cout << "accuracy = " << (float)correct_count / (float)pred_results.size() << endl;
    }
    return 0;
}

