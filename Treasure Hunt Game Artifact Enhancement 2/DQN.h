#pragma once
#include <vector>
#include <random>
#include <cmath>

class DQN {
private:
    int input_size;
    std::vector<int> hidden_sizes; // 5 hidden layers
    int output_size_;
    float lr;

    std::vector<std::vector<std::vector<float>>> weights; // weights[layer][from][to]
    std::vector<std::vector<float>> biases;               // biases[layer][to]

    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist;

    float relu(float x) { return x > 0 ? x : 0; }
    float relu_derivative(float x) { return x > 0 ? 1 : 0; }

public:
    // Constructor
    DQN(int input, const std::vector<int>& hidden, int output, float learning_rate = 0.001f);

    // Predict Q-values
    std::vector<float> predict(const std::vector<float>& state);

    // Train on batch of inputs and targets
    void fit(const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets,
        int epochs = 1);

    int output_size() const { return output_size_; }
};
