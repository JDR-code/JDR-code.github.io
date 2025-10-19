#include "DQN.h"
#include <algorithm>

// Constructor
DQN::DQN(int input, const std::vector<int>& hidden, int output, float learning_rate)
    : input_size(input), hidden_sizes(hidden), output_size_(output), lr(learning_rate), dist(-0.5f, 0.5f)
{
    rng.seed(std::random_device{}());

    int prev_size = input_size;

    for (int hsize : hidden_sizes) {
        std::vector<std::vector<float>> W(prev_size, std::vector<float>(hsize));
        for (int i = 0; i < prev_size; ++i)
            for (int j = 0; j < hsize; ++j)
                W[i][j] = dist(rng);
        weights.push_back(W);
        biases.push_back(std::vector<float>(hsize, 0.0f));
        prev_size = hsize;
    }

    // Output layer
    std::vector<std::vector<float>> W_out(prev_size, std::vector<float>(output_size_));
    for (int i = 0; i < prev_size; ++i)
        for (int j = 0; j < output_size_; ++j)
            W_out[i][j] = dist(rng);
    weights.push_back(W_out);
    biases.push_back(std::vector<float>(output_size_, 0.0f));
}

// Forward pass
std::vector<float> DQN::predict(const std::vector<float>& state) {
    std::vector<float> activations = state;

    for (size_t l = 0; l < weights.size(); ++l) {
        std::vector<float> next(weights[l][0].size(), 0.0f);
        for (size_t j = 0; j < next.size(); ++j) {
            for (size_t i = 0; i < activations.size(); ++i)
                next[j] += activations[i] * weights[l][i][j];
            next[j] += biases[l][j];
            if (l < weights.size() - 1) next[j] = relu(next[j]); // hidden layers
        }
        activations = next;
    }

    return activations; // final Q-values
}

// Training
void DQN::fit(const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    int epochs)
{
    for (int e = 0; e < epochs; ++e) {
        for (size_t k = 0; k < inputs.size(); ++k) {
            std::vector<std::vector<float>> activations;
            activations.push_back(inputs[k]);

            // Forward pass
            for (size_t l = 0; l < weights.size(); ++l) {
                std::vector<float> next(weights[l][0].size(), 0.0f);
                for (size_t j = 0; j < next.size(); ++j) {
                    for (size_t i = 0; i < activations[l].size(); ++i)
                        next[j] += activations[l][i] * weights[l][i][j];
                    next[j] += biases[l][j];
                    if (l < weights.size() - 1) next[j] = relu(next[j]);
                }
                activations.push_back(next);
            }

            // Output error
            std::vector<float> error(weights.back()[0].size(), 0.0f);
            for (size_t i = 0; i < error.size(); ++i)
                error[i] = targets[k][i] - activations.back()[i];

            std::vector<float> delta = error;

            // Backpropagation
            for (int l = (int)weights.size() - 1; l >= 0; --l) {
                std::vector<float> prev_activations = activations[l];
                std::vector<float> delta_next(weights[l].size(), 0.0f);

                for (size_t i = 0; i < weights[l].size(); ++i) {
                    for (size_t j = 0; j < weights[l][i].size(); ++j) {
                        weights[l][i][j] += lr * delta[j] * prev_activations[i];
                        delta_next[i] += delta[j] * weights[l][i][j];
                    }
                }

                for (size_t j = 0; j < biases[l].size(); ++j)
                    biases[l][j] += lr * delta[j];

                if (l > 0) {
                    for (size_t i = 0; i < delta_next.size(); ++i)
                        delta_next[i] *= relu_derivative(prev_activations[i]);
                    delta = delta_next;
                }
            }
        }
    }
}
