#include "GameExperience.h"
#include <algorithm>
#include <random>
#include <ctime>

// Constructor
GameExperience::GameExperience(int input_size,
    int num_actions,
    int max_memory,
    float discount,
    const std::vector<int>& hidden_layers,
    float lr)
    : model(input_size, hidden_layers, num_actions, lr),
    max_memory(max_memory),
    discount(discount),
    num_actions(num_actions)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

// Store an episode in memory
void GameExperience::remember(const Episode& episode) {
    if (memory.size() >= max_memory) {
        memory[index] = episode;       // Overwrite oldest experience
        index = (index + 1) % max_memory;
    }
    else {
        memory.push_back(episode);
    }
}

// Predict Q-values for a given envstate
std::vector<float> GameExperience::predict(const std::vector<float>& envstate) {
    return model.predict(envstate);
}

// Generate training data from memory
void GameExperience::get_data(std::vector<std::vector<float>>& inputs,
    std::vector<std::vector<float>>& targets,
    int data_size)
{
    if (memory.empty()) return;

    int mem_size = static_cast<int>(memory.size());
    data_size = std::min(mem_size, data_size);

    inputs.clear();
    targets.clear();
    inputs.reserve(data_size);
    targets.reserve(data_size);

    // Randomly shuffle memory indices
    std::vector<int> indices(mem_size);
    for (int i = 0; i < mem_size; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    for (int i = 0; i < data_size; ++i) {
        const Episode& e = memory[indices[i]];

        inputs.push_back(e.envstate);
        std::vector<float> target = model.predict(e.envstate);

        // Compute Q-value for next state
        float Q_sa = 0.0f;
        if (!e.game_over) {
            auto next_q = model.predict(e.envstate_next);
            Q_sa = *std::max_element(next_q.begin(), next_q.end());
        }

        target[e.action] = e.reward + (e.game_over ? 0.0f : discount * Q_sa);

        targets.push_back(target);
    }
}
