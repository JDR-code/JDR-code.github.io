#include "GameExperience.h"

// Constructor
GameExperience::GameExperience(int num_actions, int max_memory, float discount)
    : model(num_actions), max_memory(max_memory), discount(discount), num_actions(num_actions)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

// Store an episode in memory
void GameExperience::remember(const Episode& episode) {
    memory.push_back(episode);
    if (memory.size() > max_memory) {
        memory.erase(memory.begin()); // remove oldest
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

    int env_size = memory[0].envstate.size();
    int mem_size = memory.size();
    data_size = std::min(mem_size, data_size);

    inputs.resize(data_size, std::vector<float>(env_size, 0.0f));
    targets.resize(data_size, std::vector<float>(num_actions, 0.0f));

    // Randomly shuffle memory indices
    std::vector<int> indices(mem_size);
    for (int i = 0; i < mem_size; ++i) indices[i] = i;

    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    for (int i = 0; i < data_size; ++i) {
        const Episode& e = memory[indices[i]];
        inputs[i] = e.envstate;

        targets[i] = predict(e.envstate);

        float Q_sa = *std::max_element(predict(e.envstate_next).begin(),
            predict(e.envstate_next).end());

        if (e.game_over) {
            targets[i][e.action] = e.reward;
        }
        else {
            targets[i][e.action] = e.reward + discount * Q_sa;
        }
    }
}
