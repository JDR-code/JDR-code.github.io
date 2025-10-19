#pragma once
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include "DQN.h"

// Episode structure
struct Episode {
    std::vector<float> envstate;
    int action;
    float reward;
    std::vector<float> envstate_next;
    bool game_over;
};

class GameExperience {
public:
    GameExperience(int input_size, int num_actions = 4, int max_memory = 100, float discount = 0.95f,
        const std::vector<int>& hidden_layers = { 64,64,64,64,64 }, float lr = 0.001f);

    void remember(const Episode& episode);
    std::vector<float> predict(const std::vector<float>& envstate);
    void get_data(std::vector<std::vector<float>>& inputs,
        std::vector<std::vector<float>>& targets,
        int data_size = 10);

    DQN model; // the DQN neural network
    

private:
    int max_memory;
    float discount;
    int num_actions;
    std::vector<Episode> memory;
    int index = 0;
    std::default_random_engine rng;
};
