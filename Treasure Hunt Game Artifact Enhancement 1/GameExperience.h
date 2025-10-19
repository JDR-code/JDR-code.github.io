#pragma once

#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>

// --- Dummy Model class ---
class Model {
public:
    Model(int actions = 4) : num_actions(actions) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }

    std::vector<float> predict(const std::vector<float>& envstate) {
        std::vector<float> q_values(num_actions);
        for (int i = 0; i < num_actions; ++i) {
            q_values[i] = static_cast<float>(std::rand()) / RAND_MAX; // random float [0,1)
        }
        return q_values;
    }

    int output_size() const { return num_actions; }

private:
    int num_actions;
};

// --- Episode structure ---
struct Episode {
    std::vector<float> envstate;
    int action;
    float reward;
    std::vector<float> envstate_next;
    bool game_over;
};

// --- GameExperience class ---
class GameExperience {
public:
    GameExperience(int num_actions = 4, int max_memory = 100, float discount = 0.95f);

    void remember(const Episode& episode);
    std::vector<float> predict(const std::vector<float>& envstate);
    void get_data(std::vector<std::vector<float>>& inputs,
        std::vector<std::vector<float>>& targets,
        int data_size = 10);

private:
    Model model;
    int max_memory;
    float discount;
    int num_actions;
    std::vector<Episode> memory;
};
