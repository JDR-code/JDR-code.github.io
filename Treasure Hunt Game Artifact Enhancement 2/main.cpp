#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include "TreasureMaze.h"
#include "GameExperience.h"
#include "DQN.h"

// Global exploration factor
float epsilon = 0.5f;

// Format time helper
std::string format_time(double seconds) {
    std::ostringstream oss;
    if (seconds < 400) oss << seconds << " seconds";
    else if (seconds < 4000) oss << seconds / 60.0 << " minutes";
    else oss << seconds / 3600.0 << " hours";
    return oss.str();
}

// Completion check stub
bool completion_check(GameExperience& model, TreasureMaze& maze) {
    return false; // placeholder for future logic
}

// Flatten maze helper
std::vector<float> flatten_maze(const std::vector<std::vector<float>>& maze) {
    std::vector<float> flat;
    for (auto& row : maze)
        for (float v : row) flat.push_back(v);
    return flat;
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Maze definition
    std::vector<std::vector<float>> maze = {
        {1.,0.,1.,1.,1.,1.,1.,1.},
        {1.,0.,1.,1.,1.,0.,1.,1.},
        {1.,1.,1.,1.,0.,1.,0.,1.},
        {1.,1.,1.,0.,1.,1.,1.,1.},
        {1.,1.,0.,1.,1.,1.,1.,1.},
        {1.,1.,1.,0.,1.,0.,0.,0.},
        {1.,1.,1.,0.,1.,1.,1.,1.},
        {1.,1.,1.,1.,0.,1.,1.,1.}
    };

    TreasureMaze qmaze(maze);

    int input_size = static_cast<int>(maze.size() * maze[0].size());
    int num_actions = 4;
    int max_memory = 1000;       // more experience
    float discount = 0.95f;
    float lr = 0.005f;          // faster learning

    // Dynamic hidden layers
    std::vector<int> hidden_layers = { 64, 32, 16, 8, 4 }; // More values in the vector the more hidden layers

    // Initialize GameExperience with 5-hidden-layer DQN
    GameExperience experience(input_size, num_actions, max_memory, discount, hidden_layers, lr);

    int n_epoch = 15000;
    int data_size = 50;          // larger batch for training

    std::vector<int> win_history;
    int hsize = static_cast<int>((maze.size() * maze[0].size()) / 2);

    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        // Pick random starting cell
        auto& free_cells = qmaze.free_cells;
        int idx = std::rand() % static_cast<int>(free_cells.size());
        std::pair<int, int> agent_cell = free_cells[idx];

        qmaze.reset(agent_cell);

        int n_episodes = 0;
        auto envstate = qmaze.observe();

        while (true) {
            auto previous_envstate = envstate;
            int action;

            // Epsilon-greedy exploration
            if ((static_cast<float>(std::rand()) / RAND_MAX) < epsilon) {
                action = std::rand() % num_actions;
            }
            else {
                auto q_values = experience.model.predict(flatten_maze(previous_envstate));
                action = std::distance(q_values.begin(),
                    std::max_element(q_values.begin(), q_values.end()));
            }

            auto [next_env, reward, status] = qmaze.act(action);

            // Flatten for storage
            std::vector<float> flat_prev = flatten_maze(previous_envstate);
            std::vector<float> flat_next = flatten_maze(next_env);

            experience.remember({ flat_prev, action, reward, flat_next, status == "win" || status == "lose" });

            // Train on batch
            std::vector<std::vector<float>> inputs, targets;
            experience.get_data(inputs, targets, data_size);
            if (!inputs.empty()) experience.model.fit(inputs, targets);

            n_episodes++;
            envstate = next_env;

            if (status == "win" || status == "lose") {
                win_history.push_back(status == "win" ? 1 : 0);
                break;
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        std::string t = format_time(elapsed);

        // Calculate rolling win rate
        float win_rate = 0.0f;
        if (!win_history.empty()) {
            int wins = std::accumulate(win_history.end() - std::min((int)win_history.size(), hsize),
                win_history.end(), 0);
            win_rate = static_cast<float>(wins) / hsize;
        }

        printf("Epoch: %03d/%d | Episodes: %d | Wins: %d | Win rate: %.3f | Time: %s\n",
            epoch, n_epoch - 1, n_episodes,
            std::accumulate(win_history.begin(), win_history.end(), 0),
            win_rate, t.c_str());

        // Epsilon decay
        epsilon = std::max(0.05f, epsilon * 0.995f);
        if ((int)win_history.size() >= hsize && completion_check(experience, qmaze)) {
            std::cout << "Reached 100% win rate at epoch: " << epoch << std::endl;
            break;
        }
    }

    return 0;
}
