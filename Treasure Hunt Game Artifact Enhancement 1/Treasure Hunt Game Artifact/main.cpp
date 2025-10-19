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

// Global exploration factor
float epsilon = 0.5f;

// Format time like Python version
std::string format_time(double seconds) {
    std::ostringstream oss;
    if (seconds < 400) {
        oss << seconds << " seconds";
    }
    else if (seconds < 4000) {
        double m = seconds / 60.0;
        oss << m << " minutes";
    }
    else {
        double h = seconds / 3600.0;
        oss << h << " hours";
    }
    return oss.str();
}

// Completion check stub
bool completion_check(GameExperience& model, TreasureMaze& maze) {
    return false; // Dummy method for first enhancement
}

// Dummy model class for first enhancement
class Model {
public:
    Model(int actions = 4) : num_actions(actions) { std::srand((unsigned)time(nullptr)); }

    std::vector<float> predict(const std::vector<float>& envstate) {
        std::vector<float> q(num_actions);
        for (int i = 0; i < num_actions; ++i)
            q[i] = static_cast<float>(rand()) / RAND_MAX;
        return q;
    }

    void fit(const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets) {
        // Dummy fit
    }

    float evaluate(const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets) {
        return static_cast<float>(rand()) / RAND_MAX;
    }

    int output_size() const { return num_actions; }

private:
    int num_actions;
};

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Maze definition
    std::vector<std::vector<float>> maze = {
        {1., 0., 1., 1., 1., 1., 1., 1.},
        {1., 0., 1., 1., 1., 0., 1., 1.},
        {1., 1., 1., 1., 0., 1., 0., 1.},
        {1., 1., 1., 0., 1., 1., 1., 1.},
        {1., 1., 0., 1., 1., 1., 1., 1.},
        {1., 1., 1., 0., 1., 0., 0., 0.},
        {1., 1., 1., 0., 1., 1., 1., 1.},
        {1., 1., 1., 1., 0., 1., 1., 1.}
    };

    // Initialize environment
    TreasureMaze qmaze(maze);

    // Initialize experience replay object with dummy model
    GameExperience experience(Model(4), 100, 0.95f);

    int n_epoch = 1000; // reduced for demo
    int data_size = 10;

    std::vector<int> win_history;
    int hsize = static_cast<int>((maze.size() * maze[0].size()) / 2);

    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        // Random free cell
        auto& free_cells = qmaze.free_cells;
        int idx = std::rand() % static_cast<int>(free_cells.size());
        std::pair<int, int> agent_cell = free_cells[idx];

        // Reset maze
        qmaze.reset(agent_cell);

        int n_episodes = 0;

        auto envstate = qmaze.observe();

        while (true) {
            auto previous_envstate = envstate;

            int action;
            if ((static_cast<float>(std::rand()) / RAND_MAX) < epsilon) {
                action = std::rand() % 4; // Exploration
            }
            else {
                auto q_values = experience.model.predict(previous_envstate[0]);
                action = std::distance(q_values.begin(),
                    std::max_element(q_values.begin(), q_values.end()));
            }

            auto [next_env, reward, status] = qmaze.act(action);

            // Flatten envstate for storage
            std::vector<float> flat_prev, flat_next;
            for (auto& row : previous_envstate) for (float v : row) flat_prev.push_back(v);
            for (auto& row : next_env) for (float v : row) flat_next.push_back(v);

            experience.remember({ flat_prev, action, reward, flat_next, status == "win" || status == "lose" });

            // Train dummy model
            std::vector<std::vector<float>> inputs, targets;
            experience.get_data(inputs, targets, data_size);
            experience.model.fit(inputs, targets);
            float loss = experience.model.evaluate(inputs, targets);

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
        float win_rate = 0.0f;
        if (!win_history.empty()) {
            int wins = std::accumulate(win_history.end() - std::min(static_cast<int>(win_history.size()), hsize),
                win_history.end(), 0);
            win_rate = static_cast<float>(wins) / hsize;
        }

        printf("Epoch: %03d/%d | Episodes: %d | Wins: %d | Win rate: %.3f | Time: %s\n",
            epoch, n_epoch - 1, n_episodes,
            std::accumulate(win_history.begin(), win_history.end(), 0), win_rate, t.c_str());

        if (win_rate > 0.9f) epsilon = 0.05f;
        if ((int)win_history.size() >= hsize && completion_check(experience, qmaze)) {
            std::cout << "Reached 100% win rate at epoch: " << epoch << std::endl;
            break;
        }
    }

    return 0;
}
