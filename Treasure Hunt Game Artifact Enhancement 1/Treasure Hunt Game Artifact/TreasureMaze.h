#pragma once

#include <vector>
#include <tuple>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// Constants
const float visited_mark = 0.8f;
const float pirate_mark = 0.5f;

const int LEFT = 0;
const int UP = 1;
const int RIGHT = 2;
const int DOWN = 3;

class TreasureMaze {
public:
    TreasureMaze(const std::vector<std::vector<float>>& maze, std::pair<int, int> pirate = { 0,0 });

    void reset(std::pair<int, int> pirate);
    void update_state(int action);
    float get_reward();
    std::tuple<std::vector<std::vector<float>>, float, std::string> act(int action);
    std::vector<std::vector<float>> observe();
    std::vector<std::vector<float>> draw_env();
    std::string game_status();
    std::vector<int> valid_actions(std::pair<int, int> cell = { -1,-1 });

private:
    std::vector<std::vector<float>> _maze;
    std::vector<std::vector<float>> maze;
    std::pair<int, int> target;
    std::vector<std::pair<int, int>> free_cells;
    std::tuple<int, int, std::string> state; // (row, col, mode)
    float min_reward;
    float total_reward;
    std::set<std::pair<int, int>> visited;
};
