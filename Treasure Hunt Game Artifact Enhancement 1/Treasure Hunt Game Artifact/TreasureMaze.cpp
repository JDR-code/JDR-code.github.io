#include "TreasureMaze.h"

// Constructor
TreasureMaze::TreasureMaze(const std::vector<std::vector<float>>& maze_input, std::pair<int, int> pirate)
{
    _maze = maze_input;
    int nrows = _maze.size();
    int ncols = _maze[0].size();
    target = { nrows - 1, ncols - 1 };

    // Find all free cells
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            if (_maze[r][c] == 1.0f) free_cells.push_back({ r,c });
        }
    }
    free_cells.erase(std::remove(free_cells.begin(), free_cells.end(), target), free_cells.end());

    if (_maze[target.first][target.second] == 0.0f)
        throw std::runtime_error("Invalid maze: target cell cannot be blocked!");

    if (std::find(free_cells.begin(), free_cells.end(), pirate) == free_cells.end())
        throw std::runtime_error("Invalid Pirate Location: must sit on a free cell");

    reset(pirate);
}

// Reset method
void TreasureMaze::reset(std::pair<int, int> pirate) {
    this->maze = _maze;
    this->state = { pirate.first, pirate.second, "start" };
    maze[pirate.first][pirate.second] = pirate_mark;
    min_reward = -0.5f * maze.size() * maze[0].size();
    total_reward = 0;
    visited.clear();
}

// Update pirate position
void TreasureMaze::update_state(int action) {
    int pirate_row = std::get<0>(state);
    int pirate_col = std::get<1>(state);
    std::string mode = std::get<2>(state);

    if (maze[pirate_row][pirate_col] > 0.0f)
        visited.insert({ pirate_row, pirate_col });

    std::vector<int> valid = valid_actions();

    std::string new_mode = mode;

    int nrow = pirate_row;
    int ncol = pirate_col;

    if (valid.empty()) {
        new_mode = "blocked";
    }
    else if (std::find(valid.begin(), valid.end(), action) != valid.end()) {
        new_mode = "valid";
        switch (action) {
        case LEFT:  ncol -= 1; break;
        case UP:    nrow -= 1; break;
        case RIGHT: ncol += 1; break;
        case DOWN:  nrow += 1; break;
        }
    }
    else {
        new_mode = "invalid";
    }

    state = { nrow, ncol, new_mode };
}

// Get reward for current state
float TreasureMaze::get_reward() {
    int pirate_row = std::get<0>(state);
    int pirate_col = std::get<1>(state);
    std::string mode = std::get<2>(state);
    int nrows = maze.size();
    int ncols = maze[0].size();

    if (pirate_row == nrows - 1 && pirate_col == ncols - 1) return 1.0f;
    if (mode == "blocked") return min_reward - 1;
    if (visited.find({ pirate_row, pirate_col }) != visited.end()) return -0.25f;
    if (mode == "invalid") return -0.75f;
    if (mode == "valid") return -0.04f;
    return 0.0f;
}

// Act: move pirate and get environment feedback
std::tuple<std::vector<std::vector<float>>, float, std::string> TreasureMaze::act(int action) {
    update_state(action);
    float reward = get_reward();
    total_reward += reward;
    std::string status = game_status();
    std::vector<std::vector<float>> envstate = observe();
    return { envstate, reward, status };
}

// Return current environment
std::vector<std::vector<float>> TreasureMaze::observe() {
    return draw_env();
}

// Draw maze for visualization
std::vector<std::vector<float>> TreasureMaze::draw_env() {
    std::vector<std::vector<float>> canvas = maze;
    int nrows = canvas.size();
    int ncols = canvas[0].size();

    // Clear marks
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            if (canvas[r][c] > 0.0f) canvas[r][c] = 1.0f;
        }
    }

    int row = std::get<0>(state);
    int col = std::get<1>(state);
    canvas[row][col] = pirate_mark;

    return canvas;
}

// Return game status
std::string TreasureMaze::game_status() {
    if (total_reward < min_reward) return "lose";

    int pirate_row = std::get<0>(state);
    int pirate_col = std::get<1>(state);
    int nrows = maze.size();
    int ncols = maze[0].size();

    if (pirate_row == nrows - 1 && pirate_col == ncols - 1) return "win";

    return "not_over";
}

// Return list of valid actions
std::vector<int> TreasureMaze::valid_actions(std::pair<int, int> cell) {
    int row, col;
    if (cell.first == -1) {
        row = std::get<0>(state);
        col = std::get<1>(state);
    }
    else {
        row = cell.first;
        col = cell.second;
    }

    std::vector<int> actions = { LEFT, UP, RIGHT, DOWN };
    int nrows = maze.size();
    int ncols = maze[0].size();

    if (row == 0) actions.erase(std::remove(actions.begin(), actions.end(), UP), actions.end());
    if (row == nrows - 1) actions.erase(std::remove(actions.begin(), actions.end(), DOWN), actions.end());
    if (col == 0) actions.erase(std::remove(actions.begin(), actions.end(), LEFT), actions.end());
    if (col == ncols - 1) actions.erase(std::remove(actions.begin(), actions.end(), RIGHT), actions.end());

    if (row > 0 && maze[row - 1][col] == 0.0f) actions.erase(std::remove(actions.begin(), actions.end(), UP), actions.end());
    if (row < nrows - 1 && maze[row + 1][col] == 0.0f) actions.erase(std::remove(actions.begin(), actions.end(), DOWN), actions.end());
    if (col > 0 && maze[row][col - 1] == 0.0f) actions.erase(std::remove(actions.begin(), actions.end(), LEFT), actions.end());
    if (col < ncols - 1 && maze[row][col + 1] == 0.0f) actions.erase(std::remove(actions.begin(), actions.end(), RIGHT), actions.end());

    return actions;
}
