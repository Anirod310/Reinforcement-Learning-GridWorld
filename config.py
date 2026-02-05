
#---Tile codes---
START = -1
EMPTY = 0
GOAL = 1
LAVA = 2
WALL = 3

# --- Actions (strings you will use everywhere) ---
ACTIONS = ("U", "D", "L", "R")  # allowed actions list/tuple

# Map action -> (delta_row, delta_col)
# row goes down when it increases; col goes right when it increases
MOVE = {
    "U": (-1, 0),   # up = row - 1
    "D": (1, 0),    # down = row + 1
    "L": (0, -1),   # left = col - 1
    "R": (0, 1),    # right = col + 1
}

#--- Rewards ---
STEP_PENALTY = -0.01
GOAL_REWARD = 1
LAVA_REWARD = -1
BUMP_PENALTY = -0.02

#--- Animation ---

PAUSE_S = 0.25

#--- RL hyperparameters (baseline) ---
ALPHA = 0.1 #learning rate
GAMMA = 0.99 #discount factor

EPSILON_START = 1.0 #exploration rate -> starting epsilon : explores a lot
EPSILON_END = 0.05 #min epsilon
EPSILON_DECAY = 0.9995 #multiply epsilon by this each epsilon to apply epsilon decay and to avoid being too greedy

EPISODES_TRAIN = 100000 #training episodes
EPISODES_EVAL = 100000 #evaluation episodes
MAX_STEPS = 200 #to avoid infinite loop
SEED = 1 #reproducibility seed
