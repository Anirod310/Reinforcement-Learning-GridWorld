from env.maps import make_grid
from env.gridworld import GridWorld
from visuals.animate import animate_actions
import numpy as np

def main():
    grid = make_grid()

    env = GridWorld(grid)

    actions_win = ['R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'R', 'R', 'D', 'D', 'R', 'R', 'R', 'R']
    actions_loose = ['R', 'R', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'D', 'D', 'D']
    actions_random = np.random.choice(['U', 'D', 'L', 'R'], size=50).tolist()

    animate_actions(env, actions_win)

main()