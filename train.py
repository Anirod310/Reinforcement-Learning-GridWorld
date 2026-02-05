import numpy as np

from config import (
    ALPHA, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    EPISODES_TRAIN, MAX_STEPS, SEED
)
from env.maps import make_grid
from env.gridworld import GridWorld
from algos.q_learning import train_q_learning

def main():
    grid = make_grid()
    env = GridWorld(grid)

    Q, stats = train_q_learning(env=env,
                                episodes=EPISODES_TRAIN,
                                max_steps=MAX_STEPS,
                                alpha=ALPHA,
                                gamma=GAMMA,
                                eps_start=EPSILON_START,
                                eps_end=EPSILON_END,
                                eps_decay=EPSILON_DECAY,
                                seed=SEED)
    
    
    print("Training finished!")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Save Q-table for later visualization/evaluation
    np.save("q_table.npy", Q)
    print("Saved q_table.npy")


main() 