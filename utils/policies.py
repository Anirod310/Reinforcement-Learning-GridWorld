import numpy as np
from utils.state import state_to_id
from config import ACTIONS

def epsilon_greedy_action(q_row : np.ndarray, epsilon : float, rng : np.random.Generator) -> int:
    """
    Pick an action index using epsilon-greedy.

    q_row: shape (n_actions,) = Q-values for the current state
    epsilon: probability of random action
    rng: numpy random generator for reproducibility
    """
    if rng.random() < epsilon:
    # explore: random action index
        return int(rng.integers(0, q_row.shape[0]))

    # exploit: best action index (tie-breaking handled below)
    max_q = np.max(q_row)
    best_actions = np.flatnonzero(q_row == max_q)  # all argmax indices
    return int(rng.choice(best_actions))  # random tie-break

def rollout_greedy_policy(env, Q, max_steps=200):
    """
    Run the greedy policy learned from Q and returns the list of actions it learns.

    """

    actions = []
    total_return = 0.0

    r, c = env.reset()
    s = state_to_id(r, c, env.n_cols)
    
    done = False                 # track if we terminated
    steps = 0                    # track how many actions we actually executed
    final_state = (r, c)         # will be updated as we step

    for loop in range(max_steps):
        steps += 1

        #Always chooses the best action (no exploration)
        a_idx = np.argmax(Q[s])
        a_str = ACTIONS[a_idx]
        actions.append(a_str)

        (nr, nc), reward, done = env.step(a_str)
        total_return += float(reward)

        final_state = (nr, nc)

        s = state_to_id(nr, nc, env.n_cols)

        if done:
            break

    return actions, total_return, done, steps, final_state

