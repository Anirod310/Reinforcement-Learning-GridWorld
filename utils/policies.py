import numpy as np

def epsilon_greed_action(q_row : np.ndarray, epsilon : float, rng : np.random.Generator) -> int:
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