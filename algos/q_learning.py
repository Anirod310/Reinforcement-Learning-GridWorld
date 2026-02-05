import numpy as np

from config import ACTIONS
from utils.state import state_to_id
from utils.policies import epsilon_greedy_action

def train_q_learning(
        env,
        episodes: int,
        max_steps: int,
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        seed : int=0

):
    
    """
    Train tabular q-learning on gridworld environment. 

    Returns : 
    Q: np.ndarray of shape (n_states, n_actions)
    stats : dict with basic training stats

    """
    rng = np.random.default_rng(seed)

    n_actions = len(ACTIONS)
    n_states = env.n_rows * env.n_cols

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = eps_start

    episode_returns = []# Total reward per episode (for average)
    episode_lengths = []# Number of steps per episode

    successes = 0

    for ep in range(1, episodes+1):

        r, c = env.reset()
        s = state_to_id(r, c, env.n_cols)

        total_reward = 0.0
        steps = 0

        for loop in range(max_steps):
            steps += 1

            a_idx = epsilon_greedy_action(Q[s], epsilon, rng)
            a_str = ACTIONS[a_idx]

            (nr, nc), reward, done = env.step(a_str)
            ns = state_to_id(nr, nc, env.n_cols)

            total_reward += float(reward)

            td_target = reward + (0.0 if done else gamma * float(np.max(Q[ns])))
            td_error = td_target - float(Q[s, a_idx])

            Q[s, a_idx] = Q[s, a_idx] + alpha * td_error

            s = ns

            if done : 
                if reward > 0 : 
                    successes +=1
                break
        
        epsilon = max(eps_end, epsilon * eps_decay)
        
        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    stats = {
        "success_rate_train": successes / episodes,
        "avg_return": float(np.mean(episode_returns)),
        "avg_length": float(np.mean(episode_lengths)),
        "final_epsilon": float(epsilon),
    }
    return Q, stats



