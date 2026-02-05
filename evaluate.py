import numpy as np
from env.maps import make_grid
from env.gridworld import GridWorld
from visuals.animate import animate_actions
from utils.policies import rollout_greedy_policy
from config import EPISODES_EVAL, GOAL
import time


def main():
    grid = make_grid()
    env = GridWorld(grid)

    Q = np.load("q_table.npy")

    test_actions_win = ['R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'R', 'R', 'D', 'D', 'R', 'R', 'R', 'R']
    test_actions_loose = ['R', 'R', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'D', 'D', 'D']
    test_actions_random = np.random.choice(['U', 'D', 'L', 'R'], size=50).tolist()

    eval_successes = 0
    total_returns = []
    total_steps = []
    terminated = 0

    for ep in range(EPISODES_EVAL):

        learned_actions, ep_return, done, steps, final_state = rollout_greedy_policy(env, Q)

        total_returns.append(ep_return)
        total_steps.append(steps)
        terminated += int(done)

        # Success = ended on the GOAL tile (most reliable)
        fr, fc = final_state
        if done and env.grid[fr, fc] == GOAL:
            eval_successes += 1

    success_rate = 100.0 * eval_successes / EPISODES_EVAL
    avg_return = float(np.mean(total_returns))
    avg_steps = float(np.mean(total_steps))
    term_rate = 100.0 * terminated / EPISODES_EVAL

    print(f"Eval success rate: {success_rate:.2f}% ({eval_successes}/{EPISODES_EVAL})")
    print(f"Eval termination rate: {term_rate:.2f}%")
    print(f"Avg return: {avg_return:.3f}")
    print(f"Avg steps: {avg_steps:.2f}")

    # Small pause before showing an animation
    time.sleep(2)

    # Animate ONE fresh greedy rollout (so it's not "whatever happened last in the loop")
    actions, ep_return, done, steps, final_state = rollout_greedy_policy(env, Q)
    print(f"Animating one rollout | done={done} | return={ep_return:.3f} | steps={steps} | final_state={final_state}")

    animate_actions(env, actions)


main()