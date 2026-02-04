# part3_visuals/animate.py

import matplotlib.pyplot as plt  # for interactive mode, pause, show

from config import PAUSE_S  # default pause duration
from visuals.render import setup_axes, draw_special_labels, draw_agent, update_agent


def animate_actions(env, actions: list[str], pause_s: float = PAUSE_S):
    """
    Animate a manual list of actions on the environment.

    env: must support env.reset() and env.step(action)
    actions: list like ["R","R","D",...]
    pause_s: seconds between frames
    """
    plt.ion()  # enable interactive mode so updates appear during the loop

    fig, ax = plt.subplots()  # create a figure and an axis

    # Draw the static grid on this axis
    setup_axes(ax, env.grid)

    # Optional: draw labels for S/G/X/# on top of the grid
    draw_special_labels(ax, env.grid)

    # Reset env and draw agent at start
    state = env.reset()
    agent_artist = draw_agent(ax, state)

    # Iterate through the action list with a step counter
    for t, a in enumerate(actions, start=1):
        # Step environment (core rule lives in env.step)
        state, reward, done = env.step(a)

        # Update marker position (move agent visually)
        update_agent(agent_artist, state)

        # Update the title with debug info
        ax.set_title(f"Step {t} | Action: {a} | Reward: {reward:.2f} | State: {state}")

        # Force redraw + short pause to show movement
        fig.canvas.draw()
        plt.pause(pause_s)

        # Stop early if episode terminated
        if done:
            ax.set_title(f"Finished at step {t} | Final state: {state} | Final reward: {reward:.2f}")
            fig.canvas.draw()
            break

    plt.ioff()  # turn off interactive mode
    plt.show()  # show the final figure (keeps window open)
