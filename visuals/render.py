import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import START, EMPTY, GOAL, LAVA, WALL

def make_cmap() -> ListedColormap:
    """
    Create a colormap for the tile values.
    Colors are ordered to look nice; you can change them.
    """
    colors = ['orange', 'green', 'cyan', 'brown', 'black']  # list of color names
    return ListedColormap(colors)  # creates a colormap object from those colors


def setup_axes(ax: plt.Axes, grid: np.ndarray):
    """
    Draw the grid + borders on the given axis.
    Returns the AxesImage from imshow (useful later if needed).
    """
    cmap = make_cmap()  # create colormap for grid display

    # Draw grid cells as an image; interpolation='nearest' keeps crisp squares
    img = ax.imshow(grid, cmap=cmap, interpolation='nearest')

    # Major ticks show cell indices
    ax.set_xticks(range(grid.shape[1]))  # x ticks = columns 0..n_cols-1
    ax.set_yticks(range(grid.shape[0]))  # y ticks = rows 0..n_rows-1

    # Minor ticks at half-steps so grid lines land between cells
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)

    # Draw the minor grid lines (cell borders)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    return img  # return image artist


def draw_special_labels(ax: plt.Axes, grid: np.ndarray):
    """
    Optional: write text labels on special tiles.
    """
    n_rows, n_cols = grid.shape  # read grid size

    for r in range(n_rows):  # loop over rows
        for c in range(n_cols):  # loop over cols
            tile = grid[r, c]  # tile value at (r,c)

            # Decide label text based on tile code
            if tile == START:
                text = "S"
            elif tile == GOAL:
                text = "G"
            elif tile == LAVA:
                text = "X"
            elif tile == WALL:
                text = "#"
            else:
                continue  # skip empty cells to avoid clutter

            # Put text at cell center (x=col, y=row)
            ax.text(c, r, text, ha='center', va='center', fontsize=10)


def draw_agent(ax: plt.Axes, pos: tuple[int, int]):
    """
    Draw agent marker at (row, col) and return the scatter artist.
    """
    r, c = pos  # unpack position
    # ax.scatter expects x then y => x=col, y=row
    agent = ax.scatter(c, r, s=200, marker='o', edgecolors='white', colorizer='grey')
    return agent


def update_agent(agent_artist, pos: tuple[int, int]):
    """
    Move the existing agent marker to the new position.
    """
    r, c = pos  # unpack
    # set_offsets expects [x, y] => [col, row]
    agent_artist.set_offsets([c, r])