import numpy as np

from config import (
    START, GOAL, LAVA, WALL,
    MOVE,
    STEP_PENALTY, GOAL_REWARD, LAVA_REWARD, BUMP_PENALTY
)

class GridWorld:
    """A deterministic Gridworld environment.
    """
    def __init__(self, grid : np.ndarray):
        self.grid = grid
        self.n_rows, self.n_cols = grid.shape #store dimensions for bounds checking

        #Find the start position
        start_rc = np.argwhere(self.grid == START)[0]
        self.start_pos = (int(start_rc[0]), int(start_rc[1]))

        self.agent_pos = self.start_pos #current position begin at start
    
    def reset(self)->tuple[int, int]:
        """
        Reset agent position at start and return at initial state.
        """
        self.agent_pos = self.start_pos #put agent back at start.
        return self.agent_pos #return state as (row, col)
    
    def step(self, action : str)->tuple[tuple[int, int], float, bool]:
        """
        Apply one action.
        Returns : (next_state, reward, done)

        """

        r, c = self.agent_pos #unpack current postition

        #convert action into movement delta (dr, dc)
        dr, dc = MOVE[action]

        #propose a next position
        nr, nc = r + dr, c + dc

        #check bounds : inside the grid?
        in_bounds = (0 <= nr < self.n_rows) and (0 <= nc < self.n_cols)

        #if not in bounds or next cell is a wall, stay in place and apply bump penalty
        if (not in_bounds) or (self.grid[nr, nc] == WALL):
            reward = BUMP_PENALTY
            done = False
            return (r, c), reward, done
        
        #if moove is valid, update position
        self.agent_pos = (nr, nc)
        tile = self.grid[nr, nc] 

        # Termminal checks 
        if tile == GOAL :
            return self.agent_pos, GOAL_REWARD, True #reached goal
        
        if tile == LAVA :
            return self.agent_pos, LAVA_REWARD, True #stepped in lava
 
        return self.agent_pos, STEP_PENALTY, False #empty tile : penalty to encourage shorter paths