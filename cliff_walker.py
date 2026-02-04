import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

class CliffWalker(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        # Grid size: 4 rows, 12 columns (classic Cliff Walker)
        self.rows = 4
        self.cols = 12
        self.grid_size = (self.rows, self.cols)
        
        # Start (bottom-left corner) and Goal (bottom-right corner)
        self.start_pos = np.array([3, 0])
        self.goal_pos = np.array([3, 11])
        
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)
        
        # State: a single integer from 0 to 47 (for Q-Learning)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.agent_pos = self.start_pos.copy()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 60

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_state(), {}

    def step(self, action):
        # 1. Movement logic (coordinate shifts)
        # 0: Up, 1: Right, 2: Down, 3: Left
        direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1])
        }
        
        # Calculate new position
        current_pos = self.agent_pos.copy()
        new_pos = current_pos + direction[action]

        # 2. Boundary check (walls)
        # np.clip prevents moving outside the grid (0..rows-1, 0..cols-1)
        new_pos[0] = np.clip(new_pos[0], 0, self.rows - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.cols - 1)
        
        self.agent_pos = new_pos

        # 3. Check for Cliff and Goal
        terminated = False
        reward = -1 # Standard penalty per step
        
        # Cliff condition: Bottom row (row=3), between Start (col=0) and Goal (col=11)
        is_cliff = (self.agent_pos[0] == 3) and (0 < self.agent_pos[1] < 11)
        
        if is_cliff:
            reward = -100
            self.agent_pos = self.start_pos.copy() # Falling returns to start!
            
        elif np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self._get_state(), reward, terminated, False, {}

    def _get_state(self):
        # Convert (row, col) into a single integer
        return self.agent_pos[0] * self.cols + self.agent_pos[1]

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.cols * self.cell_size, self.rows * self.cell_size))
            pygame.display.set_caption("Cliff Walker RL")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.cols * self.cell_size, self.rows * self.cell_size))
        canvas.fill((255, 255, 255)) # White background

        for r in range(self.rows):
            for c in range(self.cols):
                rect = (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                # Draw the cliff (gray zone at the bottom)
                if r == 3 and 0 < c < 11:
                    pygame.draw.rect(canvas, (100, 100, 100), rect) # Gray color - The Cliff
                else:
                    pygame.draw.rect(canvas, (200, 200, 200), rect, 1) # Grid lines

        # Draw Goal (Green)
        goal_rect = (self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(canvas, (0, 255, 0), goal_rect)

        # Draw Agent (Blue circle)
        agent_center = (
            int(self.agent_pos[1] * self.cell_size + self.cell_size / 2),
            int(self.agent_pos[0] * self.cell_size + self.cell_size / 2),
        )
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, int(self.cell_size / 3))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Run this file for testing
if __name__ == '__main__':
    env = CliffWalker(render_mode="human")
    env.reset()
    
    # Take 20 random steps
    for _ in range(20):
        action = env.action_space.sample()
        env.step(action)
    env.close()