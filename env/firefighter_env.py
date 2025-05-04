# firefighter_env.py
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FireFighterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(FireFighterEnv, self).__init__()

        self.grid_size = 4
        self.max_steps = 60

        self.agent_start = (0, 0)
        self.bucket_pos = (1, 1)
        self.fire_pos = (1, 3)
        self.goal_pos = (3, 3)
        self.walls = {(1, 2), (2, 1)}

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation: (x, y, has_bucket, fire_out)
        self.observation_space = spaces.MultiDiscrete([4, 4, 2, 2])

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = list(self.agent_start)
        self.has_bucket = False
        self.fire_out = False
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos

        move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dx, dy = move[action]

        # Stochastic transitions
        if np.random.rand() > 0.8:
            dx, dy = move[np.random.choice([a for a in move if a != action])]

        new_x = np.clip(x + dx, 0, self.grid_size - 1)
        new_y = np.clip(y + dy, 0, self.grid_size - 1)

        if (new_x, new_y) in self.walls:
            reward = -5
        else:
            self.agent_pos = [new_x, new_y]
            reward = 0

        # Bucket collection
        if tuple(self.agent_pos) == self.bucket_pos and not self.has_bucket:
            self.has_bucket = True
            reward += 10

        # Extinguish fire
        if tuple(self.agent_pos) == self.fire_pos and self.has_bucket and not self.fire_out:
            self.fire_out = True
            reward += 10

        # Reaching goal
        if tuple(self.agent_pos) == self.goal_pos:
            if self.fire_out:
                reward += 10
                terminated = True
            else:
                reward -= 10
                terminated = True
        else:
            terminated = False

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([*self.agent_pos, int(self.has_bucket), int(self.fire_out)], dtype=np.int32)
    def save_state(self):
        return copy.deepcopy(self)

    def reset_to(self, saved_env):
        self.__dict__.update(saved_env.__dict__)


    def render(self):
        pass

    def close(self):
        pass                
