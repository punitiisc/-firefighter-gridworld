import os
import numpy as np
from env.firefighter_env import FireFighterEnv
from env.renderer import FireFighterRenderer
import copy
from tqdm import trange
from PIL import Image

SAVE_DIR = "assets/mcts_frames"
GIF_PATH = "assets/firefighter_mcts_success.gif"
os.makedirs(SAVE_DIR, exist_ok=True)

class MCTSAgent:
    def __init__(self, env, n_simulations=50):
        self.env = env
        self.n_simulations = n_simulations

    def rollout(self, env_copy):
        total_reward = 0
        for _ in range(10):
            action = env_copy.action_space.sample()
            _, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def select_action(self, state):
        action_rewards = np.zeros(self.env.action_space.n)

        for action in range(self.env.action_space.n):
            reward_sum = 0
            for _ in range(self.n_simulations):
                env_copy = copy.deepcopy(self.env)
                env_copy.reset_to(state)
                _, reward, _, _, _ = env_copy.step(action)
                reward += self.rollout(env_copy)
                reward_sum += reward
            action_rewards[action] = reward_sum / self.n_simulations

        return np.argmax(action_rewards)

# Run one visualized episode
def run_mcts_episode():
    env = FireFighterEnv()
    obs, _ = env.reset()
    agent = MCTSAgent(env)
    renderer = FireFighterRenderer(save_dir=SAVE_DIR)

    total_reward = 0

    for _ in range(60):
        state = env.save_state()
        action = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        renderer.render(env.agent_pos, env.has_bucket, env.fire_out,
                        env.bucket_pos, env.fire_pos, env.goal_pos, env.walls, reward)
        if terminated or truncated:
            break

    renderer.close()
    print(f"Total Reward (MCTS): {total_reward}")

    # Generate GIF
    frame_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])
    frames = [Image.open(os.path.join(SAVE_DIR, f)) for f in frame_files]
    frames[0].save(GIF_PATH, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)
    print(f"✅ MCTS animation saved to {GIF_PATH}")

if __name__ == "__main__":
    run_mcts_episode()

