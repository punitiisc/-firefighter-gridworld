import os
import numpy as np
import matplotlib.pyplot as plt
from env.firefighter_env import FireFighterEnv
import random
import copy

PLOT_PATH = "results/mcts_reward_plot.png"
MODEL_DIR = "models/mcts_firefighter"
os.makedirs(MODEL_DIR, exist_ok=True)

class MCTSAgent:
    def __init__(self, env, n_simulations=50):
        self.env = env
        self.n_simulations = n_simulations

    
    def rollout(self, env_copy):
        total_reward = 0
        for _ in range(10):  # limit rollout depth
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


def train_mcts(num_episodes=300):
    from tqdm import trange

    env = FireFighterEnv()
    all_rewards = []

    for ep in trange(num_episodes, desc="Training with MCTS"):
        obs, _ = env.reset()
        agent = MCTSAgent(env)
        total_reward = 0

        for step in range(60):
            state = env.save_state()
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        all_rewards.append(total_reward)

    # Save plot
    plt.figure()
    plt.plot(all_rewards, label="MCTS Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MCTS Training Performance")
    plt.grid()
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"✅ MCTS reward plot saved to {PLOT_PATH}")
    # Final stats
    print(f"🔍 MCTS Evaluation over {len(all_rewards)} episodes:")
    print(f"   → Mean reward     : {np.mean(all_rewards):.2f}")
    print(f"   → Std deviation   : {np.std(all_rewards):.2f}")
    print(f"   → Max reward      : {np.max(all_rewards)}")
    print(f"   → Min reward      : {np.min(all_rewards)}")
    np.save("results/mcts_rewards.npy", np.array(all_rewards))

if __name__ == "__main__":
    train_mcts(num_episodes=300)




