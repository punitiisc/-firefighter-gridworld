import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from env.firefighter_env import FireFighterEnv

SAVE_PATH = "models/dqn_firefighter"
PLOT_PATH = "results/dqn_reward_plot.png"
LOG_PATH = "logs/dqn/"

# Create vectorized environment
def make_env():
    return FireFighterEnv()

env = DummyVecEnv([make_env for _ in range(4)])
env = VecMonitor(env, filename=os.path.join(LOG_PATH, "monitor.csv"))

eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

# Callback to save best model
callback = EvalCallback(eval_env,
                        best_model_save_path=SAVE_PATH,
                        log_path=LOG_PATH,
                        eval_freq=500,
                        deterministic=True,
                        render=False)

# Instantiate DQN model
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH,
            learning_rate=1e-4, buffer_size=10000, exploration_fraction=0.1,
            exploration_final_eps=0.05, train_freq=4, batch_size=64, gamma=0.99)

# Train
model.learn(total_timesteps=20000, callback=callback)

# Save final model
model.save(os.path.join(SAVE_PATH, "dqn_final"))

# Plot reward curve
monitor_data = np.genfromtxt(os.path.join(LOG_PATH, "monitor.csv"), delimiter=",", skip_header=1)
if monitor_data.ndim > 1:
    rewards = monitor_data[:, 0]
    plt.plot(np.arange(len(rewards)), rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    print(f"✅ Reward plot saved to {PLOT_PATH}")

print(f"✅ Training complete. Best model saved to: {SAVE_PATH}/best_model.zip")

