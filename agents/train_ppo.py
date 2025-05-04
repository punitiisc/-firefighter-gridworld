import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from env.firefighter_env import FireFighterEnv

SAVE_PATH = "models/ppo_firefighter"
PLOT_PATH = "results/ppo_reward_plot.png"
LOG_PATH = "logs/ppo/"

# Vectorized environment factory
def make_env():
    return FireFighterEnv()

# Create vectorized environment
env = DummyVecEnv([make_env for _ in range(4)])
env = VecMonitor(env, filename=os.path.join(LOG_PATH, "monitor.csv"))

# Evaluation env
eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

# Callback to save best model
callback = EvalCallback(eval_env,
                        best_model_save_path=SAVE_PATH,
                        log_path=LOG_PATH,
                        eval_freq=500,
                        deterministic=True,
                        render=False)

# Instantiate PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH,
            learning_rate=3e-4, n_steps=128, batch_size=64, gamma=0.99)

# Train
model.learn(total_timesteps=20000, callback=callback)

# Save final model
model.save(os.path.join(SAVE_PATH, "ppo_final"))

# Plot reward curve
monitor_data = np.genfromtxt(os.path.join(LOG_PATH, "monitor.csv"), delimiter=",", skip_header=1)
if monitor_data.ndim > 1:
    rewards = monitor_data[:, 0]  # episode rewards
    plt.plot(np.arange(len(rewards)), rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Performance")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    print(f"✅ Reward plot saved to {PLOT_PATH}")

print(f"✅ Training complete. Best model saved to: {SAVE_PATH}/best_model.zip")

