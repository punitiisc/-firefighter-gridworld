import os
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "results"
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, "algo_comparison.png")
MAX_EPISODES = 300  # Only compare up to 300 episodes

# Load reward data from each CSV file
reward_data = {}

# PPO
ppo_csv = "logs/ppo/monitor.csv"
if os.path.exists(ppo_csv):
    ppo_data = np.genfromtxt(ppo_csv, delimiter=",", skip_header=1)
    if ppo_data.ndim > 1:
        reward_data["PPO"] = ppo_data[:, 0][:MAX_EPISODES]

# DQN
dqn_csv = "logs/dqn/monitor.csv"
if os.path.exists(dqn_csv):
    dqn_data = np.genfromtxt(dqn_csv, delimiter=",", skip_header=1)
    if dqn_data.ndim > 1:
        reward_data["DQN"] = dqn_data[:, 0][:MAX_EPISODES]

# MCTS
mcts_rewards_path = os.path.join(RESULT_DIR, "mcts_rewards.npy")
if os.path.exists(mcts_rewards_path):
    mcts_rewards = np.load(mcts_rewards_path)
    reward_data["MCTS"] = mcts_rewards[:MAX_EPISODES]
else:
    print("⚠️ Skipping MCTS (mcts_rewards.npy not found)")

# Plot all
plt.figure(figsize=(10, 6))
for algo, rewards in reward_data.items():
    plt.plot(np.arange(len(rewards)), rewards, label=algo)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Algorithm Comparison (first 300 episodes)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
print(f"✅ Comparison plot saved to {PLOT_SAVE_PATH}")

