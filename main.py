import os
import shutil
from env.firefighter_env import FireFighterEnv
from env.renderer import FireFighterRenderer
from evaluation.generate_video import generate_gif

SAVE_BASE = "assets/assets_video"
GIF_OUTPUT_BASE = "assets"
NUM_EPISODES = 5
NUM_STEPS = 60

for ep in range(NUM_EPISODES):
    print(f"\n🎬 Starting Episode {ep + 1}")

    # Step 1: Prepare directories
    SAVE_DIR = os.path.join(SAVE_BASE, f"ep_{ep + 1}")
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Step 2: Init environment & renderer
    env = FireFighterEnv()
    renderer = FireFighterRenderer(save_dir=SAVE_DIR)
    obs, _ = env.reset()
    total_reward = 0

    # Step 3: Run 60-step episode
    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        agent_pos = obs[:2]
        has_bucket = obs[2]
        fire_out = obs[3]

        renderer.render(agent_pos, has_bucket, fire_out,
                        env.bucket_pos, env.fire_pos,
                        env.goal_pos, env.walls,
                        reward)  # Pass reward for plot

    renderer.close()
    env.close()

    print(f"✅ Episode {ep + 1} complete. Total reward: {total_reward}")

    # Step 4: Generate GIF and delete PNGs
    output_gif = os.path.join(GIF_OUTPUT_BASE, f"firefighter_ep{ep + 1}.gif")
    generate_gif(frame_dir=SAVE_DIR,
                 output_path=output_gif,
                 fps=3,
                 cleanup=True)

