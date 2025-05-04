import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

class FireFighterRenderer:
    def __init__(self, save_dir=None):
        self.grid_size = 4
        self.save_dir = save_dir
        self.frame_idx = 0
        self.sprite_dir = "assets/sprites"
        self.rewards = []

        self.fig, (self.ax_grid, self.ax_plot) = plt.subplots(1, 2, figsize=(10, 5),
                                                              gridspec_kw={'width_ratios': [1, 1]})

        # Load sprites
        def load(name):
            path = os.path.join(self.sprite_dir, f"{name}.png")
            img = mpimg.imread(path)
            assert img is not None, f"Failed to load: {path}"
            return img

        self.sprites = {
            "robot_white": load("robot_white"),
            "robot_blue": load("robot_blue"),
            "robot_green": load("robot_green"),
            "bucket": load("bucket"),
            "fire": load("fire"),
            "goal": load("goal"),
            "wall": load("wall"),
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def render(self, agent_pos, has_bucket, fire_out, bucket_pos, fire_pos, goal_pos, walls, reward):
        self.ax_grid.clear()
        self.ax_plot.clear()
        self.rewards.append(reward)

        self.ax_grid.set_xlim(0, self.grid_size)
        self.ax_grid.set_ylim(0, self.grid_size)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.set_aspect('equal')

        # White background grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.ax_grid.add_patch(plt.Rectangle((y, self.grid_size - 1 - x), 1, 1,
                                                     edgecolor='black', facecolor='white', linewidth=1))

        # Function to place sprites
        def draw(sprite_name, x, y):
            sprite = self.sprites[sprite_name]
            self.ax_grid.imshow(sprite,
                                extent=(y, y + 1, self.grid_size - 1 - x, self.grid_size - x),
                                zorder=10)

        # Draw elements
        for wx, wy in walls:
            draw("wall", wx, wy)

        if not fire_out:
            fx, fy = fire_pos
            draw("fire", fx, fy)

        bx, by = bucket_pos
        draw("bucket", bx, by)

        gx, gy = goal_pos
        draw("goal", gx, gy)

        ax, ay = agent_pos
        if has_bucket and not fire_out:
            robot_color = "robot_blue"
        elif fire_out:
            robot_color = "robot_green"
        else:
            robot_color = "robot_white"
        draw(robot_color, ax, ay)

        self.ax_grid.set_title(f"Step {self.frame_idx}")

        # Reward plot
        self.ax_plot.plot(np.cumsum(self.rewards), color='green', marker='o')
        self.ax_plot.set_title("Cumulative Reward")
        self.ax_plot.set_xlabel("Step")
        self.ax_plot.set_ylabel("Total Reward")
        self.ax_plot.grid(True)

        if self.save_dir:
            frame_path = os.path.join(self.save_dir, f"frame_{self.frame_idx:03d}.png")
            self.fig.tight_layout()
            self.fig.savefig(frame_path)
            self.frame_idx += 1
        else:
            plt.pause(0.3)
            plt.draw()

    def close(self):
        plt.close(self.fig)

