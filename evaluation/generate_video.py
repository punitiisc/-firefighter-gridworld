import imageio
import os

def generate_gif(frame_dir="assets/assets_video", output_path="assets/firefighter_episode.gif", fps=3, cleanup=True):
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not files:
        print("⚠️ No frames found to generate GIF.")
        return

    images = [imageio.v2.imread(os.path.join(frame_dir, f)) for f in files]
    imageio.mimsave(output_path, images, fps=fps)
    print(f"✅ GIF saved to {output_path}")

    if cleanup:
        for f in files:
            os.remove(os.path.join(frame_dir, f))
        print("🧹 Frame PNGs deleted after GIF creation.")

