import os
os.environ["MUJOCO_GL"] = "egl"
import sys, numpy as np, torch, mujoco, imageio

SCRIPT_DIR = os.path.expanduser("~/robot_learning/so101_mujoco")
CKPT       = os.path.join(SCRIPT_DIR, "outputs/bc_visual_policy.pt")
VIDEO_OUT  = os.path.join(SCRIPT_DIR, "demo_bc.mp4")
N_EPISODES = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, SCRIPT_DIR)
from train_ppo_v2 import SO101Env, VisualActorCritic, CFG, SCENE_XML

env = SO101Env(SCENE_XML, CFG)
print("Env ok | device=" + DEVICE)

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
policy = VisualActorCritic().to(DEVICE)
policy.load_bc(CKPT)
policy.eval()
print("BC policy loaded")

renderer = mujoco.Renderer(env.model, height=480, width=640)

def get_action(img, state):
    a, _, _ = policy.get_action(
        img.unsqueeze(0).to(DEVICE),
        torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
    return a

all_frames, stats = [], []
print(f"Running {N_EPISODES} episodes...")
for ep in range(N_EPISODES):
    img, state = env.reset()
    frames, ep_rew, grasps, lifts, step = [], 0.0, 0, 0, 0
    done = False
    while not done and step < 500:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.08]
        cam.distance = 0.7
        cam.azimuth = 135
        cam.elevation = -22
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())
        action = get_action(img, state)
        (img, state), rew, done, info = env.step(action)
        ep_rew += rew
        grasps += int(info.get("grasped", False))
        lifts  += int(info.get("lifted",  False))
        step   += 1
    all_frames.extend(frames)
    stats.append(dict(rew=ep_rew, g=grasps, l=lifts))
    print(f"  Ep {ep+1:2d} | Rew={ep_rew:8.2f} | grasps={grasps} lifts={lifts}")

print(f"Saving {len(all_frames)} frames -> {VIDEO_OUT}")
with imageio.get_writer(VIDEO_OUT, fps=30, quality=8) as w:
    for f in all_frames:
        w.append_data(f)
print("Video saved:", VIDEO_OUT)
print(f"SUMMARY: Rew={np.mean([s["rew"] for s in stats]):.2f}  grasps={np.mean([s["g"] for s in stats]):.1f}  lifts={np.mean([s["l"] for s in stats]):.1f}")
