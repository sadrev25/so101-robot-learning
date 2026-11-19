"""
Visual PPO v2 — Distance-Reduction Reward Shaping
  cd ~/robot_learning/so101_mujoco
  python train_ppo_v2.py

Reward design:
  Every step: reward = how much closer gripper got to current goal
  Milestone bonuses when goal is reached
  3 phases: reach/grasp → inspect → sort
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import mujoco
import matplotlib.pyplot as plt
from collections import deque

# ── Paths ──────────────────────────────────────────────────────
SCENE_XML  = os.path.expanduser(
    "~/robot_learning/mujoco_menagerie/robotstudio_so101/inspection_scene.xml")
BC_WEIGHTS = os.path.expanduser(
    "~/robot_learning/so101_mujoco/outputs/bc_visual_policy.pt")
SAVE_DIR   = os.path.expanduser("~/robot_learning/so101_mujoco/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
assert os.path.exists(SCENE_XML),  f"Scene not found: {SCENE_XML}"
assert os.path.exists(BC_WEIGHTS), f"BC weights not found: {BC_WEIGHTS}"

# ── Reward Config ──────────────────────────────────────────────
# Philosophy: dense shaping (distance reduction) + sparse milestone bonuses
# Agent always gets gradient signal — never stuck with zero reward
CFG = dict(
    # PPO hyperparams
    clip_eps      = 0.2,
    gamma         = 0.99,
    lam           = 0.95,
    lr            = 3e-4,
    entropy_coef  = 0.005,
    critic_coef   = 0.5,
    ppo_epochs    = 4,
    mini_batch    = 128,
    rollout_steps = 1024,
    total_steps   = 500_000,
    max_ep_steps  = 300,
    # Curriculum (by step count)
    phase2_start  = 150_000,   # add inspect reward
    phase3_start  = 300_000,   # add sort reward
    # Reward — shaping (per-step, based on distance reduction)
    shape_scale   = 5.0,       # reward = dist_reduction × scale (reduced: 10 caused -72 death spiral)
    # Reward — milestones (one-time bonus when condition first met)
    r_near        = 2.0,       # first time dist < 0.10
    r_grasp       = 10.0,      # gripper closed on brick
    r_lift        = 15.0,      # brick lifted > 3cm
    r_hold        = 0.3,       # per step while holding
    r_inspect     = 20.0,      # brick at inspection height > 10cm
    r_sort        = 30.0,      # brick in correct bowl
    r_wrong       = -10.0,     # brick in wrong bowl
    r_drop        = -1.0,      # dropped after lifting (reduced: -5 caused death spiral)
    r_time        = -0.002,    # small time penalty
    # Thresholds
    dist_near     = 0.10,
    dist_grasp    = 0.05,
    lift_z        = 0.03,
    hold_z        = 0.02,
    inspect_z     = 0.10,
    gripper_close = 0.8,
    bowl_radius   = 0.06,
)

# Bowl positions (x=0.45 after scene edit)
BOWLS = np.array([
    [0.45, -0.22, 0.0],  # 0 = left
    [0.45,  0.00, 0.0],  # 1 = centre
    [0.45,  0.22, 0.0],  # 2 = right
])
CORRECT_BOWL  = {"duplo_n1": 2, "duplo_n2": 0}
INSPECT_POINT = np.array([0.15, 0.0, 0.15])  # in front of camera

# ── Environment ────────────────────────────────────────────────
class SO101Env:
    def __init__(self, xml_path, cfg):
        self.cfg  = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=64, width=64)
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
        self._init_body_ids()
        self._reset_tracking()

    def _init_body_ids(self):
        def get_id(name):
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            except:
                return None
        self.gripper_id = get_id("gripper")
        self.brick_ids  = {n: get_id(n) for n in ["duplo_n1", "duplo_n2"]
                           if get_id(n) is not None}

    def _reset_tracking(self):
        self.step_count      = 0
        self.prev_dist       = None   # distance to current goal
        self.brick_was_lifted = False
        self.near_triggered  = False
        self.grasp_triggered = False
        self.lift_triggered  = False
        self.drop_triggered  = False
        self.inspect_triggered = False
        self.prev_bz         = 0.0

    def _gripper_pos(self):
        if self.gripper_id is not None:
            return self.data.xpos[self.gripper_id].copy()
        return np.zeros(3)

    def _brick_pos(self, name="duplo_n1"):
        bid = self.brick_ids.get(name)
        return self.data.xpos[bid].copy() if bid is not None else np.zeros(3)

    def _render(self):
        self.renderer.update_scene(self.data, camera="overhead_cam")
        img = self.renderer.render()
        return torch.from_numpy(img).float().permute(2,0,1) / 255.0

    def _get_obs(self):
        return self._render(), self.data.qpos[:6].copy().astype(np.float32)

    def _denorm(self, a):
        a = np.clip(a, -1, 1)
        return self.ctrl_min + (a + 1) / 2.0 * (self.ctrl_max - self.ctrl_min)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Small random perturbation so agent doesn't overfit to one start
        for bid in self.brick_ids.values():
            jid = self.model.body_jntadr[bid]
            if jid >= 0:
                self.data.qpos[jid:jid+3] += np.random.uniform(-0.015, 0.015, 3)
        mujoco.mj_forward(self.model, self.data)
        self._reset_tracking()
        # Initialise prev_dist so first step has valid shaping
        g = self._gripper_pos()
        b = self._brick_pos()
        self.prev_dist = np.linalg.norm(g - b)
        return self._get_obs()

    def step(self, action_norm, global_step=0):
        self.data.ctrl[:] = self._denorm(action_norm)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs    = self._get_obs()
        reward, info = self._compute_reward(global_step)
        done   = self.step_count >= self.cfg["max_ep_steps"]
        return obs, reward, done, info

    def _gripper_brick_contact(self, brick_name="duplo_n1"):
        """MuJoCo 4 real contact check — brick touching anything except floor."""
        bid = self.brick_ids.get(brick_name)
        if bid is None:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == bid and b2 != 0:
                return True
            if b2 == bid and b1 != 0:
                return True
        return False

    def _compute_reward(self, global_step):
        cfg  = self.cfg
        info = {}
        g    = self._gripper_pos()
        b    = self._brick_pos("duplo_n1")
        bz   = b[2]
        gc   = self.data.ctrl[5]  # gripper ctrl
        is_closed = gc > cfg["gripper_close"]

        # ── Phase 1: Reach → Grasp → Lift ──────────────────────
        # Dense: reward = how much closer gripper got this step
        curr_dist = np.linalg.norm(g - b)
        if self.grasp_triggered and not self.lift_triggered:
            r_shape = np.clip((bz - self.prev_bz) * cfg["shape_scale"] * 15, -0.3, 1.0)
        else:
            r_shape   = np.clip((self.prev_dist - curr_dist) * cfg["shape_scale"], -0.3, 1.0)
        self.prev_dist = curr_dist
        self.prev_bz   = bz
        info["dist"] = curr_dist

        # Milestone bonuses (triggered once per episode)
        r_near = 0.0
        if not self.near_triggered and curr_dist < cfg["dist_near"]:
            r_near = cfg["r_near"]
            self.near_triggered = True

        r_grasp = 0.0
        if not self.grasp_triggered and self._gripper_brick_contact() and is_closed:
            r_grasp = cfg["r_grasp"]
            self.grasp_triggered = True
            info["grasped"] = True

        r_lift = 0.0
        if not self.lift_triggered and bz > cfg["lift_z"]:
            r_lift = cfg["r_lift"]
            self.lift_triggered = True
            info["lifted"] = True

        r_hold = cfg["r_hold"] if bz > cfg["hold_z"] else 0.0

        # ── Phase 2: Inspect ───────────────────────────────────
        r_inspect = 0.0
        if global_step >= cfg["phase2_start"]:
            if not self.inspect_triggered and bz > cfg["inspect_z"]:
                r_inspect = cfg["r_inspect"]
                self.inspect_triggered = True
                info["inspected"] = True
            # Dense shaping toward inspection point (once lifted)
            elif bz > cfg["lift_z"]:
                inspect_dist = np.linalg.norm(b - INSPECT_POINT)
                prev_inspect = getattr(self, "_prev_inspect_dist", inspect_dist)
                r_inspect = (prev_inspect - inspect_dist) * cfg["shape_scale"] * 0.5
                self._prev_inspect_dist = inspect_dist

        # ── Phase 3: Sort ──────────────────────────────────────
        r_sort = r_wrong = 0.0
        if global_step >= cfg["phase3_start"]:
            correct = CORRECT_BOWL["duplo_n1"]
            for i, bowl in enumerate(BOWLS):
                bowl_dist = np.linalg.norm(b[:2] - bowl[:2])
                if bowl_dist < cfg["bowl_radius"] and bz < 0.02:
                    if i == correct:
                        r_sort = cfg["r_sort"]; info["sorted"] = True
                    else:
                        r_wrong = cfg["r_wrong"]; info["wrong_bowl"] = True
                # Dense shaping toward correct bowl (once inspected)
                elif self.inspect_triggered and i == correct:
                    prev_bowl_d = getattr(self, "_prev_bowl_dist", bowl_dist)
                    r_sort += (prev_bowl_d - bowl_dist) * cfg["shape_scale"] * 0.5
                    self._prev_bowl_dist = bowl_dist

        # ── Drop penalty ───────────────────────────────────────
        r_drop = 0.0
        if self.brick_was_lifted and bz < 0.01 and not self.drop_triggered:
            r_drop = cfg["r_drop"]; info["dropped"] = True
            self.drop_triggered = True
        self.brick_was_lifted = self.brick_was_lifted or (bz > cfg["lift_z"])

        reward = (r_shape + r_near + r_grasp + r_lift + r_hold
                  + r_inspect + r_sort + r_wrong + r_drop + cfg["r_time"])
        return float(reward), info

# ── Actor-Critic ───────────────────────────────────────────────
class VisualActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared CNN — same as BC policy
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.actor_mlp = nn.Sequential(
            nn.Linear(128+6, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 6), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(6) - 0.5)
        self.critic_mlp = nn.Sequential(
            nn.Linear(128+6, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _feat(self, img, state):
        return torch.cat([self.cnn(img), state], dim=1)

    def get_action(self, img, state):
        with torch.no_grad():
            x    = self._feat(img, state)
            mean = self.actor_mlp(x)
            std  = torch.exp(self.log_std.clamp(-4, 2))
            dist = Normal(mean, std)
            a    = dist.sample()
            return a.cpu().numpy(), dist.log_prob(a).sum(-1), self.critic_mlp(x).squeeze(-1)

    def evaluate(self, img, state, action):
        x    = self._feat(img, state)
        mean = self.actor_mlp(x)
        std  = torch.exp(self.log_std.clamp(-4, 2))
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(-1), self.critic_mlp(x).squeeze(-1), dist.entropy().sum(-1)

    def load_bc(self, path):
        ck  = torch.load(path, map_location="cpu", weights_only=False)
        bc  = ck["policy"]
        own = self.state_dict()
        n   = sum(1 for k,v in bc.items()
                  if k.replace("mlp.","actor_mlp.") in own
                  and own[k.replace("mlp.","actor_mlp.")].shape == v.shape)
        for k, v in bc.items():
            nk = k.replace("mlp.", "actor_mlp.")
            if nk in own and own[nk].shape == v.shape:
                own[nk].copy_(v)
        print(f"  BC warm-start: {n}/{len(bc)} layers loaded")
        return ck.get("s_mean"), ck.get("s_std")

# ── GAE ────────────────────────────────────────────────────────
def gae(rewards, values, dones, next_val, gamma, lam):
    advs = np.zeros_like(rewards)
    g    = 0.0
    for t in reversed(range(len(rewards))):
        nv  = next_val if t == len(rewards)-1 else values[t+1]
        d   = rewards[t] + gamma * nv * (1-dones[t]) - values[t]
        g   = d + gamma * lam * (1-dones[t]) * g
        advs[t] = g
    return advs, advs + values

# ── PPO Update ─────────────────────────────────────────────────
def ppo_update(net, opt, batch, cfg):
    imgs, states, acts, old_lp, rets, advs = batch
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    la = lc = ent = n = 0
    for _ in range(cfg["ppo_epochs"]):
        for mb in torch.randperm(len(imgs)).split(cfg["mini_batch"]):
            lp, v, e = net.evaluate(imgs[mb], states[mb], acts[mb])
            r = torch.exp(lp - old_lp[mb])
            a = advs[mb]
            loss = (-torch.min(r*a, r.clamp(1-cfg["clip_eps"], 1+cfg["clip_eps"])*a).mean()
                    + cfg["critic_coef"] * nn.MSELoss()(v, rets[mb])
                    - cfg["entropy_coef"] * e.mean())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            la += loss.item(); n += 1
    return la/n

# ── Training ───────────────────────────────────────────────────
def train():
    env = SO101Env(SCENE_XML, CFG)
    net = VisualActorCritic().to(DEVICE)
    s_mean, s_std = net.load_bc(BC_WEIGHTS)
    s_mean = torch.from_numpy(s_mean).float().to(DEVICE) if s_mean is not None else torch.zeros(6).to(DEVICE)
    s_std  = torch.from_numpy(s_std).float().to(DEVICE)  if s_std  is not None else torch.ones(6).to(DEVICE)
    opt    = optim.Adam(net.parameters(), lr=CFG["lr"])

    # Rollout buffers
    B = dict(imgs=[], states=[], acts=[], lps=[], vals=[], rews=[], dones=[])

    step  = 0
    ep_r  = deque(maxlen=20)
    ep_grasp = ep_lift = ep_sort = 0
    cur_r = 0
    best  = -float("inf")
    hist  = dict(rew=[], loss=[], grasp_rate=[], lift_rate=[])

    (img, joints) = env.reset()

    print(f"\nTraining — {CFG['total_steps']:,} steps on {DEVICE}")
    print(f"Ph1 reach+grasp  (0 → {CFG['phase2_start']:,})")
    print(f"Ph2 + inspect    ({CFG['phase2_start']:,} → {CFG['phase3_start']:,})")
    print(f"Ph3 + sort       ({CFG['phase3_start']:,} → {CFG['total_steps']:,})\n")

    while step < CFG["total_steps"]:
        net.eval()
        for _ in range(CFG["rollout_steps"]):
            img_t = img.unsqueeze(0).to(DEVICE)
            st    = ((torch.from_numpy(joints).float().to(DEVICE) - s_mean) / s_std).unsqueeze(0)
            a, lp, v = net.get_action(img_t, st)
            (ni, nj), r, done, info = env.step(a[0], step)

            B["imgs"].append(img); B["states"].append(joints.copy())
            B["acts"].append(a[0].copy()); B["lps"].append(lp.item())
            B["vals"].append(v.item()); B["rews"].append(r); B["dones"].append(float(done))

            cur_r += r
            if info.get("grasped"): ep_grasp += 1
            if info.get("lifted"):  ep_lift  += 1
            if info.get("sorted"):  ep_sort  += 1

            img, joints = ni, nj; step += 1

            if done:
                ep_r.append(cur_r)
                hist["rew"].append(cur_r)
                hist["grasp_rate"].append(ep_grasp)
                hist["lift_rate"].append(ep_lift)
                ep_grasp = ep_lift = ep_sort = cur_r = 0
                (img, joints) = env.reset()

            if step >= CFG["total_steps"]: break

        # Bootstrap
        with torch.no_grad():
            img_t = img.unsqueeze(0).to(DEVICE)
            st    = ((torch.from_numpy(joints).float().to(DEVICE) - s_mean) / s_std).unsqueeze(0)
            _, _, nv = net.get_action(img_t, st)

        advs, rets = gae(np.array(B["rews"]), np.array(B["vals"]),
                         np.array(B["dones"]), nv.item(), CFG["gamma"], CFG["lam"])

        imgs_t   = torch.stack(B["imgs"]).to(DEVICE)
        states_t = (torch.from_numpy(np.array(B["states"])).float().to(DEVICE) - s_mean) / s_std
        acts_t   = torch.from_numpy(np.array(B["acts"])).float().to(DEVICE)
        lps_t    = torch.tensor(B["lps"],  dtype=torch.float32).to(DEVICE)
        rets_t   = torch.tensor(rets, dtype=torch.float32).to(DEVICE)
        advs_t   = torch.tensor(advs, dtype=torch.float32).to(DEVICE)

        net.train()
        loss = ppo_update(net, opt, (imgs_t, states_t, acts_t, lps_t, rets_t, advs_t), CFG)
        hist["loss"].append(loss)

        for k in B: B[k].clear()

        mean_r = np.mean(ep_r) if ep_r else 0.0
        ph = 1 if step < CFG["phase2_start"] else 2 if step < CFG["phase3_start"] else 3
        print(f"Step {step:>7,} | Ph{ph} | Rew={mean_r:+.2f} | Loss={loss:.4f} | "
              f"grasps={sum(hist['grasp_rate'][-20:])}  lifts={sum(hist['lift_rate'][-20:])}")

        if mean_r > best and len(ep_r) >= 5:
            best = mean_r
            torch.save({"policy": net.state_dict(),
                        "s_mean": s_mean.cpu().numpy(),
                        "s_std":  s_std.cpu().numpy(),
                        "step":   step},
                       os.path.join(SAVE_DIR, "ppo_v2_best.pt"))
            print(f"  ✓ saved best (rew={best:.2f})")

        # Early warning
        if step == 30_000 and mean_r < -2.0:
            print("  ⚠ step 30K: reward not improving — check body IDs with test script")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].plot(hist["rew"]);  axes[0].set_title("Episode Reward"); axes[0].grid(True)
    axes[1].plot(hist["loss"]); axes[1].set_title("PPO Loss");        axes[1].grid(True)
    axes[2].plot(hist["grasp_rate"], label="grasps")
    axes[2].plot(hist["lift_rate"],  label="lifts")
    axes[2].set_title("Skill Counts"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "ppo_v2_curves.png")
    plt.savefig(out, dpi=150); print(f"\nCurves → {out}")
    plt.show()

if __name__ == "__main__":
    train()
