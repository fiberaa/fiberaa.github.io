#!/usr/bin/env python3
"""Build a small navigation z-bank for deployment WITHOUT the 39 GB replay.

Builds separate positive and negative motion endpoints using target-velocity
rewards. Opposite z is not assumed to produce opposite motion:

    z_forward     reward = -(vx - target_linear)^2
    z_backward    reward = -(vx + target_linear)^2
    z_left/right  reward = -(vy -/+ target_linear)^2
    z_turn_l/r    reward = -(wz -/+ target_yaw)^2
    z_squat       reward = -(root_height - target_height)^2
    z_stand       reward = -(vx^2 + vy^2 + wz^2)

Emits ``deploy/exported/z_bank.json`` (plain float lists) + ``z_bank.pt``. The
C++ controller loads the JSON so a skill name maps to a z vector on the sphere.

Run (local, sim extra):

    MUJOCO_GL=egl uv run --extra sim python deploy/build_deploy_zbank.py \
        --checkpoint checkpoints/model_14500.pt --out deploy/exported \
        --episodes 24 --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import hashlib

import numpy as np
import torch

from lsvla import load_bfm
from lsvla.nav.z_bank import IDX_VX, IDX_VY, IDX_WZ, slerp_z
from lsvla.sim import make_sim


@torch.no_grad()
def measure_response(bfm, z, device, steps=100, warm=20):
    """Roll a single basis z from neutral and return its steady-state body-frame
    velocity [v_fwd, v_lat, wz] (mean over post-warmup steps). Falls over ->
    zeros. Mirrors scripts/test_global_track_feedback._measure."""
    s = make_sim(history_length=9, render_size_hw=(64, 64), action_rescale=bfm.cfg.action_rescale, enable_rendering=False)
    s.reset_to_pose(xy=(0, 0), yaw=0.0, height=0.8)
    zt = torch.as_tensor(z, device=device, dtype=torch.float32).reshape(1, -1)
    samples = []
    for t in range(steps):
        q = s.get_qpos_36()
        # yaw from root quat (wxyz at qpos[3:7]); z-height at qpos[2].
        w, x, y, zq = float(q[3]), float(q[4]), float(q[5]), float(q[6])
        yaw = math.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        if q[2] < 0.3:
            break
        lin, ang = s._get_root_vel()  # world lin, body ang
        vf = math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1]
        vl = -math.sin(yaw) * lin[0] + math.cos(yaw) * lin[1]
        if t >= warm:
            samples.append([vf, vl, float(ang[2]), float(q[2])])
        s.step(bfm.act(s.get_obs_dict(device=device), zt, mean=True).squeeze(0).cpu().numpy())
    return np.mean(samples, axis=0) if samples else np.zeros(4)


@torch.no_grad()
def measure_endpoints(bfm, endpoints, device):
    """Measure each independently optimized endpoint without fitting a linear model."""
    responses = {}
    for name, z in endpoints.items():
        response = measure_response(bfm, z, device)
        responses[name] = response.astype(float).tolist()
        print(f"[zbank] {name:>10s} response [fwd,lat,yaw,height] = {response}")
    return responses


@torch.no_grad()
def collect_samples(bfm, sim, episodes: int, steps: int, device: str):
    """Roll out the frozen BFM under a fresh random z each episode; gather the
    per-step next_obs and pelvis velocities."""
    states, las, hists, privs = [], [], [], []
    for ep in range(episodes):
        sim.reset_to_default()
        z = bfm.sample_z(1).to(device)
        for t in range(steps):
            obs = sim.get_obs_dict(device=device)
            a = bfm.act(obs, z, mean=True).squeeze(0).cpu().numpy()
            sim.step(a)
            # next_obs after the step
            states.append(sim.get_state())
            las.append(sim.get_last_action())
            hists.append(sim.get_history_actor())
            privs.append(sim.get_privileged_state())
        print(f"  episode {ep + 1}/{episodes} done ({steps} steps)")
    next_obs = {
        "state": torch.from_numpy(np.stack(states)).to(device),
        "last_action": torch.from_numpy(np.stack(las)).to(device),
        "history_actor": torch.from_numpy(np.stack(hists)).to(device),
        "privileged_state": torch.from_numpy(np.stack(privs)).to(device),
    }
    return next_obs


@torch.no_grad()
def reward_z_from_backward(bfm, backward, reward):
    """Project a centered scalar reward through a precomputed B(s') batch."""
    r = reward.reshape(-1, 1).to(backward.device, backward.dtype)
    r = r - r.mean()
    return bfm.project_z((r * backward).sum(dim=0, keepdim=True)).squeeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_14500.pt")
    ap.add_argument("--out", default="deploy/exported")
    ap.add_argument("--replay", default=None,
                    help="replay .pt (model_*.replay.pt). If given, derive skills + basis "
                         "from REAL training transitions (well-conditioned forward/turn), "
                         "instead of a random-z MuJoCo rollout (which barely locomotes).")
    ap.add_argument("--replay-samples", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--arc-blend", type=float, default=0.6)
    ap.add_argument("--target-linear", type=float, default=1.0,
                    help="target speed magnitude for forward/backward/left/right rewards")
    ap.add_argument("--target-yaw", type=float, default=3.14,
                    help="target yaw-rate magnitude for left/right turn rewards")
    ap.add_argument("--target-height", type=float, default=0.4,
                    help="target pelvis height for the squat reward")
    ap.add_argument("--calib-basis", action="store_true",
                    help="measure directional and squat endpoints and emit the stand-relative "
                         "interpolation block used by the controller")
    ap.add_argument(
        "--reduced-locomotion",
        action="store_true",
        help="derive backward=-forward while retaining independently optimized "
             "forward/left/right/turn-left/turn-right/stand endpoints",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    bfm = load_bfm(args.checkpoint, device=args.device, control_profile="bfm05_xl", verbose=False)

    # ── source the reward-weighting states: replay (preferred) or MuJoCo rollout ──
    if args.replay:
        from lsvla.replay import ReplayReader
        print(f"[zbank] sampling {args.replay_samples} transitions from {args.replay} ...")
        replay = ReplayReader.load(args.replay, map_location="cpu")
        g = torch.Generator(device="cpu").manual_seed(0)
        tr = replay.sample(args.replay_samples, generator=g)
        # B(s') consumes only these keys. Some replay versions store a shorter
        # actor history than the checkpoint; passing unrelated keys through the
        # global normalizer would incorrectly trigger a shape mismatch.
        next_obs = {
            k: tr.next_obs[k].to(args.device)
            for k in bfm.cfg.backward_input_keys
        }
        src = f"replay:{os.path.basename(args.replay)}"
    else:
        sim = make_sim(history_length=9, action_rescale=bfm.cfg.action_rescale, enable_rendering=False)
        print(f"[zbank] rolling out {args.episodes} eps x {args.steps} steps ...")
        next_obs = collect_samples(bfm, sim, args.episodes, args.steps, args.device)
        src = "mujoco_rollout"
    priv = next_obs["privileged_state"]
    vx, vy, wz = priv[:, IDX_VX], priv[:, IDX_VY], priv[:, IDX_WZ]
    root_height = priv[:, 0]
    print(f"[zbank] samples={priv.shape[0]}  vx.mean={float(vx.mean()):.3f} "
          f"vx.std={float(vx.std()):.3f}  wz.std={float(wz.std()):.3f}")

    print("[zbank] evaluating B(s') once for all reward objectives ...")
    backward = bfm.backward_map(next_obs)
    tv = float(args.target_linear)
    tw = float(args.target_yaw)
    th = float(args.target_height)
    z_forward = reward_z_from_backward(bfm, backward, -(vx - tv).square())
    z_backward = reward_z_from_backward(bfm, backward, -(vx + tv).square())
    z_left = reward_z_from_backward(bfm, backward, -(vy - tv).square())
    z_right = reward_z_from_backward(bfm, backward, -(vy + tv).square())
    z_turn_left = reward_z_from_backward(bfm, backward, -(wz - tw).square())
    z_turn_right = reward_z_from_backward(bfm, backward, -(wz + tw).square())
    z_squat = reward_z_from_backward(
        bfm, backward, -(root_height - th).square()
    )
    z_stand = reward_z_from_backward(bfm, backward, -(vx.square() + vy.square() + wz.square()))
    z_arc_left = slerp_z(z_turn_left, z_forward, args.arc_blend)
    z_arc_right = slerp_z(z_turn_right, z_forward, args.arc_blend)

    R2 = float(z_forward.norm()) ** 2
    print(f"[zbank] cos(forward,backward)={float(z_forward @ z_backward)/R2:+.3f} "
          f"cos(left,right)={float(z_left @ z_right)/R2:+.3f} "
          f"cos(turn_left,turn_right)={float(z_turn_left @ z_turn_right)/R2:+.3f}")

    if args.reduced_locomotion:
        z_backward = -z_forward
        skills = {
            "forward": z_forward,
            "backward": z_backward,
            "left": z_left,
            "right": z_right,
            "turn_left": z_turn_left,
            "turn_right": z_turn_right,
            "squat": z_squat,
            "stand": z_stand,
        }
    else:
        skills = {
            "forward": z_forward,
            "backward": z_backward,
            "left": z_left,
            "right": z_right,
            "turn_left": z_turn_left,
            "turn_right": z_turn_right,
            "squat": z_squat,
            "stand": z_stand,
            "arc_left": z_arc_left,
            "arc_right": z_arc_right,
        }
    meta = {
        "source": src,
        "source_checkpoint": "BFM-0.5-75275",
        "actor_z_dim": bfm.cfg.z_dim,
        "actor_onnx_sha256": hashlib.sha256(open("browser/actor.onnx","rb").read()).hexdigest(),
        "checkpoint": os.path.basename(args.checkpoint),
        "control_profile": bfm.cfg.control_profile,
        "z_radius": float(z_forward.norm()),
        "reward": "negative_squared_targets",
        "reward_objectives": {
            "linear": "-(v - signed_target_linear)^2",
            "yaw": "-(wz - signed_target_yaw)^2",
            "squat": "-(root_height - target_height)^2",
            "stand": "-(vx^2 + vy^2 + wz^2)",
        },
        "target_linear": tv,
        "target_yaw": tw,
        "target_height": th,
    }
    if args.reduced_locomotion:
        meta.update({
            "optimized_directions": [
                "forward",
                "left",
                "right",
                "turn_left",
                "turn_right",
                "squat",
                "stand",
            ],
            "derived_directions": {"backward": "negative_forward"},
            "yaw": "independent_target_reward_endpoints",
        })

    # The runtime interpolates from stand toward one independently optimized
    # endpoint per command sign. Multiple axes add stand-relative deltas, then
    # the combined z is projected once back onto the latent sphere.
    calib = None
    if args.calib_basis:
        print("[zbank] measuring target-reward movement endpoints ...")
        endpoints = {
            "forward": z_forward,
            "backward": z_backward,
            "left": z_left,
            "right": z_right,
            "turn_left": z_turn_left,
            "turn_right": z_turn_right,
            "squat": z_squat,
        }
        calib = {
            "mode": "directional_lerp_v1",
            "neutral": z_stand.cpu().numpy().astype(float).tolist(),
            **{name: z.cpu().numpy().astype(float).tolist()
               for name, z in endpoints.items()},
            "target_linear": tv,
            "target_yaw": tw,
            "target_height": th,
            "responses": measure_endpoints(bfm, endpoints, args.device),
        }
        meta["interpolation"] = "stand_relative_directional_lerp_v1"

    torch.save({**{k: v.cpu() for k, v in skills.items()}, "meta": meta,
                **({"calib": calib} if calib else {})},
               os.path.join(args.out, "z_bank.pt"))
    j = {k: v.cpu().numpy().astype(float).tolist() for k, v in skills.items()}
    j["meta"] = meta
    if calib:
        j["calib"] = calib
    with open(os.path.join(args.out, "z_bank.json"), "w") as f:
        json.dump(j, f)
    print(f"[zbank] wrote z_bank.json / z_bank.pt ({len(skills)} skills"
          f"{' + directional interpolation' if calib else ''}) to {args.out}")


if __name__ == "__main__":
    main()
