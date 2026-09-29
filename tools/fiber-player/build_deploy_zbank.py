#!/usr/bin/env python3
"""Build a small navigation z-bank for deployment WITHOUT the 39 GB replay.

``nav.z_bank.build_z_bank`` reads pelvis kinematics from replay transitions. For
deployment we instead **roll out the frozen BFM in MuJoCo** under random sphere
skills, collect ``(next_obs, pelvis vx/vy/wz)`` samples, and form the same
reward-weighted skills:

    z_forward     reward = +vx        (walk straight)
    z_turn_left   reward = +wz        (yaw left)
    z_turn_right  reward = -wz
    z_stand       reward = -(|vx|+|vy|+|wz|)

Emits ``deploy/exported/z_bank.json`` (plain float lists) + ``z_bank.pt``. The
C++ controller loads the JSON so a skill name maps to a z vector on the sphere.

Run (local, sim extra):

    MUJOCO_GL=egl uv run --extra sim python deploy/build_deploy_zbank.py \
        --checkpoint checkpoints/model_15500.pt --out deploy/exported \
        --episodes 24 --steps 200
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os

import numpy as np
import torch

import math

from lsvla import load_bfm
from lsvla.nav.z_bank import IDX_VX, IDX_VY, IDX_WZ, ZBank, _reward_z, slerp_z
from lsvla.sim import make_sim


@torch.no_grad()
def measure_response(bfm, z, device, steps=100, warm=20):
    """Roll a single basis z from neutral and return its steady-state body-frame
    velocity [v_fwd, v_lat, wz] (mean over post-warmup steps). Falls over ->
    zeros. Mirrors scripts/test_global_track_feedback._measure."""
    s = make_sim(history_length=9, render_size_hw=(64, 64), action_rescale=bfm.cfg.action_rescale, enable_rendering=False)
    s.reset_to_pose(xy=(0, 0), yaw=0.0, height=0.8)
    zt = torch.as_tensor(z, device=device, dtype=torch.float32).reshape(1, -1)
    vs = []
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
            vs.append([vf, vl, float(ang[2])])
        s.step(bfm.act(s.get_obs_dict(device=device), zt, mean=True).squeeze(0).cpu().numpy())
    return np.mean(vs, axis=0) if vs else np.zeros(3)


@torch.no_grad()
def calibrate_basis(bfm, Zf, Zl, Zt, device):
    """BEHAVIOR-SPACE calibration for the moving-z tracker, given the 3 directional
    basis z's (forward/lateral/turn).

    Reward-z orthogonality in z-space does NOT decouple the resulting MOTION (on
    e.g. model_72000 a 'forward' z also curves). Roll each basis z from neutral,
    measure its steady-state body velocity, and form the 3x3 response matrix M
    (columns = each basis z's [v_fwd,v_lat,wz]). Control via M^-1: a requested
    body velocity [vf,vl,wz] maps to basis coeffs M^-1 @ [vf,vl,wz], so the ACTUAL
    motion matches the request (M^-1 subsumes the turn sign). Returns (M, Minv).
    """
    cols = [measure_response(bfm, Zf, device),
            measure_response(bfm, Zl, device),
            measure_response(bfm, Zt, device)]
    M = np.stack(cols, axis=1)  # [3,3], M[:,j] = motion of basis j (fwd/lat/wz rows)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    print("[zbank] basis response M (rows=fwd/lat/wz, cols=Zf/Zl/Zt):")
    for r in range(3):
        print(f"[zbank]   {M[r]}")
    cond = float(np.linalg.cond(M))
    print(f"[zbank] cond(M) = {cond:.1f}  (>>100 => degenerate basis, weak moving-z)")
    return M, Minv


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_15500.pt")
    ap.add_argument("--out", default="deploy/exported")
    ap.add_argument("--actor-onnx", default="browser/actor.onnx")
    ap.add_argument("--replay", default=None,
                    help="replay .pt (model_*.replay.pt). If given, derive skills + basis "
                         "from REAL training transitions (well-conditioned forward/turn), "
                         "instead of a random-z MuJoCo rollout (which barely locomotes).")
    ap.add_argument("--replay-samples", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--arc-blend", type=float, default=0.6)
    ap.add_argument("--calib-basis", action="store_true",
                    help="behavior-space M^-1 calibration for the moving-z tracker "
                         "(recommended for cross-coupled checkpoints e.g. model_72000)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bfm = load_bfm(args.checkpoint, device=args.device, control_profile="bfm05_xl")

    # ── source the reward-weighting states: replay (preferred) or MuJoCo rollout ──
    if args.replay:
        from lsvla.replay import ReplayReader
        print(f"[zbank] sampling {args.replay_samples} transitions from {args.replay} ...")
        replay = ReplayReader.load(args.replay, map_location="cpu")
        g = torch.Generator(device="cpu").manual_seed(0)
        tr = replay.sample(args.replay_samples, generator=g)
        next_obs = {k: v.to(args.device) for k, v in tr.next_obs.items()}
        src = f"replay:{os.path.basename(args.replay)}"
    else:
        sim = make_sim(history_length=9, action_rescale=bfm.cfg.action_rescale, enable_rendering=False)
        print(f"[zbank] rolling out {args.episodes} eps x {args.steps} steps ...")
        next_obs = collect_samples(bfm, sim, args.episodes, args.steps, args.device)
        src = "mujoco_rollout"
    priv = next_obs["privileged_state"]
    vx, vy, wz = priv[:, IDX_VX], priv[:, IDX_VY], priv[:, IDX_WZ]
    print(f"[zbank] samples={priv.shape[0]}  vx.mean={float(vx.mean()):.3f} "
          f"vx.std={float(vx.std()):.3f}  wz.std={float(wz.std()):.3f}")

    z_forward = _reward_z(bfm, next_obs, vx)
    z_turn_left = _reward_z(bfm, next_obs, wz)
    z_turn_right = _reward_z(bfm, next_obs, -wz)
    z_lateral = _reward_z(bfm, next_obs, vy)   # +vy = strafe left (calib basis)
    z_stand = _reward_z(bfm, next_obs, -(vx.abs() + vy.abs() + wz.abs()))
    z_arc_left = slerp_z(z_turn_left, z_forward, args.arc_blend)
    z_arc_right = slerp_z(z_turn_right, z_forward, args.arc_blend)

    R2 = float(z_forward.norm()) ** 2
    print(f"[zbank] cos(forward,turn_left)={float(z_forward @ z_turn_left)/R2:+.3f} "
          f"(want ~0)  cos(forward,stand)={float(z_forward @ z_stand)/R2:+.3f}")

    meta = {"source": src, "z_radius": float(z_forward.norm()),
            "source_checkpoint": f"BFM-0.5-{bfm.cfg.iter}",
            "actor_z_dim": bfm.cfg.z_dim,
            "control_profile": bfm.cfg.control_profile}
    if os.path.isfile(args.actor_onnx):
        with open(args.actor_onnx, "rb") as actor_file:
            meta["actor_onnx_sha256"] = hashlib.file_digest(actor_file, "sha256").hexdigest()
    bank = ZBank(z_forward, z_turn_left, z_turn_right, z_stand,
                 arc_left=z_arc_left, arc_right=z_arc_right, meta=meta)

    # Behavior-space M^-1 calibration for the moving-z (velocity-command) path.
    calib = None
    if args.calib_basis:
        print("[zbank] behavior-space calibrating the fwd/lat/turn basis ...")
        Zf, Zl, Zt = z_forward, z_lateral, z_turn_left
        M, Minv = calibrate_basis(bfm, Zf, Zl, Zt, args.device)
        calib = {
            "basis_forward": Zf.cpu().numpy().astype(float).tolist(),
            "basis_lateral": Zl.cpu().numpy().astype(float).tolist(),
            "basis_turn": Zt.cpu().numpy().astype(float).tolist(),
            "M": M.astype(float).tolist(),        # rows fwd/lat/wz, cols Zf/Zl/Zt
            "Minv": Minv.astype(float).tolist(),  # body-vel request -> basis coeffs
        }
        meta["calibrated"] = True
        meta["turn_left_for_positive_error"] = bool(M[2, 2] > 0)

    torch.save({**{k: v.cpu() for k, v in bank.as_dict().items()}, "meta": meta,
                **({"calib": calib} if calib else {})},
               os.path.join(args.out, "z_bank.pt"))
    j = {k: v.cpu().numpy().astype(float).tolist() for k, v in bank.as_dict().items()}
    j["meta"] = meta
    if calib:
        j["calib"] = calib
    with open(os.path.join(args.out, "z_bank.json"), "w") as f:
        json.dump(j, f)
    print(f"[zbank] wrote z_bank.json / z_bank.pt ({len(bank.as_dict())} skills"
          f"{' + M^-1 calib' if calib else ''}) to {args.out}")
    # sanity: cos(turn_left, turn_right) should be strongly negative (antipodal)
    R2 = float(z_forward.norm()) ** 2
    cos_lr = float((z_turn_left @ z_turn_right) / R2)
    print(f"[zbank] cos(turn_left, turn_right) = {cos_lr:.3f}  (expect ~ -1)")


if __name__ == "__main__":
    main()
