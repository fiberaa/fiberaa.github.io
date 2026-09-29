#!/usr/bin/env python3
"""Export the BFM-0.5 **actor** (proprio-only, deterministic mean) to ONNX.

The deployed policy is just ``pi(a | s, z)`` evaluated at the distribution mean:
the actor consumes three obs keys — ``state`` (64), ``last_action`` (29),
``history_actor`` (H*93) — plus the skill vector ``z`` (z_dim), and emits an
action in ``[-1, 1]`` (29). No camera, no privileged state, no FK.

The per-key observation **BatchNorm normalizers are folded into the graph** (we
run the real ``ObsNormalizer`` in eval mode inside the traced module), so the C++
side feeds RAW BFM-native obs and gets the exact same output as
``lsvla.bfm.act(obs, z, mean=True)``.

Also writes ``actor_meta.json`` with everything the C++ controller needs to build
obs and post-process the action: default_q, action pipeline constants, per-joint
rescale, obs dims, history layout, z_dim / z-sphere radius, joint names.

Run (local, in the uv venv):

    uv run python deploy/export_actor_onnx.py \
        --checkpoint checkpoints/model_15500.pt --out deploy/exported
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from lsvla import load_bfm
from lsvla import obs_format as ofmt
from lsvla.sim.mujoco_sim import (
    CTRL_DT,
    DECIMATION,
    DEFAULT_DOF_POS,
    EFFORT_LIMIT_PER_JOINT,
    KD_PER_JOINT,
    KP_PER_JOINT,
)

# 29-DOF G1 joint order (USD / BFM action order). Matches the deploy stack's
# g1_definition.JOINT_NAMES; duplicated here so the export is self-contained.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class ActorONNX(torch.nn.Module):
    """Trace-friendly wrapper: (state, last_action, history_actor, z) -> mean action.

    Reproduces ``FBCprAuxPolicy.act(obs, z, mean=True)`` for the MLP actor:
      1. per-key BatchNorm normalize (folded, eval mode)
      2. concat in actor_input_keys order (state, last_action, history_actor)
      3. embed_s(obs), embed_z([obs | z]); policy([s_emb | z_emb]); tanh
    """

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.normalizer = policy._obs_normalizer
        self.actor = policy._actor
        # Guard: this exporter only supports the canonical MLP actor.
        from lsvla.fb_cpr_policy import Actor

        if not isinstance(self.actor, Actor):
            raise TypeError(
                f"export supports the MLP Actor only, got {type(self.actor).__name__}"
            )
        self.eval()

    def forward(
        self,
        state: torch.Tensor,
        last_action: torch.Tensor,
        history_actor: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        norm = self.normalizer(
            {"state": state, "last_action": last_action, "history_actor": history_actor}
        )
        obs = self.actor.input_filter(norm)  # concat in actor_input_keys order
        z_embedding = self.actor.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.actor.embed_s(obs)
        out = self.actor.policy(torch.cat([s_embedding, z_embedding], dim=-1))
        return torch.tanh(out)  # deterministic mean action in [-1, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_15500.pt")
    ap.add_argument("--out", default="deploy/exported")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bfm = load_bfm(args.checkpoint, device=args.device)
    policy = bfm.policy

    obs_dims = bfm.cfg.obs_dims
    z_dim = bfm.z_dim
    state_dim = obs_dims["state"]
    la_dim = obs_dims["last_action"]
    hist_dim = obs_dims["history_actor"]
    H = hist_dim // ofmt._HIST_FRAME_DIM
    assert H * ofmt._HIST_FRAME_DIM == hist_dim, (hist_dim, ofmt._HIST_FRAME_DIM)
    assert state_dim == 64 and la_dim == 29, (state_dim, la_dim)

    wrapper = ActorONNX(policy).to(args.device)

    # Dummy inputs (batch 1). Values irrelevant to the trace (BN buffers are fixed).
    st = torch.zeros(1, state_dim, device=args.device)
    la = torch.zeros(1, la_dim, device=args.device)
    hi = torch.zeros(1, hist_dim, device=args.device)
    z = bfm.sample_z(1).to(args.device)

    onnx_path = os.path.join(args.out, "actor.onnx")
    # Legacy TorchScript exporter (dynamo=False): emits a single-file model at a
    # lower IR version (compatible with onnxruntime 1.17.x on the deploy box) and
    # decomposes Mish at opset<18 (avoids the opset-18 Mish version-converter path).
    torch.onnx.export(
        wrapper,
        (st, la, hi, z),
        onnx_path,
        input_names=["state", "last_action", "history_actor", "z"],
        output_names=["action"],
        dynamic_axes={
            "state": {0: "batch"}, "last_action": {0: "batch"},
            "history_actor": {0: "batch"}, "z": {0: "batch"}, "action": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote {onnx_path}")

    # ---- numerical parity check vs bfm.act ---------------------------------
    rng = np.random.default_rng(0)
    st_n = torch.tensor(rng.standard_normal((4, state_dim)), dtype=torch.float32)
    la_n = torch.tensor(rng.standard_normal((4, la_dim)), dtype=torch.float32)
    hi_n = torch.tensor(rng.standard_normal((4, hist_dim)), dtype=torch.float32)
    z_n = bfm.sample_z(4)
    obs = {"state": st_n, "last_action": la_n, "history_actor": hi_n}
    a_ref = bfm.act(obs, z_n, mean=True).cpu().numpy()

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    a_onnx = sess.run(
        ["action"],
        {
            "state": st_n.numpy(), "last_action": la_n.numpy(),
            "history_actor": hi_n.numpy(), "z": z_n.cpu().numpy(),
        },
    )[0]
    max_err = float(np.abs(a_ref - a_onnx).max())
    print(f"[export] parity vs bfm.act: max|Δaction| = {max_err:.3e}")
    if max_err > 1e-4:
        raise SystemExit(f"ONNX parity check FAILED (max_err={max_err:.3e})")

    # ---- joint limits (for the deploy-time safety e-stop) ------------------
    # Read the authoritative per-joint position limits from the same vendored G1
    # XML the sim uses, so the C++ controller can auto-e-stop on a limit breach.
    import mujoco  # noqa: E402
    from lsvla.sim import DEFAULT_G1_XML

    mj = mujoco.MjModel.from_xml_path(DEFAULT_G1_XML)
    joint_limits = []
    for n in JOINT_NAMES:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = mj.jnt_range[jid]
        joint_limits.append([float(lo), float(hi)])

    # ---- deploy metadata ---------------------------------------------------
    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "iter": bfm.cfg.iter,
        "z_dim": z_dim,
        "z_sphere_radius": math.sqrt(z_dim),  # sample_z / project_z scale (norm_z)
        "action_dim": bfm.action_dim,
        "obs_dims": {"state": state_dim, "last_action": la_dim, "history_actor": hist_dim},
        "history_length": H,
        # history_actor term-major layout (each term flattens H lagged frames):
        "history_term_order": ["actions", "ang_vel", "dof_pos", "dof_vel", "gravity"],
        "history_term_dims": list(ofmt._HIST_TERM_DIMS),
        # state (64) layout: [dof_pos_dev(29), dof_vel(29), proj_gravity(3), ang_vel_scaled(3)]
        "state_layout": ["dof_pos_dev(29)", "dof_vel(29)", "proj_gravity(3)", "ang_vel_scaled(3)"],
        # action pipeline: target = clip(a*NORM, ±CLIP) * SCALE * rescale + default_q
        "action_normalize_to": ofmt.ACTION_NORMALIZE_TO,
        "action_clip": ofmt.ACTION_CLIP,
        "action_scale": ofmt.ACTION_SCALE,
        "action_rescale": list(ofmt.ACTION_RESCALE),
        "ang_vel_scale": ofmt.ANG_VEL_SCALE,
        "default_q": list(DEFAULT_DOF_POS),
        # BeyondMimic PD gains + effort limits the BFM was trained under. The
        # deploy peer's onboard PD (lowcmd motor kp/kd, τ clipped to effort) must
        # match these for sim<->train<->real consistency.
        "kp": [float(x) for x in KP_PER_JOINT],
        "kd": [float(x) for x in KD_PER_JOINT],
        "effort_limit": list(EFFORT_LIMIT_PER_JOINT),
        "control_dt": CTRL_DT,          # 0.02 s (50 Hz)
        "control_freq": round(1.0 / CTRL_DT),
        "sim_decimation": DECIMATION,   # 1 kHz physics / 50 Hz control (sim2sim)
        # Per-joint position limits (rad, from the G1 XML) for the safety e-stop.
        "joint_limits": joint_limits,
        # Safety e-stop thresholds (deploy-time): fraction of the joint range to
        # trip at, base roll magnitude (rad) to trip at, and joints exempt from
        # the position check (ankles/wrists get pegged by the action-rescale, so
        # a limit there is not a fault — matches the reference deploy config).
        # Values match the proven real-robot config (Latent-Deploy/server.py):
        # legs (hip/knee/ankle) + wrists are driven near their limits by the
        # policy, so they're exempt from the position check; roll_limit is loose
        # (a hard fall guard, not a tilt guard).
        "estop_joint_pos_frac": 0.98,
        "estop_roll_limit": 5.57,
        "estop_ignore_joints": [
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_knee_joint", "right_knee_joint",
            "left_wrist_roll_joint", "right_wrist_roll_joint",
            "left_wrist_pitch_joint", "right_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_wrist_yaw_joint",
        ],
        "joint_names": JOINT_NAMES,
        "actor_input_keys": ["state", "last_action", "history_actor"],
        "onnx_parity_max_err": max_err,
    }
    meta_path = os.path.join(args.out, "actor_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {meta_path}")
    print(f"[export] z_dim={z_dim} H={H} state={state_dim} hist={hist_dim} action={bfm.action_dim}")


if __name__ == "__main__":
    main()
