#!/usr/bin/env python3
"""Headless MuJoCo check: H1 keyframe + PD hold vs RL-style PD at training default pose."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

MJCF = Path.home() / "project/rl_sar/src/rl_sar_zoo/h1_description/mjcf/scene.xml"
DEFAULT_DOF_POS = np.array(
    [
        0.0, 0.0, -0.28, 0.79, -0.52,
        0.0, 0.0, -0.28, 0.79, -0.52,
        0.0,
        0.28, 0.0, 0.0, 0.52,
        0.28, 0.0, 0.0, 0.52,
    ],
    dtype=np.float64,
)
RL_KP = np.array(
    [
        150.0, 150.0, 200.0, 200.0, 20.0,
        150.0, 150.0, 200.0, 200.0, 20.0,
        200.0,
        40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0,
    ],
    dtype=np.float64,
)
RL_KD = np.array(
    [
        5.0, 5.0, 5.0, 5.0, 4.0,
        5.0, 5.0, 5.0, 5.0, 4.0,
        5.0,
        10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0,
    ],
    dtype=np.float64,
)
DT = 0.005
STEPS = int(5.0 / DT)


def joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def main() -> int:
    if not MJCF.is_file():
        print(f"MJCF not found: {MJCF}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        print("home keyframe missing", file=sys.stderr)
        return 1

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    joint_names = [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    ]
    q_adr = [joint_qpos_adr(model, n) for n in joint_names]
    q0 = np.array([data.qpos[a] for a in q_adr])
    max_diff = float(np.max(np.abs(q0 - DEFAULT_DOF_POS)))
    print(f"keyframe vs default_dof_pos max |diff| = {max_diff:.4f} rad")

    heights = []
    for step in range(STEPS):
        for act_id in range(model.nu):
            jnt_id = model.actuator_trnid[act_id, 0]
            q = data.qpos[model.jnt_qposadr[jnt_id]]
            dq = data.qvel[model.jnt_dofadr[jnt_id]]
            target = DEFAULT_DOF_POS[act_id]
            data.ctrl[act_id] = RL_KP[act_id] * (target - q) - RL_KD[act_id] * dq
        mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))

    t_fall = next((i * DT for i, z in enumerate(heights) if z < 0.6), None)
    print(f"pelvis z: start={heights[0]:.3f} @2.5s={heights[-1]:.3f} min={min(heights):.3f}")
    if t_fall is not None:
        print(f"FAIL: pelvis z<0.6 at t={t_fall:.2f}s (open-loop PD cannot balance humanoid)")
        return 2
    print("PASS: pelvis stayed above 0.6m for 5s with open-loop PD (unexpected for humanoid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
