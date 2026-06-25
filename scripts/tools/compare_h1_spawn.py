# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Compare H1 spawn physics: official USD vs My_H1 URDF (zero actions).

Route B step 1: isolate asset differences before retraining on URDF.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Compare H1 USD vs URDF spawn under zero actions.")
parser.add_argument(
    "--asset",
    type=str,
    choices=["usd", "urdf", "both"],
    default="both",
    help="Which robot asset to evaluate (default: both, sequentially).",
)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Unitree-My_H1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200, help="Simulation steps before exit.")
parser.add_argument(
    "--log_steps",
    type=str,
    default="1,50,100,200",
    help="Comma-separated step indices to print.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--_child_run",
    action="store_true",
    default=False,
    help=argparse.SUPPRESS,
)
args_cli, _ = parser.parse_known_args()

if args_cli.asset == "both" and not args_cli._child_run:
    child_cmd = [
        sys.executable,
        __file__,
        "--task",
        args_cli.task,
        "--num_envs",
        str(args_cli.num_envs),
        "--steps",
        str(args_cli.steps),
        "--log_steps",
        args_cli.log_steps,
        "--_child_run",
    ]
    if args_cli.disable_fabric:
        child_cmd.append("--disable_fabric")
    # Pass through AppLauncher args (e.g. --headless, --device cuda:0).
    known = {
        "--headless",
        "--livestream",
        "--enable_cameras",
        "--device",
        "--experience",
        "--kit_args",
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--asset":
            i += 2
            continue
        if arg in known:
            child_cmd.append(arg)
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                child_cmd.append(argv[i + 1])
                i += 2
                continue
        i += 1

    for asset in ("usd", "urdf"):
        print(f"\n>>> spawning isolated run: --asset {asset}", flush=True)
        subprocess.run(child_cmd + ["--asset", asset], check=True)
    sys.exit(0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG
from isaaclab_tasks.utils import parse_env_cfg

from robot_lab.assets.my_h1 import UNITREE_My_H1_CFG

ASSET_CHOICES = {
    "usd": ("H1_MINIMAL_CFG (USD)", H1_MINIMAL_CFG),
    "urdf": ("UNITREE_My_H1_CFG (h1_minimal.urdf)", UNITREE_My_H1_CFG),
}


def _patch_env_cfg_for_asset(env_cfg, asset_key: str) -> None:
    """Match reward body/joint regexes to the selected asset naming convention."""
    if asset_key == "usd":
        foot = ".*ankle_link"
        ankle_joints = [".*_ankle.*"]
        hip_joints = [".*_hip_yaw", ".*_hip_roll"]
        arm_joints = [".*_shoulder_.*", ".*_elbow"]
        torso = ["torso"]
    else:
        foot = ".*_ankle_link"
        ankle_joints = [".*_ankle_.*"]
        hip_joints = [".*hip_yaw.*", ".*hip_roll.*"]
        arm_joints = [".*shoulder.*", ".*elbow.*"]
        torso = ["torso_joint"]

    for term in (
        "feet_air_time",
        "feet_slide",
        "undesired_contacts",
        "contact_forces",
        "feet_contact",
        "feet_contact_without_cmd",
        "feet_stumble",
    ):
        rew = getattr(env_cfg.rewards, term, None)
        if rew is None:
            continue
        if "sensor_cfg" in rew.params:
            rew.params["sensor_cfg"].body_names = [foot]
        if "asset_cfg" in rew.params:
            rew.params["asset_cfg"].body_names = [foot]

    env_cfg.rewards.joint_pos_limits.params["asset_cfg"].joint_names = ankle_joints
    env_cfg.rewards.joint_deviation_hip_l1.params["asset_cfg"].joint_names = hip_joints
    env_cfg.rewards.joint_deviation_arms_l1.params["asset_cfg"].joint_names = arm_joints
    env_cfg.rewards.joint_deviation_torso_l1.params["asset_cfg"].joint_names = torso
    for term in ("base_height_l2", "body_lin_acc_l2"):
        rew = getattr(env_cfg.rewards, term, None)
        if rew is not None and "asset_cfg" in rew.params:
            rew.params["asset_cfg"].body_names = ["torso_link"]


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _run_asset(asset_key: str, label: str, robot_cfg, log_steps: set[int], max_steps: int) -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    _patch_env_cfg_for_asset(env_cfg, asset_key)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot.data, "body_names", getattr(robot, "body_names", [])))
    left_foot_idx = next((i for i, name in enumerate(body_names) if name == "left_ankle_link"), None)
    right_foot_idx = next((i for i, name in enumerate(body_names) if name == "right_ankle_link"), None)

    try:
        contact_sensor = env.unwrapped.scene["contact_forces"]
        contact_body_names = list(
            getattr(contact_sensor, "body_names", getattr(contact_sensor.data, "body_names", body_names))
        )
    except Exception:
        contact_sensor = None
        contact_body_names = body_names

    def _contact_idx(body_name: str, fallback: int | None) -> int | None:
        if body_name in contact_body_names:
            return contact_body_names.index(body_name)
        if contact_sensor is not None and fallback is not None and contact_sensor.data.net_forces_w.shape[1] == len(
            body_names
        ):
            return fallback
        return None

    left_contact_idx = _contact_idx("left_ankle_link", left_foot_idx)
    right_contact_idx = _contact_idx("right_ankle_link", right_foot_idx)

    print(f"\n{'=' * 72}\n[compare_spawn] asset={label}\n{'=' * 72}", flush=True)

    for step in range(1, max_steps + 1):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        if step not in log_steps:
            continue

        data = robot.data
        root_pos = data.root_pos_w[0]
        vel_b = data.root_lin_vel_b[0]
        roll, pitch, yaw = euler_xyz_from_quat(data.root_quat_w[0:1])
        roll = _wrap_to_pi(roll)[0].item()
        pitch = _wrap_to_pi(pitch)[0].item()
        yaw = _wrap_to_pi(yaw)[0].item()
        base_z = root_pos[2].item()

        def _foot_z(idx: int | None) -> float:
            if idx is None:
                return float("nan")
            return data.body_pos_w[0, idx, 2].item()

        def _foot_contact(idx: int | None) -> bool:
            if contact_sensor is None or idx is None:
                return False
            return torch.norm(contact_sensor.data.net_forces_w[0, idx]).item() > 5.0

        left_z = _foot_z(left_foot_idx)
        right_z = _foot_z(right_foot_idx)
        fallen = base_z < 0.55 or abs(roll) > 0.8 or abs(pitch) > 0.8
        print(
            f"[spawn] step={step:4d} base_z={base_z:.3f} roll={roll:+.3f} pitch={pitch:+.3f} yaw={yaw:+.3f} "
            f"vx={vel_b[0].item():+.3f} left_z={left_z:.3f} right_z={right_z:.3f} "
            f"L_contact={_foot_contact(left_contact_idx)} R_contact={_foot_contact(right_contact_idx)} "
            f"fallen={fallen}",
            flush=True,
        )

    env.close()


def main() -> None:
    log_steps = {int(s.strip()) for s in args_cli.log_steps.split(",") if s.strip()}
    label, cfg = ASSET_CHOICES[args_cli.asset]
    _run_asset(args_cli.asset, label, cfg, log_steps, args_cli.steps)
    simulation_app.close()


if __name__ == "__main__":
    main()
