# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Print rigid-body mass / COM / inertia for H1 USD vs URDF (Route B P2)."""

from __future__ import annotations

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Dump H1 rigid-body props.")
parser.add_argument("--asset", type=str, choices=["usd", "urdf", "both"], default="both")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Unitree-My_H1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--_child_run", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()

if args_cli.asset == "both" and not args_cli._child_run:
    child_cmd = [
        sys.executable,
        __file__,
        "--task",
        args_cli.task,
        "--num_envs",
        str(args_cli.num_envs),
        "--_child_run",
    ]
    if args_cli.disable_fabric:
        child_cmd.append("--disable_fabric")
    known = {"--headless", "--livestream", "--enable_cameras", "--device", "--experience", "--kit_args"}
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
        print(f"\n>>> rigid_body run: --asset {asset}", flush=True)
        subprocess.run(child_cmd + ["--asset", asset], check=True)
    sys.exit(0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from h1_asset_compare_utils import ASSET_CHOICES, patch_env_cfg_for_asset

asset_key = args_cli.asset
label, robot_cfg = ASSET_CHOICES[asset_key]
env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=not args_cli.disable_fabric,
)
env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
patch_env_cfg_for_asset(env_cfg, asset_key)
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
body_names = list(robot.data.body_names)
view = robot.root_physx_view
masses = view.get_masses()[0]
coms = view.get_coms()[0]
inertias = view.get_inertias()[0]

targets = ("torso_link", "pelvis", "left_ankle_link", "right_ankle_link")
print(f"\n{'=' * 72}\n[rigid_body] asset={label}\n{'=' * 72}", flush=True)
print(f"  total_mass={masses.sum().item():.4f}", flush=True)
for name in targets:
    idx = body_names.index(name)
    com = coms[idx].tolist()
    inertia = inertias[idx].tolist()
    print(
        f"  {name}: mass={masses[idx].item():.4f} com={com[:3]} "
        f"inertia_diag=({inertia[0]:.4f}, {inertia[1]:.4f}, {inertia[2]:.4f})",
        flush=True,
    )

env.close()
simulation_app.close()
