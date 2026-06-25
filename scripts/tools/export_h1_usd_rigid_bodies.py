# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Export USD H1 minimal rigid-body props for URDF alignment."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/tmp/h1_usd_rigid_bodies.txt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from h1_asset_compare_utils import ASSET_CHOICES, patch_env_cfg_for_asset

_, cfg = ASSET_CHOICES["usd"]
env_cfg = parse_env_cfg("RobotLab-Isaac-Velocity-Flat-Unitree-My_H1", device="cuda:0", num_envs=1)
env_cfg.scene.robot = cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
patch_env_cfg_for_asset(env_cfg, "usd")
env = gym.make("RobotLab-Isaac-Velocity-Flat-Unitree-My_H1", cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
masses = robot.root_physx_view.get_masses()[0]
coms = robot.root_physx_view.get_coms()[0]
inertias = robot.root_physx_view.get_inertias()[0]

lines = []
for i, name in enumerate(robot.data.body_names):
    com = coms[i, :3].tolist()
    t = inertias[i].tolist()
    ixx, ixy, ixz, iyy, iyz, izz = t[0], t[1], t[2], t[4], t[5], t[8]
    lines.append(
        f"{name}\tm={masses[i].item():.6f}\tcom={com[0]:.8f},{com[1]:.8f},{com[2]:.8f}\t"
        f"I={ixx:.8e},{ixy:.8e},{ixz:.8e},{iyy:.8e},{iyz:.8e},{izz:.8e}"
    )

with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} bodies to {args.out}")

env.close()
simulation_app.close()
