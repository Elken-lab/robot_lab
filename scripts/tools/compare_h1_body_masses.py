# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""List per-link mass USD vs URDF side by side."""

from __future__ import annotations

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Unitree-My_H1")
parser.add_argument("--_child_run", action="store_true", default=False)
parser.add_argument("--asset", type=str, default="usd")
args_cli, _ = parser.parse_known_args()

if not args_cli._child_run:
    rows = {}
    for asset in ("usd", "urdf"):
        child = [sys.executable, __file__, "--_child_run", "--asset", asset, "--task", args_cli.task]
        for arg in sys.argv[1:]:
            if arg in ("--headless",) or arg.startswith("--device"):
                child.append(arg)
            if arg == "--device" and sys.argv[sys.argv.index(arg)+1:]:
                pass
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == "--headless":
                child.append("--headless")
            if sys.argv[i] == "--device" and i + 1 < len(sys.argv):
                child.extend(["--device", sys.argv[i + 1]])
            i += 1
        if "--headless" not in child:
            child.append("--headless")
        out = subprocess.check_output(child, text=True)
        for line in out.splitlines():
            if line.startswith("BODY "):
                _, name, mass = line.split()
                rows.setdefault(name, {})[asset] = float(mass)
    print(f"{'body':<28} {'usd':>8} {'urdf':>8} {'delta':>8}")
    for name in sorted(rows):
        u = rows[name].get("usd", float("nan"))
        r = rows[name].get("urdf", float("nan"))
        d = r - u if u == u and r == r else float("nan")
        flag = " ***" if abs(d) > 0.05 else ""
        print(f"{name:<28} {u:8.3f} {r:8.3f} {d:+8.3f}{flag}")
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

label, robot_cfg = ASSET_CHOICES[args_cli.asset]
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
patch_env_cfg_for_asset(env_cfg, args_cli.asset)
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()
robot = env.unwrapped.scene["robot"]
masses = robot.root_physx_view.get_masses()[0]
for i, name in enumerate(robot.data.body_names):
    print(f"BODY {name} {masses[i].item():.6f}", flush=True)
env.close()
simulation_app.close()
