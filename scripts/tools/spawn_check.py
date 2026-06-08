# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Inspect robot spawn pose: Isaac starts once, Enter triggers env.reset()."""

"""Launch Isaac Sim Simulator first."""

import argparse
import select
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Reset loop to inspect spawn pose without restarting Isaac Sim."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-My_H1",
    help="Name of the task.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=0,
    help="Zero-action steps after each reset to watch fall under gravity (0 = spawn only).",
)
parser.add_argument(
    "--auto_reset_s",
    type=float,
    default=3.0,
    help="Auto reset every N seconds (keeps UI alive). Use 0 to wait for Enter in terminal.",
)
parser.add_argument(
    "--max_resets",
    type=int,
    default=0,
    help="Stop after N reset cycles (0 = infinite). Use 1 to inspect spawn once if 2nd reset crashes Sim.",
)
parser.add_argument(
    "--spawn_pause_s",
    type=float,
    default=0.0,
    help="After each reset, keep UI live this many seconds BEFORE zero-action steps (time to press F / Follow).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def pump_simulation(frames: int = 1) -> None:
    """Keep Isaac viewport responsive (do not block on input() without this)."""
    for _ in range(frames):
        if not simulation_app.is_running():
            return
        simulation_app.update()


def wait_for_enter() -> bool:
    """Wait for Enter while pumping the sim app. Returns False if sim stopped."""
    if not sys.stdin.isatty():
        print("[WARN] stdin is not a TTY; use --auto_reset_s instead of Enter.")
        time.sleep(1.0)
        return simulation_app.is_running()

    print("[INFO] Press Enter = reset again | Ctrl+C once = quit")
    while simulation_app.is_running():
        pump_simulation(1)
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            sys.stdin.readline()
            return True
    return False


def wait_auto_reset(seconds: float) -> None:
    """Sleep in small slices so the UI keeps updating."""
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        pump_simulation(1)
        time.sleep(0.02)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    print("[INFO] Isaac is up. Each reset respawns the robot (no full app restart).")
    print("[INFO] Edit my_h1.py -> pip install -e source/robot_lab -> Ctrl+C -> rerun this script.")
    if args_cli.settle_steps > 0:
        print(f"[INFO] After reset: {args_cli.settle_steps} zero-action steps (~landing).")
    else:
        print("[INFO] settle_steps=0: look at spawn right after reset.")
    if args_cli.auto_reset_s > 0:
        print(f"[INFO] Auto reset every {args_cli.auto_reset_s}s (--auto_reset_s 0 for Enter mode).")
    else:
        print("[INFO] Enter mode: UI stays live while waiting (do not use Isaac Pause).")
    if args_cli.max_resets > 0:
        print(f"[INFO] Will stop after {args_cli.max_resets} reset cycle(s) (--max_resets 0 for infinite).")
    if args_cli.spawn_pause_s > 0:
        print(
            f"[INFO] After reset: wait {args_cli.spawn_pause_s}s (viewport live) "
            f"then {args_cli.settle_steps} zero-action steps."
        )

    reset_count = 0
    try:
        while simulation_app.is_running():
            env.reset()
            reset_count += 1
            print(f"[INFO] env.reset() done. (cycle {reset_count})")
            pump_simulation(4)

            if args_cli.spawn_pause_s > 0:
                print(
                    f"[INFO] spawn_pause: {args_cli.spawn_pause_s}s — select pelvis, press F, Follow Mode, then watch."
                )
                wait_auto_reset(args_cli.spawn_pause_s)

            for _ in range(args_cli.settle_steps):
                with torch.inference_mode():
                    env.step(actions)

            if args_cli.max_resets > 0 and reset_count >= args_cli.max_resets:
                print(
                    f"[INFO] Reached --max_resets {args_cli.max_resets}. "
                    "Inspect the viewport; Ctrl+C to quit (Sim stays up until you exit)."
                )
                while simulation_app.is_running():
                    pump_simulation(1)
                    time.sleep(0.02)
                break

            if args_cli.auto_reset_s > 0:
                wait_auto_reset(args_cli.auto_reset_s)
            else:
                if not wait_for_enter():
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
