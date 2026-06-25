# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--follow", action="store_true", default=False, help="Auto-follow robot with viewport camera each step."
)
parser.add_argument(
    "--hold_seconds",
    type=float,
    default=0.0,
    help="Pause policy at spawn for N seconds so you can frame the viewport before stepping.",
)
parser.add_argument(
    "--plane",
    action="store_true",
    default=False,
    help="Override terrain to plane for play (same task/cfg, flat ground only).",
)
parser.add_argument(
    "--forward",
    action="store_true",
    default=False,
    help="Pin a constant forward velocity command (disables standing/heading resampling).",
)
parser.add_argument(
    "--forward_vel",
    type=float,
    default=0.3,
    help="Forward speed (m/s) when --forward is set.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    if args_cli.plane:
        print("[INFO] Play terrain override: plane (flat ground).")
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
    elif env_cfg.scene.terrain.terrain_generator is not None:
        # reduce the number of terrains to save memory
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.forward and args_cli.keyboard:
        raise ValueError("Use either --forward or --keyboard, not both.")

    if args_cli.forward:
        forward_vel = args_cli.forward_vel
        print(f"[INFO] Play command override: constant forward vx={forward_vel:.2f} m/s.")
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (forward_vel, forward_vel)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.debug_vis = True

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    obs, _ = env.reset()
    play_step = 0
    init_pos_x = env.unwrapped.scene["robot"].data.root_pos_w[0, 0].item()
    timestep = 0

    if args_cli.hold_seconds > 0.0:
        print(
            f"[INFO] Holding {args_cli.hold_seconds:.1f}s at spawn (policy paused). "
            "Frame the viewport now; use --follow to auto-track the robot."
        )
        hold_until = time.time() + args_cli.hold_seconds
        while simulation_app.is_running() and time.time() < hold_until:
            env.unwrapped.sim.render()
            if args_cli.follow or args_cli.keyboard:
                camera_follow(env)
            time.sleep(0.02)

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

    def _contact_body_index(body_name: str, fallback_idx: int | None) -> int | None:
        if body_name in contact_body_names:
            return contact_body_names.index(body_name)
        if contact_sensor is not None and fallback_idx is not None:
            forces = contact_sensor.data.net_forces_w
            if forces.shape[1] == len(body_names):
                return fallback_idx
        return None

    left_contact_idx = _contact_body_index("left_ankle_link", left_foot_idx)
    right_contact_idx = _contact_body_index("right_ankle_link", right_foot_idx)

    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        """Normalize Isaac Lab Euler angles from [0, 2*pi] to [-pi, pi]."""
        return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi

    def _foot_z(body_idx: int | None) -> float:
        if body_idx is None:
            return float("nan")
        return robot.data.body_pos_w[0, body_idx, 2].item()

    def _foot_contact(contact_idx: int | None) -> bool:
        if contact_sensor is None or contact_idx is None:
            return False
        force_norm = torch.norm(contact_sensor.data.net_forces_w[0, contact_idx]).item()
        return force_norm > 5.0

    def _pin_forward_command() -> None:
        if not args_cli.forward:
            return
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        forward = torch.tensor(
            [args_cli.forward_vel, 0.0, 0.0], device=env.unwrapped.device, dtype=torch.float32
        )
        cmd_term.is_standing_env[:] = False
        if hasattr(cmd_term, "is_heading_env"):
            cmd_term.is_heading_env[:] = False
        cmd_term.vel_command_b[:] = forward.unsqueeze(0).expand(env.unwrapped.num_envs, -1)

    _pin_forward_command()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            _pin_forward_command()
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if args_cli.forward and bool(dones[0].item() if torch.is_tensor(dones) else dones[0]):
                _pin_forward_command()
            play_step += 1
            if play_step == 1 or play_step % 100 == 0:
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0]
                robot_data = env.unwrapped.scene["robot"].data
                vel_b = robot_data.root_lin_vel_b[0]
                root_pos = robot_data.root_pos_w[0]
                pos_x = root_pos[0].item()
                base_z = root_pos[2].item()
                roll, pitch, yaw = euler_xyz_from_quat(robot_data.root_quat_w[0:1])
                roll = _wrap_to_pi(roll)[0].item()
                pitch = _wrap_to_pi(pitch)[0].item()
                yaw = _wrap_to_pi(yaw)[0].item()
                done = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
                fallen = base_z < 0.55 or abs(roll) > 0.8 or abs(pitch) > 0.8
                left_foot_z = _foot_z(left_foot_idx)
                right_foot_z = _foot_z(right_foot_idx)
                left_contact = _foot_contact(left_contact_idx)
                right_contact = _foot_contact(right_contact_idx)
                print(
                    f"[play debug] step={play_step} cmd={cmd.cpu().tolist()} "
                    f"vx={vel_b[0].item():.3f} pos_x={pos_x:.3f} (Δx={pos_x - init_pos_x:.3f}) "
                    f"base_z={base_z:.3f} roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f} "
                    f"left_foot_z={left_foot_z:.3f} right_foot_z={right_foot_z:.3f} "
                    f"left_contact={left_contact} right_contact={right_contact} "
                    f"fallen={fallen} done={done} |action|={actions[0].abs().mean().item():.4f}"
                )
            if play_step >= 500:
                break
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.follow or args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
