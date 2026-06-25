# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# Run 16: official H1 USD (isolate URDF vs USD); set False to restore My_H1 URDF.
_USE_OFFICIAL_H1_USD = False

if _USE_OFFICIAL_H1_USD:
    from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG  # isort: skip
else:
    from robot_lab.assets.my_h1 import UNITREE_My_H1_CFG  # isort: skip


@configclass
class UnitreeMyH1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """My_H1 Rough: rewards + MDP base aligned with Isaac Lab ``H1RoughEnvCfg`` (Flat-10)."""

    base_link_name = "torso_link"
    foot_link_name = ".*ankle_link" if _USE_OFFICIAL_H1_USD else ".*_ankle_link"

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        robot_cfg = H1_MINIMAL_CFG if _USE_OFFICIAL_H1_USD else UNITREE_My_H1_CFG
        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # Isaac base terrain: no restitution bounce
        self.scene.terrain.physics_material.restitution = 0.0

        # ------------------------------Actions (Isaac default scale=0.5)------------------------------
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.clip = None

        # ------------------------------Events (Isaac H1RoughEnvCfg)------------------------------
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)

        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0.0, 0.0)

        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ------------------------------Rewards (Isaac Lab H1RoughEnvCfg / H1Rewards)------------------------------
        self.rewards.is_terminated.weight = -200.0

        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = 0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        if _USE_OFFICIAL_H1_USD:
            self.rewards.create_joint_deviation_l1_rewterm(
                "joint_deviation_hip_l1", -0.2, [".*_hip_yaw", ".*_hip_roll"]
            )
            self.rewards.create_joint_deviation_l1_rewterm(
                "joint_deviation_arms_l1", -0.2, [".*_shoulder_.*", ".*_elbow"]
            )
            self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["torso"])
            ankle_joint_names = [".*_ankle.*"]
        else:
            self.rewards.create_joint_deviation_l1_rewterm(
                "joint_deviation_hip_l1", -0.2, [".*hip_yaw.*", ".*hip_roll.*"]
            )
            self.rewards.create_joint_deviation_l1_rewterm(
                "joint_deviation_arms_l1", -0.2, [".*shoulder.*", ".*elbow.*"]
            )
            self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["torso_joint"])
            ankle_joint_names = [".*_ankle_.*"]
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = ankle_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = 0
        self.rewards.joint_mirror.weight = 0

        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_mirror.weight = 0

        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.25
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_air_time_variance.weight = 0
        self.rewards.upward.weight = 0

        if self.__class__.__name__ == "UnitreeMyH1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands (Isaac UniformVelocityCommand + H1 ranges)------------------------------
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
