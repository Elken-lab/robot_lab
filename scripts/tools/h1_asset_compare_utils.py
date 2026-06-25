# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Shared USD/URDF env patching for H1 Route B compare scripts."""

from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG

from robot_lab.assets.my_h1 import UNITREE_My_H1_CFG

ASSET_CHOICES = {
    "usd": ("H1_MINIMAL_CFG (USD)", H1_MINIMAL_CFG),
    "urdf": ("UNITREE_My_H1_CFG (h1_minimal.urdf)", UNITREE_My_H1_CFG),
}


def patch_env_cfg_for_asset(env_cfg, asset_key: str) -> None:
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
