# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeMyH1RoughEnvCfg


@configclass
class UnitreeMyH1FlatEnvCfg(UnitreeMyH1RoughEnvCfg):
    """My_H1 Flat: plane terrain + Isaac ``H1FlatEnvCfg`` gait overrides (Flat-10 base)."""

    def __post_init__(self):
        super().__post_init__()

        # --- terrain: plane (match Isaac H1FlatEnvCfg) ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # --- Isaac Flat: only bump feet_air vs rough (1.0 / 0.6) ---
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6

        if self.__class__.__name__ == "UnitreeMyH1FlatEnvCfg":
            self.disable_zero_weight_rewards()
