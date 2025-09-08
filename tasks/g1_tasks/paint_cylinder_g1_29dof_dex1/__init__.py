
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import paint_cylinder_g1_29dof_dex1_joint_env_cfg
from . import paint_cylinder_g1_29dof_dex1_ik_abs_env_cfg

gym.register(
    id="Nsb-Paint-Cylinder-G129-Dex1-Ik-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_cylinder_g1_29dof_dex1_ik_abs_env_cfg.PaintG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Nsb-Paint-Cylinder-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_cylinder_g1_29dof_dex1_joint_env_cfg.PaintG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)

