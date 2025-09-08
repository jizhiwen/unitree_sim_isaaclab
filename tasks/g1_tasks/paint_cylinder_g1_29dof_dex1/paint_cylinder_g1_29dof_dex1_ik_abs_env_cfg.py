# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import torch
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from . import paint_cylinder_g1_29dof_dex1_joint_env_cfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.realman import REALMAN_HIGH_PD_CFG # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object = ObsTerm(func=mdp.object_obs)
        # cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        # cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)

        joint_pos_abs = ObsTerm(func=mdp.joint_pos)
        joint_vel_abs = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        front_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/front_camera",
            },
        )
        left_wrist_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/left_wrist_camera",
            },
        )
        right_wrist_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/right_wrist_camera",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()

##
# MDP settings
##
@configclass
class ActionsCfg:
    # Set actions for the specific robot type
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ],
        body_name="right_hand_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_hand_Joint1_1",
            "right_hand_Joint2_1",
        ],
        open_command_expr={
            "right_hand_Joint1_1": 0.00,
            "right_hand_Joint2_1": 0.00,
        },
        close_command_expr={
            "right_hand_Joint1_1": 0.02,
            "right_hand_Joint2_1": 0.02,
        },
    )

@configclass
class PaintG129DEX1BaseFixEnvCfg(paint_cylinder_g1_29dof_dex1_joint_env_cfg.PaintG129DEX1BaseFixEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions: ActionsCfg = ActionsCfg()                  # action configuration

        MAPPING = {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (125, 125, 125, 255),
            "class:UNLABELLED": (10, 10, 10, 255),
            "class:BACKGROUND": (10, 10, 10, 255),
        }
