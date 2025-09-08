# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
@configclass
class TableCylinderSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # 房间中心点
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"/home/app/nsb/asset/Simple_Warehouse/nsb_warehouse.usd",  # use simple room model
        ),
    )
    # print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
    #ISAAC_NUCLEUS_DIR: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac
    # 1. table configuration
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.2],   # initial position [x, y, z]
                                                rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",    # table model file
            usd_path=f"/home/app/nsb/asset/PackingTable/PackingTable.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    #Glue stick
    # stick = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Stick",    # table in the scene
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.15, 0.35, 0.8],   # initial position [x, y, z]
    #                                             rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
    #     spawn=UsdFileCfg(
    #         usd_path=f"/home/app/nsb/asset/jiaobang-saomiao/jiaobang-saomiao.usd",    # table model file
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         ),    # rigid body properties configuration (rigid_props)
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),    # mass properties configuration (mass)
    #         collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
    #     ),
    # )

    # stick = RigidObjectCfg(
    #         prim_path="/World/envs/env_.*/Stick",    # table in the scene
    #         init_state=AssetBaseCfg.InitialStateCfg(pos=[0.15, 0.35, 0.8],   # initial position [x, y, z]
    #                                             rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
    #         spawn=UsdFileCfg(
    #             usd_path=f"/home/app/nsb/asset/jiaobang-saomiao/jiaobang-saomiao.usd",
    #             scale=(1.0, 1.0, 1.0),
    #             rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         solver_position_iteration_count=16,
    #                         solver_velocity_iteration_count=1,
    #                         max_angular_velocity=1000.0,
    #                         max_linear_velocity=1000.0,
    #                         max_depenetration_velocity=5.0,
    #                         disable_gravity=False,
    #                     )
    #             ,
    #         ),
    #     )

    #cylinder
    #2. object configuration (cylinder)     
    # object = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",    # object in the scene
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.40, 0.24], # initial position (pos) 
    #                                               rot=[1, 0, 0, 0]), # initial rotation (rot)
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.018,    # cylinder radius (radius)
    #         height=0.25,     # cylinder height (height) 
 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         ),    # rigid body properties configuration (rigid_props)
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),    # mass properties configuration (mass)
    #         collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="max",    # friction combine mode
    #             restitution_combine_mode="min",    # restitution combine mode
    #             static_friction=2,    # static friction coefficient
    #             dynamic_friction=2,    # dynamic friction coefficient
    #             restitution=0.0,    # restitution coefficient (no restitution)
    #         ),
    #     ),
    # )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",    # object in the scene
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.12, 0.30, 0.84], # initial position (pos) 
                                                  rot=[1, 0, 0, 0]), # initial rotation (rot)
        spawn=sim_utils.CylinderCfg(
            radius=0.018,    # cylinder radius (radius)
            height=0.25,     # cylinder height (height) 
 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),    # rigid body properties configuration (rigid_props)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),    # mass properties configuration (mass)
            collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",    # friction combine mode
                restitution_combine_mode="min",    # restitution combine mode
                static_friction=2,    # static friction coefficient
                dynamic_friction=2,    # dynamic friction coefficient
                restitution=0.0,    # restitution coefficient (no restitution)
            ),
        ),
    )



    # Ground plane
    # 3. ground configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",    # ground in the scene
    #     spawn=GroundPlaneCfg( ),    # ground configuration
    # )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=( -0.00617,0.00617, 0.70708, -0.70708),
                                                    focal_length = 16.5)