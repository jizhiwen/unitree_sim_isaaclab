import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an Realman robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--denoising_steps", type=int, default=10, help="The number of denoising steps to use.")
parser.add_argument(
    "--data_config",
    type=str,
    help="Model modality.",
    choices=["realman", "realman_side_wrist", "realman_front_wrist", "realman_left_right_wrist"],
    required=True,
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab_blueprint.tasks  # noqa F401
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_blueprint.tasks.manager_based.pick_place.config.realman.joint_pos_inference_env_cfg import (
    RealmanCubePickPlaceEnvCfg,
)


def ExtractObs(obs, modality):
    gripper1 = obs["policy"]["joint_pos_abs"].cpu().numpy()[0, 6]
    gripper2 = obs["policy"]["joint_pos_abs"].cpu().numpy()[0, 7]
    obs = None
    match modality:
        case "realman":
            obs = {
                "state.single_arm": np.degrees(obs["policy"]["joint_pos_abs"].cpu().numpy()[:, :6]),
                "state.gripper": np.array([[1000 - (gripper1 + gripper2) / 0.08 * 1000]]),
                "video.front_view": obs["rgb_camera"]["table_high_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.side_view": obs["rgb_camera"]["table_side_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.wrist_view": obs["rgb_camera"]["table_cam_rgb"].cpu().numpy().astype(np.uint8),
                "annotation.human.action.task_description": "Pick up the blue cube and place it in the box.",
            }
        case "realman_side_wrist":
            obs = {
                "state.single_arm": np.degrees(obs["policy"]["joint_pos_abs"].cpu().numpy()[:, :6]),
                "state.gripper": np.array([[1000 - (gripper1 + gripper2) / 0.08 * 1000]]),
                "video.side_view": obs["rgb_camera"]["table_side_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.wrist_view": obs["rgb_camera"]["table_cam_rgb"].cpu().numpy().astype(np.uint8),
                "annotation.human.action.task_description": "Pick up the blue cube and place it in the box.",
            }
        case "realman_front_wrist":
            obs = {
                "state.single_arm": np.degrees(obs["policy"]["joint_pos_abs"].cpu().numpy()[:, :6]),
                "state.gripper": np.array([[1000 - (gripper1 + gripper2) / 0.08 * 1000]]),
                "video.front_view": obs["rgb_camera"]["table_high_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.wrist_view": obs["rgb_camera"]["table_cam_rgb"].cpu().numpy().astype(np.uint8),
                "annotation.human.action.task_description": "Pick up the blue cube and place it in the box.",
            }
        case "realman_left_right_wrist":
            obs = {
                "state.single_arm": np.degrees(obs["policy"]["joint_pos_abs"].cpu().numpy()[:, :6]),
                "state.gripper": np.array([[1000 - (gripper1 + gripper2) / 0.08 * 1000]]),
                "video.left_view": obs["rgb_camera"]["left_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.right_view": obs["rgb_camera"]["right_cam_rgb"].cpu().numpy().astype(np.uint8),
                "video.wrist_view": obs["rgb_camera"]["table_cam_rgb"].cpu().numpy().astype(np.uint8),
                "annotation.human.action.task_description": "Pick up the blue cube and place it in the box.",
            }
        case _:
            raise ValueError(f"Unknown modality: '{modality}'")

    return obs


def main():
    """Main function."""
    # setup environment
    env_cfg = RealmanCubePickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    timeout_term = None
    if hasattr(env_cfg.terminations, "time_out"):
        timeout_term = env_cfg.terminations.time_out
        env_cfg.terminations.time_out = None

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create a policy
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[args_cli.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args_cli.checkpoint,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        denoising_steps=args_cli.denoising_steps,
    )

    success_step_count = 0
    total_step_count = 0

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            gripper1 = obs["policy"]["joint_pos_abs"].cpu().numpy()[0, 6]
            gripper2 = obs["policy"]["joint_pos_abs"].cpu().numpy()[0, 7]
            realman_obs = ExtractObs(obs, args_cli.data_config)
            action_chunk = policy.get_action(realman_obs)
            single_arm = action_chunk["action.single_arm"]
            gripper = action_chunk["action.gripper"]

            for i in range(0, len(single_arm)):
                action = np.append(np.deg2rad(single_arm[i]), -1 if gripper[i] < 100 else 1)
                action = torch.tensor(action, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
                obs, _, _, _, _ = env.step(action)

                if success_term is not None:
                    if bool(success_term.func(env, **success_term.params)[0]):
                        total_step_count = total_step_count + 1
                        success_step_count = success_step_count + 1
                        print(f"[{success_step_count}/{total_step_count}] Task success ...")
                        obs, _ = env.reset()
                        break

                if timeout_term is not None:
                    if bool(timeout_term.func(env, **timeout_term.params)[0]):
                        total_step_count = total_step_count + 1
                        print(f"[{success_step_count}/{total_step_count}] Task fail ...")
                        obs, _ = env.reset()
                        break

                if 1000 - (gripper1 + gripper2) / 0.08 * 1000 < 10:
                    total_step_count = total_step_count + 1
                    print(f"[{success_step_count}/{total_step_count}] Task fail ...")
                    obs, _ = env.reset()
                    break


if __name__ == "__main__":
    main()
    simulation_app.close()
