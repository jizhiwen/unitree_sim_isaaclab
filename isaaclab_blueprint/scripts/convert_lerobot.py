import argparse
import copy
import json
import os
import shutil
import warnings
from typing import Any

import h5py
import imageio
import jsonlines
import numpy as np
import pandas as pd
import warp as wp
from PIL import Image
from tqdm.auto import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument(
    "--episode_num",
    type=int,
    default=99999999999999999,
    help="Number of episode to be exported. Default is all.",
)

parser.add_argument(
    "--dataset_file", type=str, default=None, help="File name of the dataset to be converted.", required=True
)
parser.add_argument("--image_path", type=str, default=None, help="Path of the images to be converted.", required=True)
parser.add_argument(
    "--output_path", type=str, default=None, help="Path of the converted lerobot dataset.", required=True
)
parser.add_argument("--task_description", type=str, default=None, help="Description of the task.", required=True)
parser.add_argument(
    "--type",
    type=str,
    default="rgb",
    help="Description of the task.",
    nargs="+",
    required=False,
    choices=["rgb", "semantic_segmentation", "depth"],
)

# parse the arguments
args_cli = parser.parse_args()

VIDEO_KEY = "front_view"
VIDEO_KEY2 = "side_view"
VIDEO_KEY3 = "wrist_view"
VIDEO_KEY4 = "left_view"
VIDEO_KEY5 = "right_view"
ANNOTATION_KEY_TO_TASK_INDEX = {}
COMPUTE_STATS = True
SPLITS = {"train": "0:1"}
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
USE_AV1 = True

# --- Modality Definitions (CRITICAL - Match your actual data) ---
# State modalities for joints are now discovered automatically from STATE_TOPIC.
# You can still define OTHER state modalities here (e.g., for TF poses).
STATE_MODALITIES = {
    # Example: If you ALSO want End-Effector Pose from TF (in addition to discovered joints)
    # "ee_pose": {"start": <auto_detected_joint_dim>, "end": <auto_detected_joint_dim> + 7}
    # The start/end indices for non-joint states would need careful management
    # relative to the discovered joint dimension. For simplicity, focusing only on joints now.
}

# Update ACTION definition (if using actions)
ACTION_MODALITIES = {
    # Example: 6-DoF arm + 1-DoF gripper command
    # "arm_command": {"start": 0, "end": 5},    # Example
    # "gripper_command": {"start": 5, "end": 6} # Example
}

ANNOTATION_MODALITIES = {
    "human.action.task_description": {"original_key": "task_index"},
}

CODEBASE_VERSION = "v2.1"
# A descriptive name for your dataset
DATASET_NAME = "realman"  # Used in info.json
# Specify the robot type (e.g., 'so100', 'franka', 'ur5')
ROBOT_TYPE = "RM65"

# Video and rendering settings
DEFAULT_FRAMERATE = 24.0
DEFAULT_LIGHT_DIRECTION = (0.0, 0.0, 1.0)  # Points straight down at surface


@wp.kernel
def _shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply shading to semantic segmentation using surface normals.

    Args:
        segmentation: Input semantic segmentation image (H,W,C)
        normals: Surface normal vectors (H,W,3)
        shading_out: Output shaded segmentation image (H,W,C)
        light_source: Position of light source
    """
    i, j = wp.tid()
    normal = normals[i, j]
    light_source_vec = wp.normalize(light_source[0])
    shade = 0.5 + wp.dot(wp.vec3f(normal[0], normal[1], normal[2]), light_source_vec) * 0.5

    shading_out[i, j, 0] = wp.uint8(wp.float32(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float32(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float32(segmentation[i, j, 2]) * shade)
    shading_out[i, j, 3] = wp.uint8(255)


def encode_video(
    root_dir: str, start_frame: int, num_frames: int, camera_name: str, output_path: str, env_num: int, trial_num: int
) -> None:
    """Encode a sequence of shaded segmentation frames into a video.

    Args:
        root_dir: Directory containing the input frames
        start_frame: Starting frame index
        num_frames: Number of frames to encode
        camera_name: Name of the camera (used in filename pattern)
        output_path: Output path for the encoded video
        env_num: Environment number for the sequence
        trial_num: Trial number for the sequence

    Raises:
        ValueError: If start_frame is negative or if any required frame is missing
    """
    from video_encoding import get_video_encoding_interface

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    frame_name_pattern = "{camera_name}_{modality}_trial_{trial_num}_tile_{env_num}_step_{frame_idx}.png"

    # Validate all frames exist before starting
    for frame_idx in range(start_frame, start_frame + num_frames):
        file_path_normals = os.path.join(
            root_dir,
            frame_name_pattern.format(
                camera_name=camera_name, modality="normals", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx
            ),
        )
        file_path_segmentation = os.path.join(
            root_dir,
            frame_name_pattern.format(
                camera_name=camera_name,
                modality="semantic_segmentation",
                trial_num=trial_num,
                env_num=env_num,
                frame_idx=frame_idx,
            ),
        )
        if not os.path.exists(file_path_normals) or not os.path.exists(file_path_segmentation):
            raise ValueError(f"Missing frame at frame index {frame_idx} for trial {trial_num}")

    # Initialize video encoding
    video_encoding = get_video_encoding_interface()

    # Get dimensions from first frame
    first_frame = np.array(
        Image.open(
            os.path.join(
                root_dir,
                frame_name_pattern.format(
                    camera_name=camera_name,
                    modality="semantic_segmentation",
                    trial_num=trial_num,
                    env_num=env_num,
                    frame_idx=start_frame,
                ),
            )
        )
    )
    height, width = first_frame.shape[:2]

    # Pre-allocate buffers
    normals_wp = wp.empty((height, width, 3), dtype=wp.float32, device="cuda")
    segmentation_wp = wp.empty((height, width, 4), dtype=wp.uint8, device="cuda")
    shaded_segmentation_wp = wp.empty_like(segmentation_wp)
    light_source = wp.array(DEFAULT_LIGHT_DIRECTION, dtype=wp.vec3f, device="cuda")

    video_encoding.start_encoding(
        video_filename=output_path,
        framerate=DEFAULT_FRAMERATE,
        nframes=num_frames,
        overwrite_video=True,
    )

    for frame_idx in range(start_frame, start_frame + num_frames):
        file_path_normals = os.path.join(
            root_dir,
            frame_name_pattern.format(
                camera_name=camera_name, modality="normals", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx
            ),
        )
        file_path_segmentation = os.path.join(
            root_dir,
            frame_name_pattern.format(
                camera_name=camera_name,
                modality="semantic_segmentation",
                trial_num=trial_num,
                env_num=env_num,
                frame_idx=frame_idx,
            ),
        )

        # Load and copy data to existing buffers
        normals_np = np.array(Image.open(file_path_normals)).astype(np.float32) / 255.0
        wp.copy(normals_wp, wp.from_numpy(normals_np))

        segmentation_np = np.array(Image.open(file_path_segmentation))
        wp.copy(segmentation_wp, wp.from_numpy(segmentation_np))

        # Launch kernel
        wp.launch(
            _shade_segmentation,
            dim=(height, width),
            inputs=[segmentation_wp, normals_wp, shaded_segmentation_wp, light_source],
        )

        # Encode frame
        video_encoding.encode_next_frame_from_buffer(
            shaded_segmentation_wp.numpy().tobytes(), width=width, height=height
        )

    video_encoding.finalize_encoding()


class SimpleLogger:
    """A basic logger class."""

    def info(self, msg):
        print(f"INFO: {msg}")

    def warning(self, msg):
        print(f"WARN: {msg}")

    def error(self, msg, exc_info=False):
        print(f"ERROR: {msg}", flush=True)
        if exc_info:
            import traceback

            traceback.print_exc()


logger = SimpleLogger()


def get_env_trial_frames(root_dir: str, camera_name: str, image_type: str = "rgb", min_frames: int = 30) -> dict:
    """Get the last frame number for each trial for each environment in the dataset.

    Args:
        root_dir: Directory containing the frames
        camera_name: Name of the camera used
        min_frames: Minimum number of frames required for a valid trial

    Returns:
        dict: Dictionary mapping trial numbers to (start_frame, end_frame) tuples
    """
    import re

    # Pattern to match trial and frame numbers
    pattern = rf"{camera_name}_{image_type}_trial_(\d+)_tile_(\d+)_step_(\d+).png"

    frames = {}
    for filename in os.listdir(root_dir):
        match = re.match(pattern, filename)
        if match:
            trial_num = int(match.group(1))
            env_num = int(match.group(2))
            frame_num = int(match.group(3))

            frames.setdefault(env_num, {}).setdefault(trial_num, []).append(frame_num)

    valid_trials = {}
    for env_num, trial_nums in sorted(frames.items()):
        for trial_num, frames in sorted(trial_nums.items()):
            # Skip if not enough frames
            if len(frames) < min_frames:
                continue

            # Sort frames and get range
            frames.sort()
            start_frame = frames[0]
            end_frame = frames[-1]

            # Verify frame sequence is continuous
            expected_frames = set(range(start_frame, end_frame + 1))
            actual_frames = set(frames)
            if len(expected_frames - actual_frames) > 0:
                continue

            valid_trials.setdefault(env_num, {}).setdefault(trial_num, (start_frame, end_frame))

    return valid_trials


class DatasetFormatter:
    """Handles formatting extracted data into the LeRobot dataset structure."""

    def __init__(
        self,
        output_dir: str,
        video_key: str,
        state_dim: int,
        state_dim_names: list[str],
        action_dim: int,
        action_dim_names: list[str],
        logger: SimpleLogger = logger,
    ):
        """
        Initializes the DatasetFormatter.

        Args:
            output_dir: The root directory to save the dataset.
            video_key: The key used for the video modality (e.g., 'webcam').
            state_dim: Dimension of the state vector.
            state_dim_names: List of names for each state dimension.
            action_dim: Dimension of the action vector.
            action_dim_names: List of names for each action dimension.
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.video_key = video_key
        self.logger = logger

        # Store dimensions and names passed from processor
        self.state_dim = state_dim
        self.state_dim_names = state_dim_names
        self.action_dim = action_dim
        self.action_dim_names = action_dim_names

        # Define paths
        self.meta_dir = os.path.join(self.output_dir, "meta")
        self.data_dir = os.path.join(self.output_dir, "data", "chunk-000")
        self.video_dir_base = os.path.join(self.output_dir, "videos", "chunk-000")
        self.video_dir_specific = os.path.join(self.video_dir_base, f"observation.images.{self.video_key}")

        # Validate names against dims (optional sanity check)
        if len(self.state_dim_names) != self.state_dim:
            logger.warning(
                f"Provided state_dim_names length ({len(self.state_dim_names)}) != state_dim ({self.state_dim})."
            )
        if len(self.action_dim_names) != self.action_dim:
            logger.warning(
                f"Provided action_dim_names length ({len(self.action_dim_names)}) != action_dim ({self.action_dim})."
            )

        # Prepare directories once
        self._ensure_dir(self.meta_dir)
        self._ensure_dir(self.data_dir)
        self._ensure_dir(self.video_dir_base)
        self._ensure_dir(self.video_dir_specific)

    def _ensure_dir(self, path: str):
        """Creates a directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    def _calculate_stats(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Calculates statistics for numerical columns in the DataFrame."""
        # [Keep implementation as before]
        stats = {}
        self.logger.info(f"Calculating statistics for episode {df['episode_index'].iloc[0]}...")
        cols_to_stat = []
        if "observation.state" in df.columns:
            cols_to_stat.append("observation.state")
        if "action" in df.columns:
            cols_to_stat.append("action")
        if "timestamp" in df.columns:
            cols_to_stat.append("timestamp")
        for col in ["frame_index", "episode_index", "index", "task_index", "reward", "next.reward"]:
            if col in df.columns:
                cols_to_stat.append(col)
        # Include annotation columns if they exist and are numeric
        for col in df.columns:
            if col.startswith("annotation."):
                cols_to_stat.append(col)

        for col_name in tqdm(cols_to_stat, desc="Calculating Stats"):
            try:
                if isinstance(df[col_name].iloc[0], (np.ndarray, list)):
                    data = np.stack(df[col_name].to_numpy()).astype(np.float64)
                    data[np.isinf(data)] = np.nan
                else:
                    data = df[col_name].to_numpy(dtype=np.float64)

                if np.isnan(data).any():
                    self.logger.warning(f"NaN values found in column '{col_name}'. Stats might be affected.")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    stats[col_name] = {
                        "mean": np.nanmean(data, axis=0).tolist(),
                        "std": np.nanstd(data, axis=0).tolist(),
                        "min": np.nanmin(data, axis=0).tolist(),
                        "max": np.nanmax(data, axis=0).tolist(),
                        "q01": np.nanpercentile(data, 1, axis=0).tolist(),
                        "q99": np.nanpercentile(data, 99, axis=0).tolist(),
                    }
            except Exception as e:
                self.logger.error(f"Could not calculate stats for column '{col_name}': {e}")
        self.logger.info("Statistics calculation finished.")
        return stats

    def write_metadata(
        self,
        all_episode_details: list[dict],
        unique_tasks: dict[str, int],
        fps: float,
        video_info: dict,
        last_episode_stats: dict | None = None,
    ):
        """
        Generates and writes all metadata files AFTER processing all bags.

        Args:
            all_episode_details: List of dicts, each {'index': int, 'length': int, 'tasks': List[str]}.
            unique_tasks: Dictionary mapping task description string to task_index.
            fps: Average or representative FPS (using last bag's FPS here).
            video_info: Video dimensions (using last bag's info here, assumed consistent).
            last_episode_stats: Optional stats dictionary from the last processed episode.
        """
        self.logger.info("Generating final metadata files...")

        total_episodes = len(all_episode_details)
        total_frames = sum(ep["length"] for ep in all_episode_details)
        total_tasks = len(unique_tasks)

        # 1. meta/modality.json
        _state_modalities = {}
        if len(STATE_MODALITIES) == 0:
            _state_modalities["single_arm"] = {"start": 0, "end": self.state_dim - 1}
            _state_modalities["gripper"] = {"start": self.state_dim - 1, "end": self.state_dim}
        else:
            _state_modalities = STATE_MODALITIES

        _action_modalities = {}
        if len(ACTION_MODALITIES) == 0:
            _action_modalities["single_arm"] = {"start": 0, "end": self.action_dim - 1}
            _action_modalities["gripper"] = {"start": self.action_dim - 1, "end": self.action_dim}
        else:
            _action_modalities = ACTION_MODALITIES

        modality_config = {
            "state": _state_modalities,
            "action": _action_modalities,
            "video": {
                self.video_key: {"original_key": f"observation.images.{self.video_key}"},
                VIDEO_KEY2: {"original_key": f"observation.images.{VIDEO_KEY2}"},
                VIDEO_KEY4: {"original_key": f"observation.images.{VIDEO_KEY4}"},
                VIDEO_KEY5: {"original_key": f"observation.images.{VIDEO_KEY5}"},
                VIDEO_KEY3: {"original_key": f"observation.images.{VIDEO_KEY3}"},
            },
        }
        if ANNOTATION_MODALITIES:
            modality_config["annotation"] = ANNOTATION_MODALITIES
            self.logger.info("Adding annotation modalities from config.")

        modality_file = os.path.join(self.meta_dir, "modality.json")
        with open(modality_file, "w") as f:
            json.dump(modality_config, f, indent=4)
        self.logger.info(f"Generated {modality_file}")

        # 2. meta/tasks.jsonl
        tasks_file = os.path.join(self.meta_dir, "tasks.jsonl")
        with jsonlines.open(tasks_file, mode="w") as writer:
            # Sort tasks by index for consistent order
            sorted_tasks = sorted(unique_tasks.items(), key=lambda item: item[1])
            for task_desc, task_idx in sorted_tasks:
                writer.write({"task_index": task_idx, "task": task_desc})
        self.logger.info(f"Generated {tasks_file} with {total_tasks} unique tasks.")

        # 3. meta/info.json
        info_config = {
            "codebase_version": CODEBASE_VERSION,
            "dataset_name": DATASET_NAME,
            "robot_type": ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_episodes,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": round(fps, 2),
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "features": {},
        }

        # Use instance dimensions and names
        if self.action_dim > 0:
            info_config["features"]["action"] = {
                "dtype": "float32",
                "shape": [self.action_dim],
                "names": self.action_dim_names,
            }
        if self.state_dim > 0:
            info_config["features"]["observation.state"] = {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": self.state_dim_names,  # Use discovered names
            }
        if video_info["width"] is not None:
            video_feature_key = f"observation.images.{self.video_key}"
            info_config["features"][video_feature_key] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": round(fps, 2),
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            video_feature_key2 = f"observation.images.{VIDEO_KEY2}"
            info_config["features"][video_feature_key2] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": round(fps, 2),
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            video_feature_key4 = f"observation.images.{VIDEO_KEY4}"
            info_config["features"][video_feature_key4] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": round(fps, 2),
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            video_feature_key5 = f"observation.images.{VIDEO_KEY5}"
            info_config["features"][video_feature_key5] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": round(fps, 2),
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            video_feature_key3 = f"observation.images.{VIDEO_KEY3}"
            info_config["features"][video_feature_key3] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": round(fps, 2),
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        base_scalar_features = [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "reward",
            "next.reward",
        ]
        annotation_features = [f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()]
        for col in base_scalar_features + annotation_features:
            dtype = "float32" if col in ["timestamp", "reward", "next.reward"] else "int64"
            info_config["features"][col] = {"dtype": dtype, "shape": [1], "names": None}

        # Add boolean features
        for col in ["done", "next.done"]:
            info_config["features"][col] = {"dtype": "bool", "shape": [1], "names": None}

        info_file = os.path.join(self.meta_dir, "info.json")
        with open(info_file, "w") as f:
            json.dump(info_config, f, indent=4)
        self.logger.info(f"Generated {info_file}")

        # 4. meta/episodes.jsonl
        episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
        with jsonlines.open(episodes_file, mode="w") as writer:
            # Sort by episode index just in case
            sorted_episodes = sorted(all_episode_details, key=lambda ep: ep["index"])
            for episode_detail in sorted_episodes:
                writer.write({
                    "episode_index": episode_detail["index"],
                    "tasks": episode_detail["tasks"],  # List of task descriptions for this episode
                    "length": episode_detail["length"],
                })
        self.logger.info(f"Generated {episodes_file} with {total_episodes} entries.")

        # 5. meta/stats.json (Optional, using last episode's stats)
        if COMPUTE_STATS and last_episode_stats:
            self._write_stats(last_episode_stats)
        elif COMPUTE_STATS:
            self.logger.warning(
                "COMPUTE_STATS is True, but no stats data was provided for the last episode. Skipping stats.json."
            )

    def write_parquet(self, df: pd.DataFrame, episode_index: int):
        """Writes the episode data to a Parquet file."""
        # [Keep implementation as before]
        parquet_filename = f"episode_{episode_index:06d}.parquet"
        parquet_filepath = os.path.join(self.data_dir, parquet_filename)
        try:
            df.to_parquet(parquet_filepath, engine="pyarrow", index=False)
            self.logger.info(f"Generated {parquet_filepath}")
        except Exception as e:
            self.logger.error(f"Error writing Parquet file {parquet_filepath}: {e}", exc_info=True)
            raise

    def write_video(self, episode_frames: list[Image.Image], fps: float, episode_index: int, video_key=None):
        """Writes the episode frames to an MP4 video file."""
        video_filename = f"episode_{episode_index:06d}.mp4"
        if video_key is None:
            video_filepath = os.path.join(self.video_dir_specific, video_filename)
        else:
            video_dir_specific = os.path.join(self.video_dir_base, f"observation.images.{video_key}")
            self._ensure_dir(video_dir_specific)
            video_filepath = os.path.join(video_dir_specific, video_filename)
        try:
            numpy_frames = []
            for frame in tqdm(episode_frames, desc="Converting frames for video"):
                # Ensure RGB format for imageio
                if (frame.mode in ("RGBA", "BGRA", "P")) or (frame.mode == "L"):
                    frame = frame.convert("RGB")
                elif frame.mode == "BGR":
                    # imageio expects RGB, so convert BGR
                    r, g, b = frame.split()
                    frame = Image.merge("RGB", (b, g, r))

                # Check if conversion resulted in RGB
                if frame.mode != "RGB":
                    self.logger.warning(f"Unexpected frame mode '{frame.mode}'. Skipping.")
                    continue
                numpy_frames.append(np.array(frame))

            if numpy_frames:
                ffmpeg_params = [
                    "-cpu-used",
                    "8",
                    "-crf",
                    "30",
                    "-threads",
                    "0",
                ]
                if USE_AV1:
                    imageio.mimwrite(
                        video_filepath,
                        numpy_frames,
                        fps=fps,
                        macro_block_size=16,
                        quality=8,
                        codec="libaom-av1",
                        ffmpeg_params=ffmpeg_params,
                    )
                else:
                    imageio.mimwrite(
                        video_filepath,
                        numpy_frames,
                        fps=fps,
                        macro_block_size=16,
                        quality=8,
                    )

                self.logger.info(f"Generated {video_filepath}")
            else:
                self.logger.warning("No frames collected/converted to write video.")
        except Exception as e:
            self.logger.error(f"Error writing video file {video_filepath}: {e}", exc_info=True)

    def write_extra_video(self, episode_frames: list[Image.Image], fps: float, episode_index: int, extra_id: str):
        """Writes the episode frames to an MP4 video file."""

        extra_video_dir_specific = f"{self.video_dir_specific}.{extra_id}"
        self._ensure_dir(extra_video_dir_specific)

        video_filename = f"episode_{episode_index:06d}.mp4"
        video_filepath = os.path.join(extra_video_dir_specific, video_filename)
        try:
            numpy_frames = []
            for frame in tqdm(episode_frames, desc="Converting frames for video"):
                # Ensure RGB format for imageio
                if (frame.mode in ("RGBA", "BGRA", "P")) or (frame.mode == "L"):
                    frame = frame.convert("RGB")
                elif frame.mode == "BGR":
                    # imageio expects RGB, so convert BGR
                    r, g, b = frame.split()
                    frame = Image.merge("RGB", (b, g, r))

                # Check if conversion resulted in RGB
                if frame.mode != "RGB":
                    self.logger.warning(f"Unexpected frame mode '{frame.mode}'. Skipping.")
                    continue
                numpy_frames.append(np.array(frame))

            if numpy_frames:
                imageio.mimwrite(video_filepath, numpy_frames, fps=fps, macro_block_size=16, quality=8)
                self.logger.info(f"Generated {video_filepath}")
            else:
                self.logger.warning("No frames collected/converted to write video.")
        except Exception as e:
            self.logger.error(f"Error writing video file {video_filepath}: {e}", exc_info=True)

    def get_extra_video_dir(self, extra_id: str):
        """Writes the episode frames to an MP4 video file."""

        extra_video_dir_specific = f"{self.video_dir_specific}.{extra_id}"
        self._ensure_dir(extra_video_dir_specific)
        return extra_video_dir_specific

    def _write_stats(self, stats_data: dict):
        """Writes the calculated statistics to stats.json."""
        if stats_data:
            stats_file = os.path.join(self.meta_dir, "stats.json")
            try:
                with open(stats_file, "w") as f:
                    json.dump(stats_data, f, indent=4)
                self.logger.info(f"Generated {stats_file}")
            except Exception as e:
                self.logger.error(f"Error writing stats file: {e}")
        else:
            self.logger.warning("Stats calculation failed or produced no data. Skipping stats.json.")


def main():  # noqa C901
    dataset_file = args_cli.dataset_file
    image_path = args_cli.image_path
    lerobot_path = args_cli.output_path
    episode_num = args_cli.episode_num
    task_description = args_cli.task_description

    if not os.path.isfile(dataset_file):
        raise ValueError(f"The input dataset file {dataset_file} does not exist.")
    if not os.path.isdir(image_path):
        raise ValueError(f"The input image path {image_path} does not exist.")
    if os.path.exists(lerobot_path):
        raise ValueError(f"The output path {lerobot_path} already exist.")

    # --- Prepare Output Directory ---
    if os.path.exists(lerobot_path):
        logger.warning(f"Removing existing output directory: {lerobot_path}")
        shutil.rmtree(lerobot_path)

    env_trial_frames = get_env_trial_frames(image_path, "table_high_cam", args_cli.type[0])
    trial_nums = env_trial_frames[0]

    with h5py.File(dataset_file, "r") as f:
        first_episode = True
        episode_index = 0
        dataset_formatter = None
        total_frames_processed = 0
        last_episode_stats = None
        all_episode_details = []
        unique_tasks = {task_description: 0}
        global_fps = 15
        global_video_info = {"width": 640, "height": 480, "channels": 3}

        for dataset in range(0, min(episode_num, len(f["data"]))):
            episode_data = []
            states = np.array(f[f"data/demo_{dataset}/states/articulation/robot/joint_position"][:], np.float32)
            states[:, -2] = [1000 - (finger[0] + finger[1]) / 0.08 * 1000 for finger in states[:, -2:]]
            states = np.delete(states, -1, axis=1)

            actions = []
            deta_actions = np.array(f[f"data/demo_{dataset}/actions"][:], np.float32)
            joint_actions = np.array(f[f"data/demo_{dataset}/joint_actions"][:], np.float32)
            for row in range(0, deta_actions.shape[0]):
                step = deta_actions[row]
                joint_action = joint_actions[row]
                cur_action = np.zeros(7, dtype=np.float32)
                cur_action[:-1] = joint_action
                cur_action[-1] = 1000 if step[-1] == 1 else 0
                actions.append(copy.deepcopy(cur_action))
            actions = np.array(actions, dtype=np.float32)

            states[:, :-1] = np.degrees(states[:, :-1])
            actions[:, :-1] = np.degrees(actions[:, :-1])

            timestamp_sec = 0.0
            for row in range(min(states.shape[0], actions.shape[0])):
                timestamp_sec = timestamp_sec + 1.0 / global_fps
                step_data = {
                    "timestamp": np.float32(timestamp_sec),
                    "observation.state": states[row],
                    "action": actions[row],
                    # "pil_image": pil_image, # Keep image temporarily
                }
                episode_data.append(step_data)

            if first_episode:
                dataset_formatter = DatasetFormatter(
                    output_dir=lerobot_path,
                    video_key=VIDEO_KEY,
                    state_dim=7,
                    state_dim_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "finger1_joint"],
                    action_dim=7,
                    action_dim_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "finger1_joint"],
                    logger=logger,
                )

            df = pd.DataFrame(episode_data)
            num_frames_this_episode = len(df)

            # Add standard LeRobot columns
            df["episode_index"] = np.int64(episode_index)
            df["frame_index"] = np.arange(num_frames_this_episode, dtype=np.int64)
            df["index"] = np.arange(
                total_frames_processed, total_frames_processed + num_frames_this_episode, dtype=np.int64
            )  # Global index
            df["task_index"] = 0
            for k, task_idx_val in ANNOTATION_KEY_TO_TASK_INDEX.items():
                df[f"annotation.{k}"] = np.int64(task_idx_val)
            df["reward"] = np.float32(0.0)
            df["done"] = False
            df.loc[df.index[-1], "done"] = True  # Mark last frame as done
            # Calculate next state columns (handle boundary)
            df["next.reward"] = df["reward"].shift(-1).astype(np.float32).fillna(0.0)
            df["next.done"] = df["done"].shift(-1).astype(bool).fillna(False)
            # Ensure next state/action are handled if needed (often not stored directly)

            # Define column order
            cols_order = []
            if dataset_formatter.action_dim > 0:
                cols_order.append("action")
            if dataset_formatter.state_dim > 0:
                cols_order.append("observation.state")
            cols_order.extend(["timestamp", "frame_index", "episode_index", "index", "task_index"])
            cols_order.extend([f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()])
            cols_order.extend(["reward", "done", "next.reward", "next.done"])

            final_cols = [col for col in cols_order if col in df.columns]
            missing_cols = set(cols_order) - set(final_cols)
            if missing_cols:
                logger.warning(
                    f"Columns missing from DataFrame for final ordering in ep {episode_index}: {missing_cols}"
                )
            remaining_cols = [col for col in df.columns if col not in final_cols]
            final_cols.extend(remaining_cols)
            df = df[final_cols]

            # --- Video Preprocessing ---
            video_length = min(states.shape[0], actions.shape[0])
            start_frame, end_frame = trial_nums[episode_index]
            video_start = max(start_frame, end_frame - video_length + 1)

            # --- Table High Cam Preprocessing ---
            frame_name_pattern = "table_high_cam_rgb_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            segmentation_name_pattern = (
                "table_high_cam_semantic_segmentation_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            )
            depth_name_pattern = "table_high_cam_depth_trial_{trial_num}_tile_0_step_{frame_idx}.png"

            if "rgb" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path_rgb = os.path.join(
                        image_path, frame_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path_rgb)
                    episode_frames_pil.append(pil_image)
                file_path_rgb = os.path.join(
                    image_path,
                    frame_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path_rgb)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index)
                episode_frames_pil.clear()

            if "semantic_segmentation" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, segmentation_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    segmentation_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(
                    episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY}.segmentation"
                )
                episode_frames_pil.clear()

            if "depth" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, depth_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    depth_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY}.depth")
                episode_frames_pil.clear()

            # --- Table Side Cam Preprocessing ---
            frame_name_pattern = "table_side_cam_rgb_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            segmentation_name_pattern = (
                "table_side_cam_semantic_segmentation_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            )
            depth_name_pattern = "table_side_cam_depth_trial_{trial_num}_tile_0_step_{frame_idx}.png"

            if "rgb" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path_rgb = os.path.join(
                        image_path, frame_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path_rgb)
                    episode_frames_pil.append(pil_image)
                file_path_rgb = os.path.join(
                    image_path,
                    frame_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path_rgb)
                episode_frames_pil.append(pil_image)
                # dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, video_key=VIDEO_KEY2)
                episode_frames_pil.clear()

            if "semantic_segmentation" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, segmentation_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    segmentation_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(
                    episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY2}.segmentation"
                )
                episode_frames_pil.clear()

            if "depth" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, depth_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    depth_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY2}.depth")
                episode_frames_pil.clear()

            # --- Table Cam Preprocessing ---
            frame_name_pattern = "table_cam_rgb_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            segmentation_name_pattern = "table_cam_semantic_segmentation_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            depth_name_pattern = "table_cam_depth_trial_{trial_num}_tile_0_step_{frame_idx}.png"

            if "rgb" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path_rgb = os.path.join(
                        image_path, frame_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path_rgb)
                    episode_frames_pil.append(pil_image)
                file_path_rgb = os.path.join(
                    image_path,
                    frame_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path_rgb)
                episode_frames_pil.append(pil_image)
                # dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, video_key=VIDEO_KEY3)
                episode_frames_pil.clear()

            if "semantic_segmentation" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, segmentation_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    segmentation_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(
                    episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY3}.segmentation"
                )
                episode_frames_pil.clear()

            if "depth" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, depth_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    depth_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY3}.depth")
                episode_frames_pil.clear()

            # --- Left Cam Preprocessing ---
            frame_name_pattern = "left_cam_rgb_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            segmentation_name_pattern = "left_cam_semantic_segmentation_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            depth_name_pattern = "left_cam_depth_trial_{trial_num}_tile_0_step_{frame_idx}.png"

            if "rgb" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path_rgb = os.path.join(
                        image_path, frame_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path_rgb)
                    episode_frames_pil.append(pil_image)
                file_path_rgb = os.path.join(
                    image_path,
                    frame_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path_rgb)
                episode_frames_pil.append(pil_image)
                # dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, video_key=VIDEO_KEY4)
                episode_frames_pil.clear()

            if "semantic_segmentation" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, segmentation_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    segmentation_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(
                    episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY4}.segmentation"
                )
                episode_frames_pil.clear()

            if "depth" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, depth_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    depth_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY4}.depth")
                episode_frames_pil.clear()

            # --- Right Cam Preprocessing ---
            frame_name_pattern = "right_cam_rgb_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            segmentation_name_pattern = "right_cam_semantic_segmentation_trial_{trial_num}_tile_0_step_{frame_idx}.png"
            depth_name_pattern = "right_cam_depth_trial_{trial_num}_tile_0_step_{frame_idx}.png"

            if "rgb" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path_rgb = os.path.join(
                        image_path, frame_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path_rgb)
                    episode_frames_pil.append(pil_image)
                file_path_rgb = os.path.join(
                    image_path,
                    frame_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path_rgb)
                episode_frames_pil.append(pil_image)
                # dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, video_key=VIDEO_KEY5)
                episode_frames_pil.clear()

            if "semantic_segmentation" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, segmentation_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    segmentation_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(
                    episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY5}.segmentation"
                )
                episode_frames_pil.clear()

            if "depth" in args_cli.type:
                episode_frames_pil = []
                for frame_idx in range(video_start + 1, video_start + video_length):
                    file_path = os.path.join(
                        image_path, depth_name_pattern.format(trial_num=episode_index, frame_idx=frame_idx)
                    )
                    pil_image = Image.open(file_path)
                    episode_frames_pil.append(pil_image)
                file_path = os.path.join(
                    image_path,
                    depth_name_pattern.format(trial_num=episode_index + 1, frame_idx=video_start + video_length),
                )
                pil_image = Image.open(file_path)
                episode_frames_pil.append(pil_image)
                dataset_formatter.write_parquet(df, episode_index)
                dataset_formatter.write_video(episode_frames_pil, global_fps, episode_index, f"{VIDEO_KEY5}.depth")
                episode_frames_pil.clear()

            # --- Calculate Stats (Optional) ---
            if COMPUTE_STATS:
                last_episode_stats = dataset_formatter._calculate_stats(df)  # Overwrites previous stats

            # --- Aggregate Information ---
            all_episode_details.append({
                "index": episode_index,
                "length": num_frames_this_episode,
                "tasks": [task_description],  # Store task description as a list
            })
            total_frames_processed += num_frames_this_episode

            episode_index = episode_index + 1

            # Write final metadata files using aggregated info
        logger.info("\n--- Writing Final Metadata ---")
        dataset_formatter.write_metadata(
            all_episode_details,
            unique_tasks,
            global_fps,  # Use FPS from first bag
            global_video_info,  # Use video info from first bag
            last_episode_stats if COMPUTE_STATS else None,
        )

        # table_high_cam_rgb_trial_578_tile_0_step_378884.png


if __name__ == "__main__":
    # from argparse import ArgumentParser, Namespace
    # from isaacsim import SimulationApp
    # simulation_app = SimulationApp({
    #     "headless": True,
    #     "launch_config": {
    #         "extensions": ["omni.videoencoding"],  # 启用扩展
    #     }})  # start the simulation app, with GUI open
    # from omni.isaac.core.utils.extensions import enable_extension
    # enable_extension("omni.videoencoding")  # required by OIGE
    main()
