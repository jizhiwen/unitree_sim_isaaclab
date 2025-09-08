# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import json
import threading
import weakref
from collections.abc import Callable

import carb
import isaaclab.utils.math as math_utils
import numpy as np
import omni

# pico control
import rclpy
import torch
from isaaclab.assets import Articulation
from isaaclab.devices.device_base import DeviceBase
from isaaclab.envs import ManagerBasedEnv
from rclpy.node import Node
from std_msgs.msg import String

lock = threading.Lock()
close_gripper = False
current_delta_pos = np.zeros(3)  # (x, y, z)
current_delta_rot = np.zeros(3)  # (roll, pitch, yaw)

ee_pos_curr = None
ee_quat_curr = None


def ros_thread(node):
    """ROS2节点运行线程"""
    rclpy.spin(node)


class CustomRos2Subscriber(Node):
    def __init__(self, node_name="custom_subscriber"):
        super().__init__(node_name)
        # 订阅关节信息
        self.sub_joint = self.create_subscription(String, "pico_driver/pose/sim", self.pico_topic_callback, 10)
        self.sub_joint  # 防止未使用变量警告

    def parse_pico_msg(self, msg) -> bool:
        try:
            # 解析JSON数据
            root = json.loads(msg.data)
        except json.JSONDecodeError as e:
            print(f"解析JSON失败 {str(e)}")
            return False

        # 遍历JSON数组，只处理索引为1的元素（右手数据）
        for i, obj in enumerate(root):
            if i != 1:  # 仅处理右手数据
                continue

            # 提取位置信息
            self.x = obj["position"]["x"]
            self.y = obj["position"]["y"]
            self.z = obj["position"]["z"]
            # 提取旋转信息
            self.w = obj["rotation"]["w"]
            self.rx = obj["rotation"]["x"]
            self.ry = obj["rotation"]["y"]
            self.rz = obj["rotation"]["z"]
            # 提取其他控制信息
            self.axisClick = obj["axisClick"]
            self.axisX = obj["axisX"]
            self.axisY = obj["axisY"]
            self.indexTrig = obj["indexTrig"]
            self.handTrig = obj["handTrig"]
            self.keyOne = obj["keyOne"]
            self.keyTwo = obj["keyTwo"]
            self.ts = obj["ts"]

            # 记录日志
            # print(f"Group : {i}:")
            # print(f"Group : {i}:")
            # print(f"Position : x={self.x:.17f}, y={self.y:.17f}, z={self.z:.17f}")
            # print(f"Rotation : w={self.w:.17f}, x={self.rx:.17f}, y={self.ry:.17f}, z={self.rz:.17f}")
            # print(f"axisClick : {self.axisClick}")
            # print(f"axisX : {self.axisX:.17f}, axisY : {self.axisY:.17f}")
            # print(f"indexTrig : {self.indexTrig:.17f}")
            # print(f"handTrig : {'true' if self.handTrig else 'false'}")
            # print(f"keyone : {self.keyOne}, keytwo : {self.keyTwo}")
            # print(f"timestamp : {self.ts}")

        return True

    def fit_to_key(self):
        global close_gripper
        if self.indexTrig <= 0.5:
            close_gripper = False
        else:
            close_gripper = True

        with lock:
            current_delta_pos[0] = self.z * 1.0  # 前后
            current_delta_pos[1] = -self.x * 1.0  # 左右
            current_delta_pos[2] = self.y * 1.0     # 上下

            # 目前pico的姿态信息还没有发布出来，目前仅使用位置xyz
            #current_delta_rot[0] = self.rx * 0.5  # 横滚
            #current_delta_rot[1] = self.rz * 0.8  # 俯仰
            current_delta_rot[0] = self.rx  # 横滚
            current_delta_rot[1] = self.rz  # 俯仰
            current_delta_rot[2] = self.ry  # 航向

        global ee_pos_curr, ee_quat_curr
        with lock:
            if not self.handTrig:
                ee_pos_curr = None
                ee_quat_curr = None

    def pico_topic_callback(self, msg):
        # print("""接收到 pico_driver/pose 消息后的处理逻辑""")

        if self.parse_pico_msg(msg):
            self.fit_to_key()
        else:
            print("pico 数据解析失败")


# pico control


class Se3PicoG1Abs(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, env: ManagerBasedEnv, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store asset
        self._asset: Articulation = env.scene["robot"]
        # parse the body index
        body_ids, body_names = self._asset.find_bodies("right_hand_base_link")
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._offset_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, device=torch.device("cuda:0"))
        self._offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device=torch.device("cuda:0"))

        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        # pico control
        rclpy.init()
        self.subscriber_node = CustomRos2Subscriber()
        self.ros_t = threading.Thread(target=ros_thread, args=(self.subscriber_node,))
        self.ros_t.start()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset

        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        global ee_pos_curr, ee_quat_curr, close_gripper
        # obtain quantities from simulation
        with lock:
            if ee_pos_curr is None or ee_quat_curr is None:
                ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

            # source_pos: torch.Tensor, source_rot: torch.Tensor, delta_pose: torch.Tensor, eps: float = 1.0e-6
            target_pos, target_rot = math_utils.apply_delta_pose(
                ee_pos_curr,
                ee_quat_curr,
                torch.tensor(
                    np.array([np.concatenate((current_delta_pos, current_delta_rot))]), device=torch.device("cuda:0")
                ),
            )
        # print(current_delta_pos, current_delta_rot)
        # print(np.concatenate((target_pos.cpu().numpy()[0] , target_rot.cpu().numpy()[0])))
        return (
            np.concatenate((target_pos.cpu().numpy()[0], target_rot.cpu().numpy()[0])),
            close_gripper or self._close_gripper,
        )

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward)
            "A": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "S": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "W": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "X": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "Z": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }
