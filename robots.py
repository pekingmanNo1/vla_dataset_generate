# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declarative robot configuration for the manipulation examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize_quaternion(orientation: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (4,):
        raise ValueError("Quaternion must contain four WXYZ values.")
    norm = np.linalg.norm(orientation)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must be finite and nonzero.")
    return orientation / norm


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for measured, controlled, and physical grasp frames.

    Position values are expressed in meters. Orientations use WXYZ quaternion ordering.

    Args:
        controller_frame: Robot site controlled by the motion controller.
        measurement_prim_path: Robot-relative prim path used to measure the tool pose.
        measurement_to_controller_position: Translation from the measurement frame to the controller frame.
        measurement_to_controller_orientation: Rotation from the measurement frame to the controller frame.
        controller_to_grasp_position: Translation from the controller frame to the physical grasp frame.
        controller_to_grasp_orientation: Rotation from the controller frame to the physical grasp frame.

    Raises:
        ValueError: If a path, position, or orientation is invalid.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import ToolConfig

        >>> tool = ToolConfig("tool0", "ee_link")
    """

    controller_frame: str
    #: Robot site controlled by the motion controller.
    measurement_prim_path: str
    #: Robot-relative prim path used to measure the tool pose.
    measurement_to_controller_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Translation from the measurement frame to the controller frame.
    measurement_to_controller_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: WXYZ rotation from the measurement frame to the controller frame.
    controller_to_grasp_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Translation from the controller frame to the physical grasp frame.
    controller_to_grasp_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: WXYZ rotation from the controller frame to the physical grasp frame.

    def __post_init__(self) -> None:
        """Validate and normalize the configured tool transforms."""
        if not self.controller_frame or not self.measurement_prim_path or self.measurement_prim_path.startswith("/"):
            raise ValueError("Tool frame must be non-empty and measurement prim path must be robot-relative.")
        for field in ("measurement_to_controller_position", "controller_to_grasp_position"):
            position = np.asarray(getattr(self, field), dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"{field} must contain three finite XYZ values.")
            object.__setattr__(self, field, tuple(float(value) for value in position))
        for field in ("measurement_to_controller_orientation", "controller_to_grasp_orientation"):
            orientation = _normalize_quaternion(getattr(self, field))
            object.__setattr__(self, field, tuple(float(value) for value in orientation))


@dataclass(frozen=True)
class JointGripperConfig:
    """Configuration for a gripper controlled by articulation joints.

    Args:
        joint_names: Names of the articulation joints that actuate the gripper.
        open_positions: Joint positions for an open gripper.
        closed_positions: Joint positions for a closed gripper.
        position_tolerance: Maximum joint-position error for exact completion.
        velocity_tolerance: Maximum joint speed for stopped-short completion.
        minimum_close_fraction: Minimum commanded travel required for stopped-short completion.

    Raises:
        ValueError: If joint names, positions, or completion thresholds are invalid.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import JointGripperConfig

        >>> gripper = JointGripperConfig(("finger_joint",), (0.04,), (0.0,))
    """

    joint_names: tuple[str, ...]
    #: Names of the articulation joints that actuate the gripper.
    open_positions: tuple[float, ...]
    #: Joint positions for an open gripper.
    closed_positions: tuple[float, ...]
    #: Joint positions for a closed gripper.
    position_tolerance: float = 0.002
    #: Maximum joint-position error for exact completion.
    velocity_tolerance: float = 0.01
    #: Maximum joint speed for stopped-short completion.
    minimum_close_fraction: float = 0.1
    #: Minimum commanded travel required for stopped-short completion.

    def __post_init__(self) -> None:
        """Validate the joint gripper configuration."""
        size = len(self.joint_names)
        if size == 0 or len(self.open_positions) != size or len(self.closed_positions) != size:
            raise ValueError("Joint gripper names and command positions must have the same non-zero length.")
        if len(set(self.joint_names)) != size:
            raise ValueError("Joint gripper names must be unique.")
        if not np.isfinite((*self.open_positions, *self.closed_positions)).all():
            raise ValueError("Gripper command positions must be finite.")
        if self.position_tolerance <= 0.0 or self.velocity_tolerance <= 0.0:
            raise ValueError("Gripper tolerances must be positive.")
        if not 0.0 < self.minimum_close_fraction <= 1.0:
            raise ValueError("minimum_close_fraction must be in (0, 1].")


@dataclass(frozen=True)
class SurfaceGripperConfig:
    """Configuration for a surface gripper attached below the robot prim.

    Args:
        relative_path: Robot-relative path to the surface gripper prim.

    Raises:
        ValueError: If ``relative_path`` is empty or absolute.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import SurfaceGripperConfig

        >>> gripper = SurfaceGripperConfig("ee_link/SurfaceGripper")
    """

    relative_path: str
    #: Robot-relative path to the surface gripper prim.

    def __post_init__(self) -> None:
        """Validate the surface gripper path."""
        if not self.relative_path or self.relative_path.startswith("/"):
            raise ValueError("Surface gripper path must be non-empty and robot-relative.")


@dataclass(frozen=True)
class RobotConfig:
    """Data needed to load and control a supported example robot.

    Paths other than ``asset_path`` are relative to the robot prim. Quaternions use
    Isaac Sim's ``[w, x, y, z]`` convention.

    Args:
        name: Case-insensitive name used to select the robot.
        asset_path: Assets-root-relative path to the robot USD.
        variants: Variant-set selections applied when loading the default asset.
        default_joint_positions: Named articulation positions used for the default state.
        tool: Tool-frame and grasp-frame configuration.
        grasp_orientation: Default world-space grasp orientation in WXYZ ordering.
        gripper: Joint-driven or surface-gripper configuration.

    Raises:
        ValueError: If robot identity, joints, or grasp orientation are invalid.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import get_robot_config

        >>> config = get_robot_config("franka")
        >>> print(config.name)
        franka
    """

    name: str
    #: Case-insensitive name used to select the robot.
    asset_path: str
    #: Assets-root-relative path to the robot USD.
    variants: tuple[tuple[str, str], ...]
    #: Variant-set selections applied when loading the default asset.
    default_joint_positions: tuple[tuple[str, float], ...]
    #: Named articulation positions used for the default state.
    tool: ToolConfig
    #: Tool-frame and grasp-frame configuration.
    grasp_orientation: tuple[float, float, float, float]
    #: Default world-space grasp orientation in WXYZ ordering.
    gripper: JointGripperConfig | SurfaceGripperConfig
    #: Gripper configuration used by the example controllers.

    def __post_init__(self) -> None:
        """Validate the robot configuration."""
        if not self.name or not self.asset_path:
            raise ValueError("Robot name and asset path must be non-empty.")
        names = [name for name, _ in self.default_joint_positions]
        values = [value for _, value in self.default_joint_positions]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Default joint names must be non-empty and unique.")
        if not np.isfinite(values).all():
            raise ValueError("Default joint positions must be finite.")
        object.__setattr__(self, "grasp_orientation", tuple(_normalize_quaternion(self.grasp_orientation)))


_FRANKA = RobotConfig(
    name="franka",
    asset_path="/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
    variants=(("Gripper", "alternatefinger"), ("Mesh", "performance")),
    default_joint_positions=(
        ("panda_joint1", 0.012),
        ("panda_joint2", -0.568),
        ("panda_joint3", 0.0),
        ("panda_joint4", -2.811),
        ("panda_joint5", 0.0),
        ("panda_joint6", 3.037),
        ("panda_joint7", 0.741),
        ("panda_finger_joint1", 0.04),
        ("panda_finger_joint2", 0.04),
    ),
    tool=ToolConfig(
        controller_frame="panda_hand",
        measurement_prim_path="panda_hand",
        controller_to_grasp_position=(0.0, 0.0, 0.1034),
    ),
    grasp_orientation=(0.0, 1.0, 0.0, 0.0),
    gripper=JointGripperConfig(
        joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        open_positions=(0.04, 0.04),
        closed_positions=(0.0, 0.0),
        # The Franka asset reports about 0.08 m/s residual finger velocity at contact.
        velocity_tolerance=0.1,
    ),
)

_UR10 = RobotConfig(
    name="ur10",
    asset_path="/Isaac/Robots_Multiphysics/UniversalRobots/ur10/ur10.usda",
    variants=(("Gripper", "short_suction"),),
    default_joint_positions=(
        ("shoulder_pan_joint", -1.5707963268),
        ("shoulder_lift_joint", -1.5707963268),
        ("elbow_joint", -1.5707963268),
        ("wrist_1_joint", -1.5707963268),
        ("wrist_2_joint", 1.5707963268),
        ("wrist_3_joint", 0.0),
    ),
    tool=ToolConfig(
        controller_frame="tool0",
        measurement_prim_path="ee_link",
        measurement_to_controller_orientation=(0.0, 0.70710678, 0.0, 0.70710678),
        controller_to_grasp_position=(0.0, 0.0, 0.17),
        controller_to_grasp_orientation=(0.5, 0.5, -0.5, 0.5),
    ),
    grasp_orientation=(0.0, 0.70710678, 0.0, -0.70710678),
    gripper=SurfaceGripperConfig(relative_path="ee_link/SurfaceGripper"),
)

_SUPPORTED_ROBOTS = {config.name: config for config in (_FRANKA, _UR10)}


def get_robot_config(name: str) -> RobotConfig:
    """Get a supported robot configuration by name.

    Args:
        name: Case-insensitive robot name.

    Returns:
        Configuration for the requested robot.

    Raises:
        ValueError: If the robot name is unsupported.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import get_robot_config

        >>> get_robot_config("franka").name
        'franka'
    """
    try:
        return _SUPPORTED_ROBOTS[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUPPORTED_ROBOTS))
        raise ValueError(f"Unsupported robot {name!r}. Expected one of: {supported}.") from exc


__all__ = [
    "JointGripperConfig",
    "RobotConfig",
    "SurfaceGripperConfig",
    "ToolConfig",
    "get_robot_config",
]
