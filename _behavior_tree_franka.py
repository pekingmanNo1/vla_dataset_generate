# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""cuMotion-backed Franka motion adapter shared by behavior-tree nodes."""

from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot

from .robots import JointGripperConfig
from .scenario import ManipulationScenario

_PICKUP_OBJECT_ROOT = "/World/Pyramid"
_ROBOT_PATHS = ("/World/franka_left", "/World/franka_right")


class FrankaMotionAdapter:
    """Motion adapter for an existing Franka behavior-tree robot.

    Args:
        robot_path: Absolute prim path of a Franka already present on the stage.

    Raises:
        RuntimeError: If the robot, motion configuration, or collision world cannot initialize.
    """

    def __init__(self, robot_path: str) -> None:
        self._scenario = ManipulationScenario("franka", robot_prim_path=robot_path)
        self._scenario.bind_existing_robot()
        self._scenario.initialize_world_binding(exclude_prim_paths=[*_ROBOT_PATHS, _PICKUP_OBJECT_ROOT])

        config = self._scenario.robot_config
        if not isinstance(config.gripper, JointGripperConfig):
            raise RuntimeError("The Franka behavior-tree example requires a joint-driven gripper.")
        self._gripper = config.gripper
        self._gripper_indices = [self._scenario.joint_space.index(name) for name in self._gripper.joint_names]

        cumotion_robot = load_cumotion_supported_robot(config.name)
        site_space = cumotion_robot.robot_description.tool_frame_names()
        tool_frame = config.tool.controller_frame
        if tool_frame not in site_space:
            raise RuntimeError(
                f"cuMotion configuration for {config.name!r} does not support tool frame {tool_frame!r}."
            )
        self._site_space = site_space
        self._tool_frame = tool_frame
        self._controller = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=self._scenario.world_interface,
            robot_joint_space=self._scenario.joint_space,
            robot_site_space=site_space,
            tool_frame=tool_frame,
        )
        self._motion_time = 0.0
        self._motion_reset_needed = True

    def get_tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the measured controller-frame pose.

        Returns:
            Batched world-space positions and WXYZ orientations.
        """
        return self._scenario.read_tool_pose()

    def move_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        delta_time: float,
        *,
        reset: bool = False,
    ) -> None:
        """Advance cuMotion toward a world-space tool pose.

        Args:
            position: Target position with shape ``(3,)`` or ``(1, 3)``.
            orientation: Target WXYZ quaternion with shape ``(4,)`` or ``(1, 4)``.
            delta_time: Elapsed simulation time since the previous tick.
            reset: Whether to reset RMPflow before advancing.

        Raises:
            RuntimeError: If RMPflow rejects a reset.
            ValueError: If the target pose or time step is invalid.
        """
        target_position = np.asarray(position, dtype=np.float32).reshape(-1, 3)
        target_orientation = np.asarray(orientation, dtype=np.float32).reshape(-1, 4)
        if target_position.shape != (1, 3) or target_orientation.shape != (1, 4):
            raise ValueError("Franka motion targets must contain exactly one pose.")
        if not np.isfinite((*target_position[0], *target_orientation[0], delta_time)).all() or delta_time < 0.0:
            raise ValueError("Franka motion targets and delta time must be finite, with non-negative delta time.")

        estimated = self._scenario.read_robot_state()
        setpoint = self._make_setpoint(target_position[0], target_orientation[0])
        self._scenario.sync_world()
        if reset or self._motion_reset_needed:
            self._motion_time = 0.0
            if not self._controller.reset(estimated, setpoint, self._motion_time):
                self._motion_reset_needed = True
                raise RuntimeError("RmpFlowController reset failed.")
            self._motion_reset_needed = False

        desired = self._controller.forward(estimated, setpoint, self._motion_time)
        self._scenario.apply_robot_state(desired)
        self._motion_time += delta_time

    def open_gripper(self) -> None:
        """Command the Franka gripper open."""
        self._scenario.articulation.set_dof_position_targets(
            self._gripper.open_positions, dof_indices=self._gripper_indices
        )

    def close_gripper(self) -> None:
        """Command the Franka gripper closed."""
        self._scenario.articulation.set_dof_position_targets(
            self._gripper.closed_positions, dof_indices=self._gripper_indices
        )

    def get_downward_orientation(self) -> np.ndarray:
        """Get the configured downward-facing tool orientation.

        Returns:
            WXYZ quaternion for a downward Franka grasp.
        """
        return np.asarray(self._scenario.robot_config.grasp_orientation, dtype=np.float32)

    def reset_to_default_pose(self) -> None:
        """Reset the robot to its configured default pose."""
        defaults = dict(self._scenario.robot_config.default_joint_positions)
        names = self._scenario.joint_space
        indices = [index for index, name in enumerate(names) if name in defaults]
        positions = [defaults[names[index]] for index in indices]
        self._scenario.articulation.set_dof_positions(positions, dof_indices=indices)
        self._scenario.articulation.set_dof_position_targets(positions, dof_indices=indices)
        self._motion_reset_needed = True

    def is_valid(self) -> bool:
        """Return whether the bound articulation remains valid.

        Returns:
            True when the stage and physics tensor views remain valid.
        """
        articulation = self._scenario.articulation
        return bool(articulation.valid and articulation.is_physics_tensor_entity_valid())

    def cleanup(self) -> None:
        """Release references owned by this controller."""
        self._scenario.cleanup()

    def _make_setpoint(self, position: np.ndarray, orientation: np.ndarray) -> mg.RobotState:
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._tool_frame], wp.array([position], dtype=wp.float32, device="cpu")),
                orientations=([self._tool_frame], wp.array([orientation], dtype=wp.float32, device="cpu")),
            )
        )


_ADAPTERS: dict[str, FrankaMotionAdapter] = {}


def get_franka_motion_adapter(robot_path: str) -> FrankaMotionAdapter:
    """Get the shared motion adapter for one behavior-tree robot.

    Args:
        robot_path: Absolute prim path of the Franka.

    Returns:
        Shared motion adapter for the requested robot.
    """
    adapter = _ADAPTERS.get(robot_path)
    if adapter is not None and adapter.is_valid():
        return adapter
    if adapter is not None:
        adapter.cleanup()
    adapter = FrankaMotionAdapter(robot_path)
    _ADAPTERS[robot_path] = adapter
    return adapter


def discard_franka_motion_adapter(robot_path: str) -> None:
    """Discard a cached motion adapter after a failed stage binding.

    Args:
        robot_path: Absolute prim path of the Franka.
    """
    adapter = _ADAPTERS.pop(robot_path, None)
    if adapter is not None:
        adapter.cleanup()


def clear_franka_motion_adapters() -> None:
    """Release all cached behavior-tree motion adapters."""
    for adapter in _ADAPTERS.values():
        adapter.cleanup()
    _ADAPTERS.clear()
