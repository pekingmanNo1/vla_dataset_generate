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

"""Behavior-tree nodes for controlling a Franka manipulator."""

from __future__ import annotations

import math
from typing import TypeVar

import carb
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from omni.behavior.tree.core import (
    BlackboardRef,
    IBehaviorActionNodeContext,
    IBehaviorBlackboard,
    NodeResetReason,
    NodeStatus,
    UsdActionNode,
    node,
    value_port,
)

from ._behavior_tree_franka import (
    FrankaMotionAdapter,
    discard_franka_motion_adapter,
    get_franka_motion_adapter,
)

_MAX_ROBOT_RESOLVE_ATTEMPTS = 30
_STACK_SIZE = 3
_BlackboardValue = TypeVar("_BlackboardValue")

_PYRAMID_CUBE_PATHS = (
    "/World/Pyramid/CubeTop",
    "/World/Pyramid/CubeMiddleLeft",
    "/World/Pyramid/CubeMiddleRight",
    "/World/Pyramid/CubeBaseLeft",
    "/World/Pyramid/CubeBaseCenter",
    "/World/Pyramid/CubeBaseRight",
)


def _get_robot_id(prim_path: str) -> str:
    """Return the stable cooperative-task ID encoded in a robot prim name."""
    return "right" if "right" in prim_path.rsplit("/", 1)[-1].lower() else "left"


def _get_local_blackboard(context: IBehaviorActionNodeContext) -> IBehaviorBlackboard | None:
    """Resolve the per-tree local blackboard."""
    return BlackboardRef("local", "").resolve(context.get_tree())


def _blackboard_get(blackboard: IBehaviorBlackboard, key: str, default: _BlackboardValue) -> _BlackboardValue:
    """Read a blackboard entry with a default value."""
    return blackboard[key] if key in blackboard else default


def _get_local_string(local: IBehaviorBlackboard | None, key: str) -> str:
    """Read a string from the tree-local blackboard."""
    return str(_blackboard_get(local, key, "") or "") if local is not None else ""


def _get_downward_orientation(robot: FrankaMotionAdapter, yaw_degrees: float) -> np.ndarray:
    """Return the standard downward orientation with a world-Z yaw offset."""
    downward_orientation = np.asarray(robot.get_downward_orientation(), dtype=np.float32)
    yaw_radians = math.radians(yaw_degrees)
    yaw_orientation = np.asarray([math.cos(yaw_radians / 2.0), 0.0, 0.0, math.sin(yaw_radians / 2.0)], dtype=np.float32)
    return transform_utils.quaternion_multiplication(yaw_orientation, downward_orientation, device="cpu").numpy()


def _set_pickup_reference(local: IBehaviorBlackboard, cube_position: np.ndarray) -> None:
    """Store the cube position used to detect motion during pickup."""
    local["pickup_reference"] = cube_position.tolist()


def _pickup_moved(local: IBehaviorBlackboard, cube_position: np.ndarray, tolerance: float) -> bool:
    """Check whether a cube moved away from its recorded pickup position."""
    reference = np.asarray(_blackboard_get(local, "pickup_reference", [0.0, 0.0, 0.0]), dtype=np.float32)
    return float(np.linalg.norm(cube_position - reference)) > tolerance


def _update_pick_targets(local: IBehaviorBlackboard, cube_position: np.ndarray) -> None:
    """Move the pickup target Xforms to the cube's latest position."""
    for key, offset in (
        ("approach_target", np.asarray([0.0, 0.0, 0.20], dtype=np.float32)),
        ("grasp_target", np.asarray([0.0, 0.0, 0.09], dtype=np.float32)),
        ("lift_target", np.asarray([0.0, 0.0, 0.35 - float(cube_position[2])], dtype=np.float32)),
    ):
        target_path = _get_local_string(local, key)
        if target_path:
            XformPrim(target_path).set_world_poses(positions=cube_position + offset)
    _set_pickup_reference(local, cube_position)


def _request_recovery(context: IBehaviorActionNodeContext, local: IBehaviorBlackboard | None) -> bool:
    """Raise this tree instance's recovery flag on the shared blackboard."""
    recovery_key = _get_local_string(local, "recovery_key")
    if not recovery_key:
        return False
    context.get_blackboard()[recovery_key] = True
    return True


class _FrankaNodeBase(UsdActionNode):
    """Base class that lazily resolves the Franka bound to the tree prim."""

    def __init__(self) -> None:
        super().__init__()
        self._robot: FrankaMotionAdapter | None = None
        self._resolve_attempts = 0
        self._resolve_failed = False

    def on_init(self, context: IBehaviorActionNodeContext) -> None:
        """Initialize transient node state."""
        super().on_init(context)
        self._robot = None
        self._resolve_attempts = 0
        self._resolve_failed = False

    def on_reset(self, reason: NodeResetReason) -> None:
        """Release transient robot state when execution resets."""
        self._robot = None
        self._resolve_attempts = 0
        self._resolve_failed = False

    def _resolve_robot(self) -> FrankaMotionAdapter | None:
        """Resolve the Franka motion adapter associated with the tree prim."""
        if self._robot is not None:
            try:
                if self._robot.is_valid():
                    return self._robot
            except (IndexError, RuntimeError, ValueError):
                pass
            self._robot = None
        if self._resolve_failed:
            return None

        robot_path = self.get_prim_path()
        try:
            robot = get_franka_motion_adapter(robot_path)
            robot.get_tool_pose()
            self._robot = robot
        except (IndexError, RuntimeError, ValueError) as exc:
            discard_franka_motion_adapter(robot_path)
            self._resolve_attempts += 1
            if self._resolve_attempts >= _MAX_ROBOT_RESOLVE_ATTEMPTS:
                carb.log_error(f"[{type(self).__name__}] Failed to resolve Franka at '{robot_path}': {exc}")
                self._resolve_failed = True
        return self._robot

    def _status_while_resolving(self) -> NodeStatus:
        """Return the current status for unresolved robot state."""
        return NodeStatus.FAILURE if self._resolve_failed else NodeStatus.RUNNING


@node(
    type="FrankaReset",
    doc="Reset the bound Franka to its default pick-and-place pose.",
    keywords=["franka", "manipulation", "reset", "home"],
    ports=[value_port("settle_duration", 0.25)],
)
class FrankaReset(_FrankaNodeBase):
    """Reset the Franka to its default pose and wait briefly for physics."""

    def __init__(self) -> None:
        super().__init__()
        self._applied = False
        self._elapsed = 0.0

    def on_init(self, context: IBehaviorActionNodeContext) -> None:
        """Initialize reset timing state."""
        super().on_init(context)
        self._applied = False
        self._elapsed = 0.0

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Apply the default pose and report success after settling."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()

        local = _get_local_blackboard(context)
        if local is not None and bool(_blackboard_get(local, "initialized", False)):
            return NodeStatus.SUCCESS

        settle_duration = float(context.get_input("settle_duration"))
        if settle_duration < 0.0:
            carb.log_error("[FrankaReset] 'settle_duration' must be non-negative")
            return NodeStatus.FAILURE

        if not self._applied:
            robot.reset_to_default_pose()
            self._applied = True
        self._elapsed += context.get_delta_time()
        if self._elapsed < settle_duration:
            return NodeStatus.RUNNING
        if local is not None:
            local["initialized"] = True
        return NodeStatus.SUCCESS

    def on_reset(self, reason: NodeResetReason) -> None:
        """Release robot state and reset timing."""
        super().on_reset(reason)
        self._applied = False
        self._elapsed = 0.0


@node(
    type="FrankaMoveToTarget",
    doc="Move the bound Franka end effector to a target prim pose.",
    keywords=["franka", "manipulation", "move", "inverse kinematics"],
    ports=[
        value_port("target_prim", ""),
        value_port("position_tolerance", 0.01),
        value_port("orientation_tolerance", 0.05),
        value_port("use_target_orientation", False),
        value_port("end_effector_yaw_degrees", 0.0),
        value_port("monitor_pickup_motion", False),
        value_port("monitor_grasp", False),
        value_port("movement_tolerance", 0.008),
        value_port("maximum_grasp_distance", 0.14),
        value_port("warning_interval", 5.0),
    ],
)
class FrankaMoveToTarget(_FrankaNodeBase):
    """Move the Franka end effector to a target Xform pose."""

    def __init__(self) -> None:
        super().__init__()
        self._target: XformPrim | None = None
        self._target_path = ""
        self._motion_started = False
        self._nonconverged_elapsed = 0.0

    def on_init(self, context: IBehaviorActionNodeContext) -> None:
        """Initialize robot and target state."""
        super().on_init(context)
        self._target = None
        self._target_path = ""
        self._motion_started = False
        self._nonconverged_elapsed = 0.0

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Command one cuMotion update and report pose convergence."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()

        target_path = str(context.get_input("target_prim") or "")
        if not target_path:
            carb.log_error("[FrankaMoveToTarget] 'target_prim' must be a non-empty prim path")
            return NodeStatus.FAILURE

        position_tolerance = float(context.get_input("position_tolerance"))
        orientation_tolerance = float(context.get_input("orientation_tolerance"))
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
            carb.log_error("[FrankaMoveToTarget] Pose tolerances must be positive")
            return NodeStatus.FAILURE

        if self._target is None or target_path != self._target_path:
            try:
                self._target = XformPrim(target_path)
            except (RuntimeError, ValueError) as exc:
                carb.log_error(f"[FrankaMoveToTarget] Invalid target prim '{target_path}': {exc}")
                return NodeStatus.FAILURE
            self._target_path = target_path

        local = _get_local_blackboard(context)
        current_cube = _get_local_string(local, "current_cube")
        if (
            local is not None
            and current_cube
            and (context.get_input("monitor_pickup_motion") or context.get_input("monitor_grasp"))
        ):
            try:
                cube_position = RigidPrim(current_cube).get_world_poses()[0].numpy()[0]
                hand_position, _ = robot.get_tool_pose()
            except (IndexError, RuntimeError, ValueError) as exc:
                carb.log_error(f"[FrankaMoveToTarget] Failed to monitor '{current_cube}': {exc}")
                return NodeStatus.FAILURE
            pickup_moved = bool(context.get_input("monitor_pickup_motion")) and _pickup_moved(
                local, cube_position, float(context.get_input("movement_tolerance"))
            )
            grasp_lost = bool(context.get_input("monitor_grasp")) and float(
                np.linalg.norm(cube_position - hand_position[0])
            ) > float(context.get_input("maximum_grasp_distance"))
            if (pickup_moved or grasp_lost) and _request_recovery(context, local):
                return NodeStatus.RUNNING

        try:
            target_position_wp, target_orientation_wp = self._target.get_world_poses()
            target_position = target_position_wp.numpy()
            target_orientation = target_orientation_wp.numpy()
            if not bool(context.get_input("use_target_orientation")):
                target_orientation = _get_downward_orientation(
                    robot, float(context.get_input("end_effector_yaw_degrees"))
                )
            target_orientation = np.asarray(target_orientation).reshape(1, 4)

            delta_time = context.get_delta_time()
            robot.move_to_pose(
                position=target_position,
                orientation=target_orientation,
                delta_time=delta_time,
                reset=not self._motion_started,
            )
            self._motion_started = True
            current_position, current_orientation = robot.get_tool_pose()
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaMoveToTarget] Failed to command target '{target_path}': {exc}")
            return NodeStatus.FAILURE

        position_error = float(np.linalg.norm(current_position[0] - target_position[0]))
        quaternion_dot = float(abs(np.dot(current_orientation[0], target_orientation[0])))
        orientation_error = 2.0 * math.acos(float(np.clip(quaternion_dot, -1.0, 1.0)))
        if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
            self._nonconverged_elapsed = 0.0
            return NodeStatus.SUCCESS
        if local is not None and current_cube and bool(context.get_input("monitor_pickup_motion")):
            warning_interval = float(context.get_input("warning_interval"))
            if warning_interval <= 0.0:
                carb.log_error("[FrankaMoveToTarget] 'warning_interval' must be positive")
                return NodeStatus.FAILURE
            self._nonconverged_elapsed += delta_time
            if self._nonconverged_elapsed >= warning_interval and _request_recovery(context, local):
                carb.log_warn(
                    f"[FrankaMoveToTarget] Motion to '{current_cube}' has not converged after "
                    f"{warning_interval:.2f} seconds; switching to continuous recovery"
                )
                self._nonconverged_elapsed %= warning_interval
        return NodeStatus.RUNNING

    def on_reset(self, reason: NodeResetReason) -> None:
        """Release robot and target state."""
        super().on_reset(reason)
        self._target = None
        self._target_path = ""
        self._motion_started = False
        self._nonconverged_elapsed = 0.0


@node(
    type="FrankaSetGripper",
    doc="Open or close the bound Franka gripper with configurable pre-command and post-command settling.",
    keywords=["franka", "manipulation", "gripper", "grasp"],
    ports=[
        value_port("command", "open"),
        value_port("pre_open_settle_duration", 0.25),
        value_port("pre_close_settle_duration", 0.25),
        value_port("settle_duration", 0.25),
        value_port("movement_tolerance", 0.008),
    ],
)
class FrankaSetGripper(_FrankaNodeBase):
    """Open or close the Franka gripper using physical finger contact."""

    def __init__(self) -> None:
        super().__init__()
        self._pre_open_elapsed = 0.0
        self._pre_close_elapsed = 0.0
        self._elapsed = 0.0

    def on_init(self, context: IBehaviorActionNodeContext) -> None:
        """Initialize gripper command timing."""
        super().on_init(context)
        self._pre_open_elapsed = 0.0
        self._pre_close_elapsed = 0.0
        self._elapsed = 0.0

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Apply the gripper command until its settling duration elapses."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()

        command = str(context.get_input("command") or "").lower()
        settle_duration = float(context.get_input("settle_duration"))
        if settle_duration < 0.0:
            carb.log_error("[FrankaSetGripper] 'settle_duration' must be non-negative")
            return NodeStatus.FAILURE
        delta_time = context.get_delta_time()

        if command == "open":
            pre_open_settle_duration = float(context.get_input("pre_open_settle_duration"))
            if pre_open_settle_duration < 0.0:
                carb.log_error("[FrankaSetGripper] 'pre_open_settle_duration' must be non-negative")
                return NodeStatus.FAILURE
            if self._pre_open_elapsed < pre_open_settle_duration:
                self._pre_open_elapsed += delta_time
                if self._pre_open_elapsed < pre_open_settle_duration:
                    return NodeStatus.RUNNING
            robot.open_gripper()
        elif command == "close":
            local = _get_local_blackboard(context)
            current_cube = _get_local_string(local, "current_cube")
            if local is not None and current_cube:
                try:
                    cube_position = RigidPrim(current_cube).get_world_poses()[0].numpy()[0]
                except (IndexError, RuntimeError, ValueError) as exc:
                    carb.log_error(f"[FrankaSetGripper] Failed to monitor '{current_cube}': {exc}")
                    return NodeStatus.FAILURE
                moved = _pickup_moved(local, cube_position, float(context.get_input("movement_tolerance")))
                if moved and _request_recovery(context, local):
                    return NodeStatus.RUNNING
            pre_close_settle_duration = float(context.get_input("pre_close_settle_duration"))
            if pre_close_settle_duration < 0.0:
                carb.log_error("[FrankaSetGripper] 'pre_close_settle_duration' must be non-negative")
                return NodeStatus.FAILURE
            if self._pre_close_elapsed < pre_close_settle_duration:
                self._pre_close_elapsed += delta_time
                if self._pre_close_elapsed < pre_close_settle_duration:
                    return NodeStatus.RUNNING
            robot.close_gripper()
        else:
            carb.log_error("[FrankaSetGripper] 'command' must be 'open' or 'close'")
            return NodeStatus.FAILURE

        self._elapsed += delta_time
        return NodeStatus.SUCCESS if self._elapsed >= settle_duration else NodeStatus.RUNNING

    def on_reset(self, reason: NodeResetReason) -> None:
        """Release robot state and reset command timing."""
        super().on_reset(reason)
        self._pre_open_elapsed = 0.0
        self._pre_close_elapsed = 0.0
        self._elapsed = 0.0


@node(
    type="FrankaAcquireCube",
    doc="Atomically reserve the next pyramid cube and publish robot-local motion targets.",
    keywords=["franka", "coordination", "blackboard", "reservation"],
)
class FrankaAcquireCube(_FrankaNodeBase):
    """Reserve one shared cube and configure target Xforms for this robot."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Acquire the center-zone lock and reserve the next available cube."""
        if self._resolve_robot() is None:
            return self._status_while_resolving()

        shared = context.get_blackboard()
        local = _get_local_blackboard(context)
        if local is None:
            carb.log_error("[FrankaAcquireCube] Tree-local blackboard is unavailable")
            return NodeStatus.FAILURE

        robot_id = _get_robot_id(self.get_prim_path())
        current_cube = _get_local_string(local, "current_cube")
        center_owner = str(_blackboard_get(shared, "center_owner", "") or "")
        if center_owner and center_owner != robot_id:
            return NodeStatus.RUNNING

        selected_new_cube = not current_cube
        if selected_new_cube:
            cube_paths = _PYRAMID_CUBE_PATHS
            if robot_id == "right":
                cube_paths = tuple(_PYRAMID_CUBE_PATHS[index] for index in (0, 2, 1, 5, 4, 3))
            for cube_path in cube_paths:
                state_key = f"cube_state:{cube_path}"
                if _blackboard_get(shared, state_key, "available") == "available":
                    current_cube = cube_path
                    break
        if not current_cube:
            return NodeStatus.FAILURE

        placed_count = int(_blackboard_get(shared, f"placed_count:{robot_id}", 0))
        try:
            cube_position = RigidPrim(current_cube).get_world_poses()[0].numpy()[0]
            stack_y = -0.28 if robot_id == "left" else 0.28
            goal_position = np.asarray([0.46, stack_y, 0.025 + 0.05 * placed_count], dtype=np.float32)
            target_root = f"/World/Targets/{robot_id}"
            target_positions = {
                "approach_target": cube_position + np.asarray([0.0, 0.0, 0.20]),
                "grasp_target": cube_position + np.asarray([0.0, 0.0, 0.09]),
                "lift_target": np.asarray([cube_position[0], cube_position[1], 0.35]),
                "place_approach_target": goal_position + np.asarray([0.0, 0.0, 0.325]),
                "place_target": goal_position + np.asarray([0.0, 0.0, 0.125]),
                "retract_target": goal_position + np.asarray([0.0, 0.0, 0.325]),
                "object_goal": goal_position,
            }
            target_paths = {name: f"{target_root}/{name}" for name in target_positions}
            for name, position in target_positions.items():
                XformPrim(target_paths[name]).set_world_poses(positions=position)
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaAcquireCube] Failed to prepare targets for '{current_cube}': {exc}")
            return NodeStatus.FAILURE

        if selected_new_cube:
            shared[f"cube_state:{current_cube}"] = f"reserved:{robot_id}"
            shared["center_owner"] = robot_id
            local["current_cube"] = current_cube
        local["recovery_key"] = f"recovery_requested:{robot_id}"
        local["completion_key"] = f"stack_complete:{robot_id}"
        if local["completion_key"] not in shared:
            shared[local["completion_key"]] = False
        for name, target_path in target_paths.items():
            local[name] = target_path
        _set_pickup_reference(local, cube_position)
        shared[local["recovery_key"]] = False
        return NodeStatus.SUCCESS


@node(
    type="FrankaRecoverCube",
    doc="Restage above a cube that moved during pickup or fell from the gripper.",
    keywords=["franka", "manipulation", "recovery", "reactive"],
    ports=[
        value_port("movement_tolerance", 0.008),
        value_port("staging_height", 0.20),
        value_port("position_tolerance", 0.025),
        value_port("stable_duration", 0.20),
        value_port("warning_interval", 5.0),
        value_port("end_effector_yaw_degrees", 90.0),
    ],
)
class FrankaRecoverCube(_FrankaNodeBase):
    """Reactive recovery branch for moved or dropped reserved cubes."""

    def __init__(self) -> None:
        super().__init__()
        self._last_cube_position: np.ndarray | None = None
        self._stable_elapsed = 0.0
        self._warning_elapsed = 0.0
        self._motion_started = False

    def on_init(self, context: IBehaviorActionNodeContext) -> None:
        """Initialize recovery stability tracking."""
        super().on_init(context)
        self._last_cube_position = None
        self._stable_elapsed = 0.0
        self._warning_elapsed = 0.0
        self._motion_started = False

    def on_reset(self, reason: NodeResetReason) -> None:
        """Release robot state and reset recovery stability tracking."""
        super().on_reset(reason)
        self._last_cube_position = None
        self._stable_elapsed = 0.0
        self._warning_elapsed = 0.0
        self._motion_started = False

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Preempt the transfer and restage above the cube when recovery is needed."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()

        shared = context.get_blackboard()
        local = _get_local_blackboard(context)
        current_cube = _get_local_string(local, "current_cube")
        if local is None or not current_cube:
            return NodeStatus.FAILURE

        try:
            cube_position = RigidPrim(current_cube).get_world_poses()[0].numpy()[0]
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaRecoverCube] Failed to read recovery state for '{current_cube}': {exc}")
            return NodeStatus.FAILURE

        movement_tolerance = float(context.get_input("movement_tolerance"))
        recovery_key = _get_local_string(local, "recovery_key")
        recovery_requested = bool(_blackboard_get(shared, recovery_key, False)) if recovery_key else False
        if not recovery_requested:
            return NodeStatus.FAILURE

        robot_id = _get_robot_id(self.get_prim_path())
        center_owner = str(_blackboard_get(shared, "center_owner", "") or "")
        if center_owner and center_owner != robot_id:
            return NodeStatus.RUNNING

        warning_interval = float(context.get_input("warning_interval"))
        if warning_interval <= 0.0:
            carb.log_error("[FrankaRecoverCube] 'warning_interval' must be positive")
            return NodeStatus.FAILURE
        shared["center_owner"] = robot_id
        shared[f"cube_state:{current_cube}"] = f"reserved:{robot_id}"
        delta_time = context.get_delta_time()
        try:
            robot.open_gripper()
            if self._last_cube_position is None:
                self._last_cube_position = cube_position.copy()
                self._stable_elapsed = 0.0
            elif float(np.linalg.norm(cube_position - self._last_cube_position)) > movement_tolerance * 0.5:
                self._stable_elapsed = 0.0
                self._last_cube_position = cube_position.copy()
                self._motion_started = False
            else:
                self._stable_elapsed += delta_time

            _update_pick_targets(local, cube_position)
            staging_position = cube_position + np.asarray(
                [0.0, 0.0, float(context.get_input("staging_height"))], dtype=np.float32
            )
            XformPrim(str(local["approach_target"])).set_world_poses(positions=staging_position)
            orientation = _get_downward_orientation(robot, float(context.get_input("end_effector_yaw_degrees")))
            robot.move_to_pose(
                position=staging_position,
                orientation=orientation,
                delta_time=delta_time,
                reset=not self._motion_started,
            )
            self._motion_started = True
            current_position, _ = robot.get_tool_pose()
            position_error = float(np.linalg.norm(current_position[0] - staging_position))
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaRecoverCube] Failed to restage above '{current_cube}': {exc}")
            return NodeStatus.FAILURE

        stable_duration = float(context.get_input("stable_duration"))
        position_tolerance = float(context.get_input("position_tolerance"))
        if position_error > position_tolerance:
            self._warning_elapsed += delta_time
            if self._warning_elapsed >= warning_interval:
                carb.log_warn(
                    f"[FrankaRecoverCube] Motion to '{current_cube}' has not converged for "
                    f"'{robot_id}' after another {warning_interval:.2f} seconds; still trying"
                )
                self._warning_elapsed %= warning_interval
            return NodeStatus.RUNNING
        self._warning_elapsed = 0.0
        if self._stable_elapsed < stable_duration:
            return NodeStatus.RUNNING

        shared[recovery_key] = False
        self._last_cube_position = None
        self._stable_elapsed = 0.0
        self._warning_elapsed = 0.0
        self._motion_started = False
        return NodeStatus.FAILURE


@node(
    type="FrankaReleaseCenter",
    doc="Release the shared center-zone lock after a cube is lifted.",
    keywords=["franka", "coordination", "blackboard", "lock"],
)
class FrankaReleaseCenter(UsdActionNode):
    """Mark the reserved cube as carried and release shared pickup access."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Release the center lock owned by this robot."""
        shared = context.get_blackboard()
        local = _get_local_blackboard(context)
        robot_id = _get_robot_id(self.get_prim_path())
        current_cube = _get_local_string(local, "current_cube")
        if not current_cube:
            return NodeStatus.FAILURE
        shared[f"cube_state:{current_cube}"] = f"carried:{robot_id}"
        if _blackboard_get(shared, "center_owner", "") == robot_id:
            shared["center_owner"] = ""
        return NodeStatus.SUCCESS


@node(
    type="FrankaCalibratePlaceTargets",
    doc="Adjust place targets from the measured held-object offset.",
    keywords=["franka", "manipulation", "calibration", "place"],
)
class FrankaCalibratePlaceTargets(_FrankaNodeBase):
    """Calibrate placement targets from the physically grasped cube pose."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Move the hand targets so the held cube, rather than the hand, reaches the goal."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()
        local = _get_local_blackboard(context)
        current_cube = _get_local_string(local, "current_cube")
        goal_path = _get_local_string(local, "object_goal")
        if not current_cube or not goal_path:
            return NodeStatus.FAILURE

        try:
            hand_position = robot.get_tool_pose()[0][0]
            object_position = RigidPrim(current_cube).get_world_poses()[0].numpy()[0]
            goal_position = XformPrim(goal_path).get_world_poses()[0].numpy()[0]
            place_position = goal_position - (object_position - hand_position)
            for key, offset in (
                ("place_target", np.asarray([0.0, 0.0, 0.0])),
                ("place_approach_target", np.asarray([0.0, 0.0, 0.20])),
                ("retract_target", np.asarray([0.0, 0.0, 0.20])),
            ):
                target_path = _get_local_string(local, key)
                if not target_path:
                    raise ValueError(f"Missing '{key}' target path")
                XformPrim(target_path).set_world_poses(positions=place_position + offset)
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaCalibratePlaceTargets] Failed to calibrate placement targets: {exc}")
            return NodeStatus.FAILURE
        return NodeStatus.SUCCESS


@node(
    type="FrankaMarkCubePlaced",
    doc="Commit a completed placement to the shared blackboard.",
    keywords=["franka", "coordination", "blackboard", "place"],
    ports=[value_port("required_count", _STACK_SIZE)],
)
class FrankaMarkCubePlaced(UsdActionNode):
    """Record task progress and clear this tree's local cube selection."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Mark the current cube placed by this robot."""
        shared = context.get_blackboard()
        local = _get_local_blackboard(context)
        robot_id = _get_robot_id(self.get_prim_path())
        current_cube = _get_local_string(local, "current_cube")
        if not current_cube:
            return NodeStatus.FAILURE
        required_count = int(context.get_input("required_count"))
        if required_count <= 0:
            carb.log_error("[FrankaMarkCubePlaced] 'required_count' must be positive")
            return NodeStatus.FAILURE
        shared[f"cube_state:{current_cube}"] = f"placed:{robot_id}"
        placed_count = int(_blackboard_get(shared, f"placed_count:{robot_id}", 0)) + 1
        shared[f"placed_count:{robot_id}"] = placed_count
        completion_key = _get_local_string(local, "completion_key")
        if completion_key and placed_count >= required_count:
            shared[completion_key] = True
        local["current_cube"] = ""
        recovery_key = _get_local_string(local, "recovery_key")
        if recovery_key:
            shared[recovery_key] = False
        return NodeStatus.SUCCESS


@node(
    type="FrankaCheckStackComplete",
    doc="Check whether this robot has placed its assigned number of cubes.",
    keywords=["franka", "coordination", "condition"],
    ports=[value_port("required_count", _STACK_SIZE)],
)
class FrankaCheckStackComplete(UsdActionNode):
    """Return success after this robot completes its stack."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Check this robot's shared placement counter."""
        robot_id = _get_robot_id(self.get_prim_path())
        count = int(_blackboard_get(context.get_blackboard(), f"placed_count:{robot_id}", 0))
        return NodeStatus.SUCCESS if count >= int(context.get_input("required_count")) else NodeStatus.FAILURE


@node(
    type="FrankaCheckObjectLifted",
    doc="Check that a rigid object is elevated and remains near the Franka end effector.",
    keywords=["franka", "manipulation", "grasp", "condition"],
    ports=[
        value_port("object_prim", ""),
        value_port("minimum_height", 0.15),
        value_port("maximum_grasp_distance", 0.15),
        value_port("recover_on_failure", False),
    ],
)
class FrankaCheckObjectLifted(_FrankaNodeBase):
    """Check whether a rigid object was successfully lifted."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Return success when object height and grasp distance pass."""
        robot = self._resolve_robot()
        if robot is None:
            return self._status_while_resolving()

        object_path = str(context.get_input("object_prim") or "")
        if not object_path:
            carb.log_error("[FrankaCheckObjectLifted] 'object_prim' must be a non-empty prim path")
            return NodeStatus.FAILURE

        try:
            object_position = RigidPrim(object_path).get_world_poses()[0].numpy()[0]
            end_effector_position = robot.get_tool_pose()[0][0]
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaCheckObjectLifted] Failed to read '{object_path}': {exc}")
            return NodeStatus.FAILURE

        minimum_height = float(context.get_input("minimum_height"))
        maximum_distance = float(context.get_input("maximum_grasp_distance"))
        lifted = float(object_position[2]) >= minimum_height
        retained = float(np.linalg.norm(object_position - end_effector_position)) <= maximum_distance
        if lifted and retained:
            return NodeStatus.SUCCESS
        if bool(context.get_input("recover_on_failure")):
            local = _get_local_blackboard(context)
            if _request_recovery(context, local):
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE


@node(
    type="FrankaCheckObjectAtTarget",
    doc="Check that a rigid object is within horizontal tolerance of a target prim.",
    keywords=["franka", "manipulation", "place", "condition"],
    ports=[
        value_port("object_prim", ""),
        value_port("target_prim", ""),
        value_port("horizontal_tolerance", 0.05),
        value_port("recover_on_failure", False),
    ],
)
class FrankaCheckObjectAtTarget(UsdActionNode):
    """Check whether a rigid object reached its placement target."""

    def on_tick(self, context: IBehaviorActionNodeContext) -> NodeStatus:
        """Return success when object and target XY positions are close."""
        object_path = str(context.get_input("object_prim") or "")
        target_path = str(context.get_input("target_prim") or "")
        tolerance = float(context.get_input("horizontal_tolerance"))
        if not object_path or not target_path or tolerance <= 0.0:
            carb.log_error(
                "[FrankaCheckObjectAtTarget] Object and target paths must be non-empty and tolerance positive"
            )
            return NodeStatus.FAILURE

        try:
            object_position = RigidPrim(object_path).get_world_poses()[0].numpy()[0]
            target_position = XformPrim(target_path).get_world_poses()[0].numpy()[0]
        except (IndexError, RuntimeError, ValueError) as exc:
            carb.log_error(f"[FrankaCheckObjectAtTarget] Failed to read placement state: {exc}")
            return NodeStatus.FAILURE

        horizontal_error = float(np.linalg.norm(object_position[:2] - target_position[:2]))
        if horizontal_error <= tolerance:
            return NodeStatus.SUCCESS
        if bool(context.get_input("recover_on_failure")):
            local = _get_local_blackboard(context)
            if _request_recovery(context, local):
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE
