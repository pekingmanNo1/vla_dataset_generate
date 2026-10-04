# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""UR10 bin palletizing backend with motion-generation controllers and a small FSM."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.asset.gen.conveyor import create_conveyor_belt
from isaacsim.core.experimental.objects import Capsule, DistantLight, GroundPlane, Sphere
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from isaacsim.core.experimental.utils.impl.transform import quaternion_conjugate, quaternion_multiplication
from isaacsim.robot.surface_gripper import _surface_gripper
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation.robots import get_robot_config
from isaacsim.robot_motion.schema import apply_motion_planning_api
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from transitions import Machine

__all__ = [
    "ApproachParams",
    "BinSpawner",
    "BinStackingContext",
    "BinState",
    "CloseSuctionGripper",
    "DirectJointTargetController",
    "DirectSurfaceGripper",
    "GripperCommand",
    "MotionCommand",
    "Move",
    "MoveToFlipStation",
    "OpenSuctionGripper",
    "OrientToHome",
    "PalletizerCommand",
    "PalletizerController",
    "PalletizerDecision",
    "PalletizerFSM",
    "PalletizerObservation",
    "PosePq",
    "ReachToPick",
    "ReachToPlace",
    "ReleaseFlipStationBin",
    "TestSurfaceGripper",
    "TimedLift",
    "ToolSpaceRmpFlowController",
    "Ur10",
    "Ur10Assets",
    "Ur10MotionController",
    "adjust_about_x_if_opposite",
    "build_scene",
    "create_planning_obstacles",
    "get_bin_under",
    "random_bin_spawn_transform",
    "shift_for_approach",
]

_PHYSICS_DT = 1.0 / 60.0
_PLANNING_OBSTACLE_ROOT = "/World/Ur10PalletizingPlanningObstacles"
_FLIP_STATION_OBSTACLE_PATHS = [f"{_PLANNING_OBSTACLE_ROOT}/FlipStationSphere"]
_CONVEYOR_SPEED = 0.30
_NAVIGATION_OBSTACLE_PATHS = [
    f"{_PLANNING_OBSTACLE_ROOT}/NavigationDome",
    f"{_PLANNING_OBSTACLE_ROOT}/NavigationBarrier",
    f"{_PLANNING_OBSTACLE_ROOT}/NavigationFlipStation",
]
_PLANNING_OBSTACLE_PATHS = _FLIP_STATION_OBSTACLE_PATHS + _NAVIGATION_OBSTACLE_PATHS

# ---------------------------------------------------------------------------
# Geometry helpers and motion-command types
# ---------------------------------------------------------------------------


def _read_world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a prim's world pose via USD. Returns ``(position, quat_wxyz)`` as numpy."""
    stage = stage_utils.get_current_stage(backend="usd")
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform.Orthonormalize()
    q = transform.ExtractRotationQuat()
    return (
        np.array(transform.ExtractTranslation(), dtype=np.float64),
        np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64),
    )


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to (w,x,y,z) quaternion."""
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion to 3x3 rotation matrix."""
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def _transforms_close(T1: np.ndarray, T2: np.ndarray, p_thresh: float, R_thresh: float) -> bool:
    """True if the two 4x4 homogeneous transforms agree within position and per-axis rotation thresholds."""
    diff = T1 - T2
    return bool(np.linalg.norm(diff[:3, 3]) <= p_thresh and np.linalg.norm(diff[:3, :3]) / 3 <= R_thresh)


def _pR_to_T(p: np.ndarray, R: np.ndarray) -> np.ndarray:  # noqa: N802
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


@dataclass
class PosePq:
    """End-effector target as (position, quaternion_wxyz)."""

    p: np.ndarray
    q: np.ndarray

    def to_T(self) -> np.ndarray:  # noqa: N802
        T = np.eye(4)
        T[:3, :3] = _quat_to_matrix(self.q)
        T[:3, 3] = self.p
        return T


@dataclass
class ApproachParams:
    """RBF funnel pulling the EE toward the target along ``direction``."""

    direction: np.ndarray
    std_dev: float


@dataclass
class MotionCommand:
    target_pose: PosePq
    approach_params: ApproachParams | None = None
    joint_target: np.ndarray | None = None


class GripperCommand(Enum):
    """Backend-neutral suction-gripper commands."""

    HOLD = auto()
    OPEN = auto()
    CLOSE = auto()


@dataclass(frozen=True)
class PalletizerObservation:
    """One coherent estimate consumed by the palletizing FSM."""

    time: float
    robot_fk_T: np.ndarray
    robot_home_fk_T: np.ndarray
    robot_default_config: np.ndarray
    gripper_closed: bool
    has_active_bin: bool
    active_bin_attached: bool
    active_bin_needs_flip: bool
    active_bin_grasp_T: np.ndarray | None
    active_bin_position: np.ndarray | None
    stack_target: np.ndarray | None
    bin_under_position: np.ndarray | None
    active_bin_at_stack_target: bool
    placement_completes_stack: bool


@dataclass(frozen=True)
class PalletizerDecision:
    """High-level state selected by the palletizing FSM."""

    state: str
    state_changed: bool = False
    complete_bin: bool = False


@dataclass(frozen=True)
class PalletizerCommand:
    """Controller output applied by ``BinStackingContext``."""

    motion: MotionCommand | None = None
    retain_motion: bool = False
    gripper: GripperCommand = GripperCommand.HOLD
    navigation_obstacles: bool = False
    flip_station_obstacle: bool = False
    complete_bin: bool = False
    conveyor_running: bool | None = None


def shift_for_approach(target_T: np.ndarray, eff_T: np.ndarray, approach_params: ApproachParams) -> np.ndarray:
    """Shift the target back along ``direction`` by an RBF in the EE's orthogonal distance to the approach line."""
    target_R, target_p = target_T[:3, :3], target_T[:3, 3]
    eff_R, eff_p = eff_T[:3, :3], eff_T[:3, 3]
    direction = approach_params.direction
    std_dev = approach_params.std_dev
    v = eff_p - target_p
    an = direction / np.linalg.norm(direction)
    dist = np.linalg.norm(v - np.dot(v, an) * an)
    dist += 0.5 * np.linalg.norm(target_R - eff_R) / 3
    alpha = 1.0 - np.exp(-0.5 * dist * dist / (std_dev * std_dev))
    return target_p - alpha * direction


def _approach_params(
    direction: np.ndarray,
    distance: float,
    max_length: float,
    lookahead: float,
    std_dev: float,
) -> ApproachParams | None:
    length = min(max_length, max(0.0, distance - lookahead))
    return ApproachParams(direction=length * direction, std_dev=std_dev) if length > 1.0e-6 else None


def create_planning_obstacles() -> None:
    stage_utils.define_prim(_PLANNING_OBSTACLE_ROOT, "Xform")
    Sphere(_FLIP_STATION_OBSTACLE_PATHS[0], positions=[0.73, 0.76, -0.13], radii=0.2)
    Sphere(_NAVIGATION_OBSTACLE_PATHS[0], positions=[-0.031, -0.018, -1.086], radii=1.1)

    az = np.array([1.0, 0.0, -0.3], dtype=np.float64)
    az /= np.linalg.norm(az)
    ax = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ay = np.cross(az, ax)
    Capsule(
        _NAVIGATION_OBSTACLE_PATHS[1],
        positions=[0.471, 0.276, -0.563],
        orientations=_matrix_to_quat(np.column_stack([ax, ay, az])),
        radii=0.5,
        heights=0.9,
    )
    Capsule(_NAVIGATION_OBSTACLE_PATHS[2], positions=[0.766, 0.755, -0.5], radii=0.5, heights=0.5)

    stage = stage_utils.get_current_stage(backend="usd")
    for path in _PLANNING_OBSTACLE_PATHS:
        apply_motion_planning_api(stage.GetPrimAtPath(path), enabled=False)
    XformPrim(_PLANNING_OBSTACLE_PATHS).set_visibilities(False)


# ---------------------------------------------------------------------------
# UR10 motion controllers
# ---------------------------------------------------------------------------


def _rigid_prim_path(rigid_prim: RigidPrim) -> str:
    return rigid_prim.paths[0]


def _rigid_world_pose(rigid_prim: RigidPrim) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = rigid_prim.get_world_poses()
    return np.asarray(positions.numpy()[0], dtype=np.float64), np.asarray(orientations.numpy()[0], dtype=np.float64)


def _set_rigid_world_pose(rigid_prim: RigidPrim, position: np.ndarray, orientation: np.ndarray) -> None:
    rigid_prim.set_world_poses(positions=[position], orientations=[orientation])


def _set_rigid_linear_velocity(rigid_prim: RigidPrim, velocity: list[float] | np.ndarray) -> None:
    rigid_prim.set_velocities(linear_velocities=[velocity])


class _ControllerMode(Enum):
    """Controller selections used by the UR10 motion container."""

    RMPFLOW = auto()
    JOINT_TARGET = auto()


class ToolSpaceRmpFlowController(mg.BaseController):
    """Tool-space RMPflow controller with world-binding synchronization."""

    def __init__(
        self,
        rmp_flow: RmpFlowController,
        world_binding: mg.WorldBinding,
        articulation: Articulation,
        tool_frame: str,
        reset_position_delta: float,
        reset_rotation_delta: float,
    ) -> None:
        self._rmp_flow = rmp_flow
        self._world_binding = world_binding
        self._articulation = articulation
        self._tool_frame = tool_frame
        self._reset_position_delta = reset_position_delta
        self._reset_rotation_delta = reset_rotation_delta
        self._reset_needed = True
        self._last_setpoint_p: np.ndarray | None = None
        self._last_setpoint_R: np.ndarray | None = None

    def request_reset(self) -> None:
        self._reset_needed = True

    def needs_reset(self, setpoint_state: mg.RobotState | None) -> bool:
        return self._reset_needed or self._target_discontinuous(setpoint_state)

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        self._sync_world_binding()
        if not self._rmp_flow.reset(estimated_state, setpoint_state, t, **kwargs):
            self._reset_needed = True
            return False
        self._reset_needed = False
        self._remember_setpoint(setpoint_state)
        return True

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState | None:
        if self._target_discontinuous(setpoint_state):
            self._reset_needed = True
        if self._reset_needed and not self.reset(estimated_state, setpoint_state, t, **kwargs):
            raise RuntimeError("RmpFlowController reset failed.")

        self._sync_world_binding()
        self._remember_setpoint(setpoint_state)
        return self._rmp_flow.forward(estimated_state, setpoint_state, t, **kwargs)

    def _sync_world_binding(self) -> None:
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(
            self._articulation.get_world_poses()
        )
        self._world_binding.synchronize_transforms()

    def _remember_setpoint(self, setpoint_state: mg.RobotState | None) -> None:
        pose = self._get_tool_setpoint_pose(setpoint_state)
        if pose is None:
            return
        self._last_setpoint_p, self._last_setpoint_R = pose

    def _target_discontinuous(self, setpoint_state: mg.RobotState | None) -> bool:
        pose = self._get_tool_setpoint_pose(setpoint_state)
        if pose is None:
            return False
        position, rotation = pose
        if self._last_setpoint_p is None or self._last_setpoint_R is None:
            return True
        p_delta = np.linalg.norm(position - self._last_setpoint_p)
        r_delta = np.linalg.norm(rotation - self._last_setpoint_R) / 3
        return bool(p_delta > self._reset_position_delta or r_delta > self._reset_rotation_delta)

    def _get_tool_setpoint_pose(self, setpoint_state: mg.RobotState | None) -> tuple[np.ndarray, np.ndarray] | None:
        if setpoint_state is None or setpoint_state.sites is None:
            return None
        sites = setpoint_state.sites
        if self._tool_frame not in sites.position_names or self._tool_frame not in sites.orientation_names:
            return None
        if sites.positions is None or sites.orientations is None:
            return None
        position = np.asarray(sites.positions.numpy()[sites.position_names.index(self._tool_frame)], dtype=np.float64)
        orientation = np.asarray(
            sites.orientations.numpy()[sites.orientation_names.index(self._tool_frame)], dtype=np.float64
        )
        return position.copy(), _quat_to_matrix(orientation)


class DirectJointTargetController(mg.BaseController):
    """Joint-space controller that advances arm targets by a clipped delta."""

    def __init__(self, robot_joint_space: list[str], controlled_joint_names: list[str], delta_q_max: float) -> None:
        self._robot_joint_space = robot_joint_space
        self._controlled_joint_names = controlled_joint_names
        self._delta_q_max = delta_q_max

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        return self._positions_for_names(estimated_state.joints, self._controlled_joint_names) is not None

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState | None:
        if setpoint_state is None:
            return None

        current_q = self._positions_for_names(estimated_state.joints, self._controlled_joint_names)
        target_q = self._positions_for_names(setpoint_state.joints, self._controlled_joint_names)
        if current_q is None or target_q is None:
            return None

        delta_q = self._clip_delta_q(target_q - current_q)
        dt = max(float(kwargs.get("dt", _PHYSICS_DT)), 1.0e-6)
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._robot_joint_space,
                positions=(self._controlled_joint_names, wp.array((current_q + delta_q).tolist(), dtype=wp.float32)),
                velocities=(self._controlled_joint_names, wp.array((delta_q / dt).tolist(), dtype=wp.float32)),
            )
        )

    def _clip_delta_q(self, delta_q: np.ndarray) -> np.ndarray:
        max_abs = float(np.max(np.abs(delta_q)))
        return delta_q * (self._delta_q_max / max_abs) if max_abs > self._delta_q_max else delta_q

    @staticmethod
    def _positions_for_names(joints: mg.JointState | None, joint_names: list[str]) -> np.ndarray | None:
        if joints is None or joints.positions is None:
            return None
        if not set(joint_names).issubset(joints.position_names):
            return None
        positions = joints.positions.numpy()
        return np.asarray([positions[joints.position_names.index(name)] for name in joint_names], dtype=np.float64)


class Ur10MotionController:
    """UR10 motion backend that dispatches between RMPflow and direct joint targets."""

    _HOME_ARM_CONFIG = np.array([-1.57, -1.57, -1.57, -1.57, 1.57, 0.0])
    _SUCTION_TIP_LOCAL_OFFSET = np.array([0.17, 0.0, 0.0], dtype=np.float64)
    _NUM_ARM_DOFS = 6
    _DELTA_Q_MAX = 0.05
    _RESET_POSITION_DELTA = 0.12
    _RESET_ROTATION_DELTA = 0.2
    _EE_LINK_FROM_WRIST = Rotation.from_euler("z", np.pi / 2.0).as_quat()[[3, 0, 1, 2]]
    _TOOL0_FROM_WRIST = Rotation.from_euler("x", -np.pi / 2.0).as_quat()[[3, 0, 1, 2]]

    def __init__(self) -> None:
        self._articulation: Articulation | None = None
        self._exp_end_effector: RigidPrim | None = None
        self._world_binding: mg.WorldBinding | None = None
        self._selectable_controller: mg.SelectableController | None = None
        self._tool_space_controller: ToolSpaceRmpFlowController | None = None
        self._site_space: list[str] | None = None
        self._tool_frame: str | None = None
        self._controller_time = 0.0
        self._home_fk_T: np.ndarray | None = None
        self._planning_obstacle_enabled: dict[str, bool] = {}

    def initialize(self, articulation: Articulation, asset_root: str) -> None:
        self._articulation = articulation
        self._exp_end_effector = RigidPrim(f"{asset_root}/ee_link")
        self._build_world_binding()
        self._build_controller()

    def reset(self) -> None:
        self._controller_time = 0.0
        self._home_fk_T = None
        if self._tool_space_controller is not None:
            self._tool_space_controller.request_reset()
        if self._selectable_controller is not None:
            self._selectable_controller.set_next_controller(_ControllerMode.RMPFLOW)

    @property
    def default_config(self) -> np.ndarray:
        if self._articulation is None:
            raise RuntimeError("Ur10MotionController.initialize must be called first")
        q = np.zeros(self._articulation.num_dofs)
        q[: self._NUM_ARM_DOFS] = self._HOME_ARM_CONFIG
        return q

    def get_fk_T(self) -> np.ndarray:  # noqa: N802
        """Cup-tip pose: ee_link orientation + position shifted along ee_link's +X."""
        positions, orientations = self._exp_end_effector.get_world_poses()
        p_wrist = np.asarray(positions.numpy()[0], dtype=np.float64)
        q_wrist = np.asarray(orientations.numpy()[0], dtype=np.float64)
        R = _quat_to_matrix(q_wrist)
        return _pR_to_T(p_wrist + R @ self._SUCTION_TIP_LOCAL_OFFSET, R)

    @property
    def home_fk_T(self) -> np.ndarray:  # noqa: N802
        if self._home_fk_T is None:
            raise RuntimeError("Home FK not yet captured; needs at least one step() call.")
        return self._home_fk_T

    def step(self, command: MotionCommand | None, dt: float) -> None:
        if self._articulation is None:
            return

        self._capture_home_fk()
        if command is None:
            return
        if command.joint_target is not None:
            self._select_controller(_ControllerMode.JOINT_TARGET)
            self._step_controller(self._estimated_state(), self._joint_setpoint_state(command.joint_target), dt)
            return

        target_p_tool, target_q_tool = self._tool_target_from_command(command)
        self._select_controller(_ControllerMode.RMPFLOW)
        self._step_controller(self._estimated_state(), self._site_setpoint_state(target_p_tool, target_q_tool), dt)

    def set_planning_obstacles_enabled(self, paths: list[str], enabled: bool) -> None:
        if self._world_binding is None:
            return
        changed = [path for path in paths if self._planning_obstacle_enabled.get(path) != enabled]
        if not changed:
            return
        self._world_binding.get_world_interface().update_obstacle_enables(
            changed, wp.array([enabled] * len(changed), dtype=wp.bool)
        )
        for path in changed:
            self._planning_obstacle_enabled[path] = enabled

    def _capture_home_fk(self) -> None:
        if self._home_fk_T is None:
            self._home_fk_T = self.get_fk_T()

    def _tool_target_from_command(self, command: MotionCommand) -> tuple[np.ndarray, np.ndarray]:
        # Commands target the cup tip; cuMotion controls tool0.
        target_p_cup = command.target_pose.p
        target_q_ee = command.target_pose.q
        if command.approach_params is not None:
            target_p_cup = shift_for_approach(command.target_pose.to_T(), self.get_fk_T(), command.approach_params)

        target_R_ee = _quat_to_matrix(target_q_ee)
        target_p_tool = target_p_cup - target_R_ee @ self._SUCTION_TIP_LOCAL_OFFSET
        target_q_tool = self._get_tool_orientation_from_ee_orientation(target_q_ee)
        return target_p_tool, target_q_tool

    def _step_controller(self, estimated: mg.RobotState, setpoint: mg.RobotState, dt: float) -> None:
        controller_time = self._controller_time
        active_controller = self._selectable_controller.get_active_controller_enum()
        if active_controller == _ControllerMode.RMPFLOW and self._tool_space_controller.needs_reset(setpoint):
            controller_time = 0.0
            self._controller_time = 0.0

        desired = self._selectable_controller.forward(estimated, setpoint, controller_time, dt=dt)
        if self._selectable_controller.get_active_controller_enum() == _ControllerMode.RMPFLOW:
            self._controller_time += dt
        if desired is None or desired.joints is None:
            return

        self._apply_desired_joints(desired.joints)

    def _select_controller(self, mode: _ControllerMode) -> None:
        active_controller = self._selectable_controller.get_active_controller_enum()
        self._selectable_controller.set_next_controller(mode)
        if active_controller == mode:
            return
        if mode == _ControllerMode.RMPFLOW:
            self._controller_time = 0.0
            self._tool_space_controller.request_reset()

    @staticmethod
    def _joint_indices(indices: Any) -> Any:
        if indices is None:
            return None
        if hasattr(indices, "numpy"):
            return indices.numpy().flatten().astype(np.int64).tolist()
        return np.asarray(indices, dtype=np.int64).flatten().tolist()

    def _apply_desired_joints(self, joints: Any) -> None:
        if joints.positions is not None:
            self._articulation.set_dof_position_targets(
                joints.positions, dof_indices=self._joint_indices(joints.position_indices)
            )
        if joints.velocities is not None:
            self._articulation.set_dof_velocity_targets(
                joints.velocities, dof_indices=self._joint_indices(joints.velocity_indices)
            )
        if joints.efforts is not None:
            self._articulation.set_dof_efforts(joints.efforts, dof_indices=self._joint_indices(joints.effort_indices))

    def _build_world_binding(self) -> None:
        robot_pos, robot_ori = self._articulation.get_world_poses()
        obstacle_strategy = mg.ObstacleStrategy()

        self._world_binding = mg.WorldBinding(
            world_interface=CumotionWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=_PLANNING_OBSTACLE_PATHS,
            tracked_collision_api=mg.TrackableApi.MOTION_GENERATION_COLLISION,
        )
        self._world_binding.initialize()
        self._planning_obstacle_enabled = {path: False for path in _PLANNING_OBSTACLE_PATHS}
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
        self._world_binding.synchronize_transforms()

    def _build_controller(self) -> None:
        cumotion_robot = load_cumotion_supported_robot("ur10")
        self._site_space = cumotion_robot.robot_description.tool_frame_names()
        if not self._site_space:
            raise RuntimeError("No cuMotion tool frames found for UR10.")
        self._tool_frame = self._site_space[0]
        rmp_flow_controller = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=self._world_binding.get_world_interface(),
            robot_joint_space=self._articulation.dof_names,
            robot_site_space=self._site_space,
            tool_frame=self._tool_frame,
        )
        rmp_flow_controller.get_rmp_flow_config().set_param("cspace_target_rmp/metric_scalar", 1.0)
        self._tool_space_controller = ToolSpaceRmpFlowController(
            rmp_flow=rmp_flow_controller,
            world_binding=self._world_binding,
            articulation=self._articulation,
            tool_frame=self._tool_frame,
            reset_position_delta=self._RESET_POSITION_DELTA,
            reset_rotation_delta=self._RESET_ROTATION_DELTA,
        )
        direct_joint_controller = DirectJointTargetController(
            robot_joint_space=self._articulation.dof_names,
            controlled_joint_names=self._articulation.dof_names[: self._NUM_ARM_DOFS],
            delta_q_max=self._DELTA_Q_MAX,
        )
        self._selectable_controller = mg.SelectableController(
            controller_options={
                _ControllerMode.RMPFLOW: self._tool_space_controller,
                _ControllerMode.JOINT_TARGET: direct_joint_controller,
            },
            initial_controller_selection=_ControllerMode.RMPFLOW,
        )

    def _estimated_state(self) -> mg.RobotState:
        names = self._articulation.dof_names
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self._articulation.get_dof_positions()),
                velocities=(names, self._articulation.get_dof_velocities()),
            )
        )

    def _site_setpoint_state(self, position: np.ndarray, orientation: np.ndarray) -> mg.RobotState:
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._tool_frame], wp.array([position.tolist()], dtype=wp.float32)),
                orientations=([self._tool_frame], wp.array([orientation.tolist()], dtype=wp.float32)),
            )
        )

    def _joint_setpoint_state(self, joint_target: np.ndarray) -> mg.RobotState:
        joint_target = np.asarray(joint_target, dtype=np.float64)
        joint_names = self._articulation.dof_names[: self._NUM_ARM_DOFS]
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._articulation.dof_names,
                positions=(joint_names, wp.array(joint_target[: self._NUM_ARM_DOFS].tolist(), dtype=wp.float32)),
            )
        )

    @staticmethod
    def _get_tool_orientation_from_ee_orientation(ee_orientation: np.ndarray) -> np.ndarray:
        """Convert desired USD ee_link orientation to cuMotion tool0 orientation."""
        wrist_from_ee_link = quaternion_conjugate(
            wp.array([Ur10MotionController._EE_LINK_FROM_WRIST], dtype=wp.float32)
        )
        ee_orientation_wp = wp.array([ee_orientation.tolist()], dtype=wp.float32)
        tool0_from_wrist_wp = wp.array([Ur10MotionController._TOOL0_FROM_WRIST], dtype=wp.float32)
        tool_orientation = quaternion_multiplication(
            quaternion_multiplication(ee_orientation_wp, wrist_from_ee_link), tool0_from_wrist_wp
        ).numpy()[0]
        return tool_orientation / np.linalg.norm(tool_orientation)


# ---------------------------------------------------------------------------
# Robot, assets, and bin spawning
# ---------------------------------------------------------------------------


class DirectSurfaceGripper:
    """Thin wrapper around the SurfaceGripper plugin interface used by the suction examples."""

    def __init__(self, surface_gripper_path: str) -> None:
        self.surface_gripper_path = surface_gripper_path
        self._iface = None
        self._status_enum = None

    def initialize(self) -> None:
        self._iface = _surface_gripper.acquire_surface_gripper_interface()
        self._status_enum = _surface_gripper.GripperStatus
        self.open()

    def post_reset(self) -> None:
        self.open()

    def open(self) -> None:
        if self._iface is not None:
            self._iface.open_gripper(self.surface_gripper_path)

    def close(self) -> None:
        if self._iface is not None:
            self._iface.close_gripper(self.surface_gripper_path)

    def is_closed(self) -> bool:
        if self._iface is None:
            return False
        status = self._iface.get_gripper_status(self.surface_gripper_path)
        if self._status_enum is not None:
            return bool(status == self._status_enum.Closed)
        return "Closed" in str(status)


class TestSurfaceGripper:
    def __init__(self, _surface_gripper_path: str) -> None:
        self._closed = False

    def initialize(self) -> None:
        self.open()

    def post_reset(self) -> None:
        self.open()

    def open(self) -> None:
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed


class Ur10:
    """UR10 with a surface gripper at ``ee_link/SurfaceGripper`` and a motion backend."""

    def __init__(self, prim_path: str, use_test_gripper: bool = False) -> None:
        self.prim_path = prim_path
        self.articulation: Articulation | None = None
        self.motion_generator = Ur10MotionController()
        end_effector_prim_path = prim_path + "/ee_link"
        gripper_cls = TestSurfaceGripper if use_test_gripper else DirectSurfaceGripper
        self.suction_gripper = gripper_cls(end_effector_prim_path + "/SurfaceGripper")
        self.uses_test_gripper = use_test_gripper
        self._latest_command: MotionCommand | None = None
        self._gripper_initialized = False

    def setup(self) -> None:
        if self.articulation is not None:
            return
        self.articulation = Articulation(self.prim_path)
        self.articulation.set_link_enabled_gravities([False])
        self.motion_generator.initialize(self.articulation, self.prim_path)
        self.articulation.set_default_state(dof_positions=[self.default_config])

    def initialize(self) -> None:
        self.setup()
        if not self._gripper_initialized:
            self.suction_gripper.initialize()
            self._gripper_initialized = True

    def reset(self) -> None:
        if self.articulation is not None:
            default_config = self.default_config
            zero_velocities = np.zeros_like(default_config)
            self.articulation.set_dof_positions(default_config.tolist())
            self.articulation.set_dof_position_targets(default_config.tolist())
            self.articulation.set_dof_velocities(zero_velocities.tolist())
            self.articulation.set_dof_velocity_targets(zero_velocities.tolist())
            self.motion_generator.reset()
        self._latest_command = None
        self.suction_gripper.post_reset()

    @property
    def default_config(self) -> np.ndarray:
        return self.motion_generator.default_config

    def get_fk_T(self) -> np.ndarray:  # noqa: N802
        return self.motion_generator.get_fk_T()

    @property
    def home_fk_T(self) -> np.ndarray:  # noqa: N802
        return self.motion_generator.home_fk_T

    def send(self, command: MotionCommand | None) -> None:
        self._latest_command = command

    def step(self, dt: float) -> None:
        self.motion_generator.step(self._latest_command, dt)

    def set_planning_obstacles_enabled(self, paths: list[str], enabled: bool) -> None:
        self.motion_generator.set_planning_obstacles_enabled(paths, enabled)


class Ur10Assets:
    """USD asset paths for the UR10 bin-stacking scene."""

    def __init__(self) -> None:
        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder")
        self.conveyor_usd = self.assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A05.usd"
        self.flip_stack_usd = self.assets_root_path + "/Isaac/Props/Flip_Stack/flip_stack.usd"
        self.pallet_usd = self.assets_root_path + "/Isaac/Props/Pallet/pallet.usd"
        self.dolly_usd = self.assets_root_path + "/Isaac/Props/Dolly/dolly.usd"
        self.ur10_mount_usd = self.assets_root_path + "/Isaac/Props/Mounts/ur10_mount.usd"
        self.sortbot_housing_usd = self.assets_root_path + "/Isaac/Props/Sortbot_Housing/sortbot_housing.usd"
        ur10_config = get_robot_config("ur10")
        self.ur10_usd = self.assets_root_path + ur10_config.asset_path
        self.ur10_variants = list(ur10_config.variants)
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"


def random_bin_spawn_transform() -> tuple[np.ndarray, np.ndarray]:
    """Random spawn position + orientation for a bin (50% chance upside-down)."""
    position = np.array([random.uniform(-0.15, 0.15), 1.5, -0.15])
    rot = Rotation.from_euler("z", random.uniform(-np.pi, np.pi))
    if random.random() > 0.5:
        rot = rot * Rotation.from_euler("y", np.pi)
        print("<flip>")
    else:
        print("<no flip>")
    return position, rot.as_quat()[[3, 0, 1, 2]]


class BinSpawner:
    """Spawn and track bins on the conveyor."""

    def __init__(self, env_path: str, assets: Ur10Assets) -> None:
        self.assets = assets
        self.env_path = env_path
        self.bins: list[RigidPrim] = []
        self.on_conveyor: RigidPrim | None = None

    def _spawn_bin(self, rigid_bin: RigidPrim, position: np.ndarray, orientation: np.ndarray) -> None:
        _set_rigid_world_pose(rigid_bin, position, orientation)
        _set_rigid_linear_velocity(rigid_bin, [0.0, -0.30, 0.0])
        rigid_bin.set_visibilities(True)

    def reset(self) -> None:
        stage = stage_utils.get_current_stage(backend="usd")
        for rigid_bin in self.bins:
            prim_path = _rigid_prim_path(rigid_bin)
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage_utils.delete_prim(prim_path)
        self.bins.clear()
        self.on_conveyor = None

    def step(self) -> None:
        if self.on_conveyor is not None:
            x, y, _ = _rigid_world_pose(self.on_conveyor)[0]
            if y > 0.0 and -0.4 < x < 0.4:
                return

        name = f"bin_{len(self.bins)}"
        prim_path = self.env_path + f"/bins/{name}"
        x, q = random_bin_spawn_transform()
        stage_utils.add_reference_to_stage(usd_path=self.assets.small_klt_usd, path=prim_path)
        self.on_conveyor = RigidPrim(prim_path, positions=[x], orientations=[q], reset_xform_op_properties=True)
        self._spawn_bin(self.on_conveyor, x, q)
        self.bins.append(self.on_conveyor)


# ---------------------------------------------------------------------------
# Bin-stacking state
# ---------------------------------------------------------------------------


@dataclass
class BinState:
    bin_obj: RigidPrim
    grasp_T: np.ndarray | None = None
    is_attached: bool | None = None
    needs_flip: bool | None = None
    root_from_grasp_T: np.ndarray | None = None

    @property
    def bin_base_prim_path(self) -> str:
        # Grasp anchors to the collision sub-prim, not the rigid root.
        return _rigid_prim_path(self.bin_obj) + "/Collision/Cube_03"


def get_bin_under(p: np.ndarray, stacked_bins: list) -> Any | None:
    x, y, _ = p
    xy = np.array([x, y])
    for b in reversed(stacked_bins):
        bin_x, bin_y, _bin_z = _rigid_world_pose(b.bin_obj)[0]
        bin_xy = np.array([bin_x, bin_y])
        if np.linalg.norm(bin_xy - xy) < 0.05:
            return b
    return None


def adjust_about_x_if_opposite(eff_R: np.ndarray, target_R: np.ndarray, threshold: float = -0.9) -> np.ndarray:
    if np.dot(target_R[:3, 0], eff_R[:3, 0]) < threshold:
        return target_R @ np.diag([1, -1, -1])
    return target_R


class BinStackingContext:
    _BIN_HEIGHT = 0.135
    _STACK_PAD = 0.0075
    _STACK_X_SHIFT = 0.05

    def __init__(self, robot: Ur10, spawner: BinSpawner, max_bins: int | None = None) -> None:
        self.robot = robot
        self.spawner = spawner
        self.env_path = spawner.env_path

        stack_xs = np.array([1.00, 0.79, 0.58]) + self._STACK_X_SHIFT
        stack_ys = [-0.62, -0.31, 0]
        stack_z = -0.59374 + self._BIN_HEIGHT / 2 + self._STACK_PAD
        self.stack_coordinates = [np.array([x, y, stack_z]) for y in stack_ys for x in stack_xs]
        if max_bins is not None:
            self.stack_coordinates = self.stack_coordinates[:max_bins]

        self.bins: list[BinState] = []
        self.active_bin: BinState | None = None
        self.stacked_bins: list[BinState] = []
        self._latest_observation: PalletizerObservation | None = None
        self._conveyor_running: bool | None = None

        self.monitors = [
            BinStackingContext.monitor_bins,
            BinStackingContext.monitor_active_bin,
            BinStackingContext.monitor_active_bin_grasp_T,
            BinStackingContext.monitor_active_bin_grasp_reached,
        ]
        if self.robot.uses_test_gripper:
            self.monitors.append(BinStackingContext.monitor_test_attached_bin)

    def reset(self) -> None:
        self.bins.clear()
        self.active_bin = None
        self.stacked_bins.clear()
        self._latest_observation = None
        self._set_conveyor_running(True)
        self.disable_planning_obstacles()

    def read(self, time: float) -> PalletizerObservation:
        """Update scene estimates and return one backend-neutral snapshot."""
        for monitor in self.monitors:
            monitor(self)

        robot_fk_T = self.robot.get_fk_T().copy()
        try:
            robot_home_fk_T = self.robot.home_fk_T.copy()
        except RuntimeError:
            robot_home_fk_T = robot_fk_T.copy()

        active_bin_position = None
        active_bin_grasp_T = None
        active_bin_attached = False
        active_bin_needs_flip = False
        if self.active_bin is not None:
            active_bin_position = _rigid_world_pose(self.active_bin.bin_obj)[0].copy()
            active_bin_grasp_T = None if self.active_bin.grasp_T is None else self.active_bin.grasp_T.copy()
            active_bin_attached = bool(self.active_bin.is_attached)
            active_bin_needs_flip = bool(self.active_bin.needs_flip)

        stack_target = None
        bin_under_position = None
        if len(self.stacked_bins) < len(self.stack_coordinates):
            stack_target = self.stack_coordinates[len(self.stacked_bins)].copy()
            bin_under = get_bin_under(stack_target, self.stacked_bins)
            if bin_under is not None:
                bin_under_position = _rigid_world_pose(bin_under.bin_obj)[0].copy()

        observation = PalletizerObservation(
            time=time,
            robot_fk_T=robot_fk_T,
            robot_home_fk_T=robot_home_fk_T,
            robot_default_config=self.robot.default_config.copy(),
            gripper_closed=self.robot.suction_gripper.is_closed(),
            has_active_bin=self.has_active_bin,
            active_bin_attached=active_bin_attached,
            active_bin_needs_flip=active_bin_needs_flip,
            active_bin_grasp_T=active_bin_grasp_T,
            active_bin_position=active_bin_position,
            stack_target=stack_target,
            bin_under_position=bin_under_position,
            active_bin_at_stack_target=self.active_bin_at_stack_target(),
            placement_completes_stack=len(self.stacked_bins) + 1 >= len(self.stack_coordinates),
        )
        self._latest_observation = observation
        return observation

    def apply(self, command: PalletizerCommand, dt: float) -> None:
        """Apply one FSM output to the scene and robot backend."""
        if self._latest_observation is None:
            raise RuntimeError("Call read() before apply().")
        if command.motion is None:
            self.disable_planning_obstacles()
            if not command.retain_motion:
                self.robot.send(None)
        else:
            self.update_planning_obstacles(
                command.motion.target_pose.p,
                self._latest_observation,
                navigation=command.navigation_obstacles,
                flip_station=command.flip_station_obstacle,
            )
            self.robot.send(command.motion)

        if command.gripper is GripperCommand.OPEN:
            self.robot.suction_gripper.open()
        elif command.gripper is GripperCommand.CLOSE:
            self.robot.suction_gripper.close()
        if command.conveyor_running is not None:
            self._set_conveyor_running(command.conveyor_running)
            if not command.conveyor_running and self.active_bin is not None:
                _set_rigid_linear_velocity(self.active_bin.bin_obj, np.zeros(3))
        if command.complete_bin:
            self.mark_active_bin_as_complete()
        self.robot.step(dt)
        self._latest_observation = None

    def disable_planning_obstacles(self) -> None:
        self.robot.set_planning_obstacles_enabled(_PLANNING_OBSTACLE_PATHS, False)

    def update_planning_obstacles(
        self,
        target_p: np.ndarray,
        observation: PalletizerObservation,
        *,
        navigation: bool = False,
        flip_station: bool = False,
    ) -> None:
        self.robot.set_planning_obstacles_enabled(
            _NAVIGATION_OBSTACLE_PATHS,
            navigation and self._navigation_obstacles_required(target_p, observation),
        )
        self.robot.set_planning_obstacles_enabled(
            _FLIP_STATION_OBSTACLE_PATHS,
            flip_station and self._flip_station_obstacle_required(observation),
        )

    def _navigation_obstacles_required(self, target_p: np.ndarray, observation: PalletizerObservation) -> bool:
        ref_p = np.array([0.6, 0.37, 0.0])
        eff_p = observation.robot_fk_T[:3, 3].copy()
        target_p = np.asarray(target_p, dtype=np.float64).copy()
        eff_p[2] = 0.0
        target_p[2] = 0.0
        return bool(np.sign(np.cross(target_p, ref_p)[2]) * np.sign(np.cross(eff_p, ref_p)[2]) < 0.0)

    def _flip_station_obstacle_required(self, observation: PalletizerObservation) -> bool:
        if observation.active_bin_grasp_T is None:
            return True
        eff_T = observation.robot_fk_T
        grasp_p = observation.active_bin_grasp_T[:3, 3]
        grasp_ax = observation.active_bin_grasp_T[:3, 0]
        v = eff_T[:3, 3] - grasp_p
        dist = v.dot(grasp_ax)
        orth_dist = np.linalg.norm(v - dist * grasp_ax)
        return not (dist < 0.02 and grasp_ax.dot(eff_T[:3, 0]) > 0.75 and orth_dist < 0.03)

    @property
    def stack_complete(self) -> bool:
        return len(self.stacked_bins) == len(self.stack_coordinates)

    @property
    def has_active_bin(self) -> bool:
        return self.active_bin is not None

    def monitor_bins(self) -> None:
        if self.active_bin is not None:
            return
        for bin_obj in self.spawner.bins[len(self.bins) :]:
            self.bins.append(BinState(bin_obj))
        on_conveyor = []
        for bs in self.bins:
            x, y, _ = _rigid_world_pose(bs.bin_obj)[0]
            if 0.0 < y < 0.7 and -0.4 < x < 0.4:
                on_conveyor.append((y, bs))
        if on_conveyor:
            self.active_bin = min(on_conveyor, key=lambda t: t[0])[1]

    def monitor_active_bin(self) -> None:
        if self.active_bin is None:
            self._set_conveyor_running(True)
            return
        p, _ = _rigid_world_pose(self.active_bin.bin_obj)
        if p[2] < -1.0:
            self.active_bin = None
            self._set_conveyor_running(True)
            return
        if self.active_bin.is_attached:
            self._set_conveyor_running(True)

    def _set_conveyor_running(self, running: bool) -> None:
        running = bool(running)
        if running == self._conveyor_running:
            return
        stage = stage_utils.get_current_stage(backend="usd")
        graph = stage.GetPrimAtPath(f"{self.env_path}/conveyor/ConveyorBeltGraph")
        velocity = graph.GetAttribute("graph:variable:Velocity")
        if not velocity.IsValid():
            return
        velocity.Set(_CONVEYOR_SPEED if running else 0.0)
        self._conveyor_running = running

    def monitor_active_bin_grasp_T(self) -> None:  # noqa: N802
        if self.active_bin is None:
            return
        bin_p, bin_q = _read_world_pose(self.active_bin.bin_base_prim_path)
        bin_R = _quat_to_matrix(bin_q)
        bin_ax, bin_az = bin_R[:, 0], bin_R[:, 2]

        # When the bin is in the gripper, gravity gives a misleading "up". Use the
        # current EE -X (the cup's outward axis) as the "which way is up" reference.
        if self.active_bin.is_attached:
            up_vec = -self.robot.get_fk_T()[:3, 0]
        else:
            up_vec = np.array([0.0, 0.0, 1.0])

        self.active_bin.needs_flip = up_vec.dot(bin_az) > 0.0
        if self.active_bin.needs_flip:
            # Open upward; flip before stacking.
            target_ax = -bin_az
            margin = 0.0025
        else:
            # Open downward; stack directly.
            target_ax = bin_az
            margin = -0.0025
        target_ay = -bin_ax if bin_ax[1] < 0.0 else bin_ax
        target_az = np.cross(target_ax, target_ay)
        self.active_bin.grasp_T = _pR_to_T(
            bin_p + margin * bin_az,
            np.column_stack([target_ax, target_ay, target_az]),
        )

    def monitor_active_bin_grasp_reached(self) -> None:
        if self.has_active_bin:
            fk_T = self.robot.get_fk_T()
            was_attached = bool(self.active_bin.is_attached)
            self.active_bin.is_attached = (
                _transforms_close(self.active_bin.grasp_T, fk_T, 0.1, 1.0) and self.robot.suction_gripper.is_closed()
            )
            if self.active_bin.is_attached and not was_attached:
                root_p, root_q = _rigid_world_pose(self.active_bin.bin_obj)
                root_T = _pR_to_T(np.asarray(root_p, dtype=np.float64), _quat_to_matrix(root_q))
                self.active_bin.root_from_grasp_T = np.linalg.inv(self.active_bin.grasp_T) @ root_T
            elif not self.active_bin.is_attached:
                self.active_bin.root_from_grasp_T = None

    def monitor_test_attached_bin(self) -> None:
        bs = self.active_bin
        if bs is None or not bs.is_attached or bs.root_from_grasp_T is None:
            return
        root_T = self.robot.get_fk_T() @ bs.root_from_grasp_T
        _set_rigid_world_pose(bs.bin_obj, root_T[:3, 3], _matrix_to_quat(root_T[:3, :3]))
        _set_rigid_linear_velocity(bs.bin_obj, np.zeros(3))

    def active_bin_at_stack_target(self) -> bool:
        if self.active_bin is None or len(self.stacked_bins) >= len(self.stack_coordinates):
            return False
        p, _ = _rigid_world_pose(self.active_bin.bin_obj)
        p = np.asarray(p, dtype=np.float64)
        target = self.stack_coordinates[len(self.stacked_bins)]
        return np.linalg.norm(p[:2] - target[:2]) < 0.12 and abs(p[2] - target[2]) < 0.16

    def mark_active_bin_as_complete(self) -> None:
        if self.active_bin is None:
            raise RuntimeError("No active bin to complete.")
        self.stacked_bins.append(self.active_bin)
        self.active_bin = None


# ---------------------------------------------------------------------------
# Controller substeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubStateResult:
    done: bool
    command: PalletizerCommand = PalletizerCommand()


class _SubState:
    """Backend-neutral controller substep."""

    def enter(self, observation: PalletizerObservation) -> None:
        pass

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        return _SubStateResult(True)


class _Wait(_SubState):
    def __init__(self, wait_time: float, pause_conveyor_on_done: bool = False) -> None:
        self._wait_time = wait_time
        self._pause_conveyor_on_done = pause_conveyor_on_done
        self._started = 0.0

    def enter(self, observation: PalletizerObservation) -> None:
        self._started = observation.time

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        done = observation.time - self._started >= self._wait_time
        return _SubStateResult(
            done,
            PalletizerCommand(
                retain_motion=True,
                conveyor_running=False if done and self._pause_conveyor_on_done else None,
            ),
        )


class Move(_SubState):
    _USE_NAVIGATION_OBSTACLES = False
    _USE_FLIP_STATION_OBSTACLE = False

    def __init__(self, p_thresh: float, R_thresh: float) -> None:
        self.p_thresh = p_thresh
        self.R_thresh = R_thresh

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        raise NotImplementedError

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        motion = self.get_motion(observation)
        done = _transforms_close(motion.target_pose.to_T(), observation.robot_fk_T, self.p_thresh, self.R_thresh)
        return _SubStateResult(
            done,
            PalletizerCommand(
                motion=motion,
                navigation_obstacles=self._USE_NAVIGATION_OBSTACLES,
                flip_station_obstacle=self._USE_FLIP_STATION_OBSTACLE,
            ),
        )


class ReachToPick(Move):
    _USE_NAVIGATION_OBSTACLES = True
    _USE_FLIP_STATION_OBSTACLE = True
    _FINAL_APPROACH_LOOKAHEAD = 0.08
    _OPEN_TOP_LATERAL_THRESH = 0.03

    def __init__(self) -> None:
        super().__init__(p_thresh=0.005, R_thresh=2.0)

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        grasp_T = observation.active_bin_grasp_T
        if grasp_T is None:
            raise RuntimeError("Active bin has no grasp estimate.")
        R, p = grasp_T[:3, :3], grasp_T[:3, 3]
        ax = R[:, 0]
        fk_T = observation.robot_fk_T
        R = adjust_about_x_if_opposite(fk_T[:3, :3], R)
        max_approach_length = 0.3 if observation.active_bin_needs_flip else 0.1
        distance_to_grasp = np.linalg.norm(p - fk_T[:3, 3])
        if observation.active_bin_needs_flip:
            v = fk_T[:3, 3] - p
            lateral_dist = np.linalg.norm(v - np.dot(v, ax) * ax)
            approach_params = (
                ApproachParams(max_approach_length * ax, 0.005)
                if lateral_dist > self._OPEN_TOP_LATERAL_THRESH
                else _approach_params(ax, distance_to_grasp, max_approach_length, self._FINAL_APPROACH_LOOKAHEAD, 0.005)
            )
        else:
            approach_params = _approach_params(
                ax, distance_to_grasp, max_approach_length, self._FINAL_APPROACH_LOOKAHEAD, 0.005
            )
        return MotionCommand(PosePq(p, _matrix_to_quat(R)), approach_params)


class ReachToPlace(Move):
    _USE_NAVIGATION_OBSTACLES = True
    _FINAL_APPROACH_LOOKAHEAD = 0.08

    def __init__(self) -> None:
        super().__init__(p_thresh=0.02, R_thresh=2.0)
        self._base_target_p: np.ndarray | None = None
        self._target_p: np.ndarray | None = None
        target_ax = np.array([0.0, 0.0, -1.0])
        target_az = np.array([0.0, -1.0, 0.0])
        self._target_R = np.column_stack([target_ax, np.cross(target_az, target_ax), target_az])

    def enter(self, observation: PalletizerObservation) -> None:
        if observation.stack_target is None:
            raise RuntimeError("No remaining stack target.")
        self._base_target_p = observation.stack_target.copy()
        self._target_p = self._base_target_p.copy()

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        if observation.bin_under_position is not None and observation.active_bin_position is not None:
            xy_err = observation.bin_under_position[:2] - observation.active_bin_position[:2]
            if np.linalg.norm(xy_err) < 0.02:
                self._target_p[:2] = self._base_target_p[:2] + 0.1 * xy_err
        fk_T = observation.robot_fk_T
        self._target_R = adjust_about_x_if_opposite(fk_T[:3, :3], self._target_R)
        approach_params = _approach_params(
            np.array([0.0, 0.0, -1.0]),
            np.linalg.norm(self._target_p - fk_T[:3, 3]),
            0.35,
            self._FINAL_APPROACH_LOOKAHEAD,
            0.005,
        )
        return MotionCommand(PosePq(self._target_p, _matrix_to_quat(self._target_R)), approach_params)


class CloseSuctionGripper(_SubState):
    def enter(self, observation: PalletizerObservation) -> None:
        print("<close gripper>")

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        return _SubStateResult(
            observation.gripper_closed,
            PalletizerCommand(retain_motion=True, gripper=GripperCommand.CLOSE),
        )


class OpenSuctionGripper(_SubState):
    def enter(self, observation: PalletizerObservation) -> None:
        print("<open gripper>")

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        return _SubStateResult(
            True,
            PalletizerCommand(retain_motion=True, gripper=GripperCommand.OPEN),
        )


class OrientToHome(Move):
    """Return to the elbow-up home branch through joint-space control."""

    def __init__(self) -> None:
        super().__init__(p_thresh=0.005, R_thresh=0.1)

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        T = observation.robot_home_fk_T
        return MotionCommand(
            PosePq(T[:3, 3], _matrix_to_quat(T[:3, :3])), joint_target=observation.robot_default_config
        )


class MoveToFlipStation(Move):
    _FINAL_APPROACH_LOOKAHEAD = 0.08
    _MAX_APPROACH_LENGTH = 0.4
    _APPROACH_DIR = np.array([0.5, -0.3, -0.75]) / np.linalg.norm([0.5, -0.3, -0.75])

    def __init__(self) -> None:
        super().__init__(p_thresh=0.065, R_thresh=2.0)
        self.target_pose = PosePq(
            np.array([0.7916634, 0.73902607, -0.02897218]),
            np.array([0.52239186, 0.6296602, -0.5042411, 0.27636158]),
        )

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        approach_params = _approach_params(
            self._APPROACH_DIR,
            np.linalg.norm(self.target_pose.p - observation.robot_fk_T[:3, 3]),
            self._MAX_APPROACH_LENGTH,
            self._FINAL_APPROACH_LOOKAHEAD,
            0.05,
        )
        return MotionCommand(self.target_pose, approach_params)


class ReleaseFlipStationBin(_SubState):
    _TOWARD_BASE_DIR = np.array([-1.0, -0.3, 0.0]) / np.linalg.norm([-1.0, -0.3, 0.0])
    _TOWARD_BASE_ALPHA = 0.2

    def enter(self, observation: PalletizerObservation) -> None:
        fk_T = observation.robot_fk_T
        target_p = fk_T[:3, 3] - 0.3 * fk_T[:3, 0] + self._TOWARD_BASE_ALPHA * self._TOWARD_BASE_DIR
        self._target_pose = PosePq(target_p, _matrix_to_quat(np.diag([1.0, -1.0, -1.0])))

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        motion = MotionCommand(
            self._target_pose,
            ApproachParams(self._TOWARD_BASE_ALPHA * self._TOWARD_BASE_DIR, 0.1),
        )
        done = np.linalg.norm(self._target_pose.p - observation.robot_fk_T[:3, 3]) < 0.15
        return _SubStateResult(done, PalletizerCommand(motion=motion))


class _LiftToClearance(Move):
    """Lift vertically until the measured end-effector reaches the clearance target."""

    def __init__(self, height: float) -> None:
        super().__init__(p_thresh=0.02, R_thresh=2.0)
        self._height = height
        self._target_pose: PosePq | None = None

    def enter(self, observation: PalletizerObservation) -> None:
        fk_T = observation.robot_fk_T
        target_p = fk_T[:3, 3] + np.array([0.0, 0.0, self._height])
        self._target_pose = PosePq(target_p, _matrix_to_quat(fk_T[:3, :3]))

    def get_motion(self, observation: PalletizerObservation) -> MotionCommand:
        if self._target_pose is None:
            raise RuntimeError("Lift clearance target is unavailable before entering the state.")
        return MotionCommand(self._target_pose)


class TimedLift(_SubState):
    """Lift along world Z for a fixed simulation duration."""

    def __init__(self, height: float, duration: float) -> None:
        self.height = height
        self._duration = duration
        self._started = 0.0
        self._target_pose: PosePq | None = None

    def enter(self, observation: PalletizerObservation) -> None:
        self._started = observation.time
        fk_T = observation.robot_fk_T
        self._target_pose = PosePq(fk_T[:3, 3] + np.array([0.0, 0.0, self.height]), _matrix_to_quat(fk_T[:3, :3]))

    def step(self, observation: PalletizerObservation) -> _SubStateResult:
        done = observation.time - self._started >= self._duration
        return _SubStateResult(done, PalletizerCommand(motion=MotionCommand(self._target_pose)))


# ---------------------------------------------------------------------------
# Palletizer FSM
# ---------------------------------------------------------------------------


class PalletizerFSM:
    """Backend-neutral palletizing state machine."""

    # <start-snippet-transitions-table>
    _STATES = ["idle", "picking", "flipping", "placing", "done"]

    _TRANSITIONS = [
        {"trigger": "go_pick", "source": "idle", "dest": "picking"},
        {"trigger": "pick_done", "source": "picking", "dest": "flipping", "conditions": "needs_flip"},
        {"trigger": "pick_done", "source": "picking", "dest": "placing", "unless": "needs_flip"},
        {"trigger": "flip_done", "source": "flipping", "dest": "picking"},
        {
            "trigger": "place_done",
            "source": "placing",
            "dest": "done",
            "conditions": "placement_completes_stack",
        },
        {
            "trigger": "place_done",
            "source": "placing",
            "dest": "idle",
            "unless": "placement_completes_stack",
        },
    ]
    # <end-snippet-transitions-table>

    def __init__(self) -> None:
        self._observation: PalletizerObservation | None = None
        self.machine = Machine(
            model=self,
            states=self._STATES,
            transitions=self._TRANSITIONS,
            initial="idle",
            after_state_change="_log_state",
        )

    def step(self, observation: PalletizerObservation, *, sequence_done: bool = False) -> PalletizerDecision:
        """Advance high-level task state from one observation."""
        self._observation = observation
        previous_state = self.state
        complete_bin = False
        if self.state == "idle":
            if observation.has_active_bin and not observation.active_bin_attached:
                self.go_pick()
        elif self.state != "done" and not observation.has_active_bin:
            print("<active bin lost; returning to idle>")
            self.reset()
        elif sequence_done:
            complete_bin = self._finish_sequence(observation)
        return PalletizerDecision(self.state, self.state != previous_state, complete_bin)

    def _finish_sequence(self, observation: PalletizerObservation) -> bool:
        if self.state == "picking":
            self.pick_done()
        elif self.state == "flipping":
            self.flip_done()
        elif self.state == "placing" and observation.active_bin_at_stack_target:
            self.place_done()
            return True
        elif self.state == "placing":
            print("<placement missed target; returning to idle>")
            self.reset()
        return False

    def reset(self) -> None:
        self.machine.set_state("idle")

    def needs_flip(self) -> bool:
        return self._observation.has_active_bin and self._observation.active_bin_needs_flip

    def placement_completes_stack(self) -> bool:
        return self._observation.placement_completes_stack

    def _log_state(self) -> None:
        print(f"  -> {self.state}")


class PalletizerController:
    """Run FSM-selected motion and gripper sequences."""

    def __init__(self) -> None:
        self.fsm = PalletizerFSM()
        self._sequence: list[_SubState] = []
        self._sequence_index = 0
        self._picking_after_flip = False

    @property
    def state(self) -> str:
        return self.fsm.state

    def reset(self) -> None:
        self.fsm.reset()
        self._sequence = []
        self._sequence_index = 0
        self._picking_after_flip = False

    def step(self, observation: PalletizerObservation) -> PalletizerCommand:
        decision = self.fsm.step(observation)
        if decision.state_changed:
            self._load_sequence_for_state(decision.state, observation)
        if not self._sequence:
            return PalletizerCommand()

        result = self._sequence[self._sequence_index].step(observation)
        if not result.done:
            return result.command
        return self._advance_sequence(observation, result.command)

    def _advance_sequence(self, observation: PalletizerObservation, command: PalletizerCommand) -> PalletizerCommand:
        self._sequence_index += 1
        if self._sequence_index < len(self._sequence):
            self._sequence[self._sequence_index].enter(observation)
            return command

        finished_flip = self.fsm.state == "flipping"
        decision = self.fsm.step(observation, sequence_done=True)
        self._picking_after_flip = finished_flip and decision.state == "picking"
        self._load_sequence_for_state(decision.state, observation)
        return replace(command, complete_bin=decision.complete_bin)

    # <start-snippet-on-enter-picking>
    def _load_sequence_for_state(self, state: str, observation: PalletizerObservation) -> None:
        if state == "picking":
            pickup_wait = (
                _Wait(wait_time=1.0) if self._picking_after_flip else _Wait(wait_time=1.0, pause_conveyor_on_done=True)
            )
            lift = (
                TimedLift(height=0.3, duration=0.1)
                if observation.active_bin_needs_flip
                else _LiftToClearance(height=0.5)
            )
            self._sequence = [
                pickup_wait,
                ReachToPick(),
                _Wait(wait_time=1.0),
                CloseSuctionGripper(),
                lift,
            ]
        elif state == "flipping":
            self._sequence = [
                OrientToHome(),
                MoveToFlipStation(),
                OpenSuctionGripper(),
                _Wait(wait_time=0.25),
                ReleaseFlipStationBin(),
                OrientToHome(),
            ]
        elif state == "placing":
            self._sequence = [
                ReachToPlace(),
                _Wait(wait_time=0.5),
                OpenSuctionGripper(),
                TimedLift(height=0.1, duration=0.25),
                OrientToHome(),
            ]
        else:
            self._sequence = []
        self._sequence_index = 0
        if self._sequence:
            self._sequence[0].enter(observation)

    # <end-snippet-on-enter-picking>


# ---------------------------------------------------------------------------
# Scene setup and main loop
# ---------------------------------------------------------------------------


def _add_scene_reference(
    usd_path: str,
    prim_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    variants: list[tuple[str, str]] | None = None,
) -> None:
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path, variants=variants)
    XformPrim(
        prim_path,
        translations=[translation],
        orientations=[orientation],
        scales=[scale],
        reset_xform_op_properties=True,
    )


def _author_conveyor_kinematic_override(stage: Usd.Stage, rollers_path: str) -> None:
    """Keep the referenced conveyor rigid body kinematic during physics parsing."""
    edit_target = stage.GetEditTarget()
    edit_layer = edit_target.GetLayer()
    rollers_spec_path = edit_target.MapToSpecPath(Sdf.Path(rollers_path))

    rollers_spec = Sdf.CreatePrimInLayer(edit_layer, rollers_spec_path)
    kinematic_spec = Sdf.AttributeSpec(
        rollers_spec, "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, Sdf.VariabilityUniform
    )
    kinematic_spec.default = True


def build_scene(env_path: str, ur10_assets: Ur10Assets, use_test_gripper: bool = False) -> Ur10:
    """Build the UR10 palletizing scene from reusable asset references."""
    stage_utils.define_prim(env_path)
    stage = stage_utils.get_current_stage(backend="usd")
    stage.DefinePrim(f"{env_path}/conveyor", "Xform")
    with Sdf.ChangeBlock():
        _add_scene_reference(
            ur10_assets.conveyor_usd,
            f"{env_path}/conveyor",
            translation=(-0.0544, 2.409, -1.4),
            orientation=(0.7071068, 0.0, 0.0, -0.7071068),
        )
        _author_conveyor_kinematic_override(stage, f"{env_path}/conveyor/Rollers")
    _add_scene_reference(
        ur10_assets.flip_stack_usd,
        f"{env_path}/pallet_holder",
        translation=(0.9087183896885608, 0.6628936651831716, -0.9452558688718844),
        orientation=(0.7071068, 0.0, 0.0, -0.7071068),
    )
    _add_scene_reference(
        ur10_assets.pallet_usd,
        f"{env_path}/pallet",
        translation=(0.8826279802717268, -0.32810499266628174, -0.809778981900029),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),
    )
    _add_scene_reference(
        ur10_assets.dolly_usd,
        f"{env_path}/dolly",
        translation=(0.887508980162628, -0.3328929925592616, -1.1810699736010284),
    )
    _add_scene_reference(
        ur10_assets.ur10_mount_usd,
        f"{env_path}/ur10_mount",
        translation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    _add_scene_reference(
        ur10_assets.sortbot_housing_usd,
        f"{env_path}/sortbot_housing",
        translation=(-0.01799999959766865, -0.2819999936968088, -1.1749999737367034),
    )
    _add_scene_reference(
        ur10_assets.ur10_usd,
        f"{env_path}/ur10",
        translation=(0.0, 0.0, 0.0),
        variants=ur10_assets.ur10_variants,
    )

    ground = GroundPlane(
        f"{env_path}/staticPlaneActor",
        sizes=50.0,
        translations=[(0.0, 0.0, -1.1818999735824764)],
        templates=None,
    )
    ground.set_visibilities(False)

    stage = stage_utils.get_current_stage(backend="usd")
    robot_collision_group = UsdPhysics.CollisionGroup.Define(stage, f"{env_path}/CollisionGroup")
    robot_collision_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path(f"{env_path}/ur10"))
    housing_collision_group = UsdPhysics.CollisionGroup.Define(stage, f"{env_path}/CollisionGroup_01")
    housing_collision_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(
        Sdf.Path(f"{env_path}/sortbot_housing")
    )
    robot_collision_group.CreateFilteredGroupsRel().AddTarget(housing_collision_group.GetPrim().GetPath())

    stage_utils.add_reference_to_stage(usd_path=ur10_assets.background_usd, path="/World/Background")
    XformPrim(
        ["/World/Background"],
        positions=[10.00, 2.00, -1.18180],
        orientations=[0.7071, 0, 0, 0.7071],
        reset_xform_op_properties=True,
    )
    distant_light = DistantLight(
        "/World/DistantLight",
        angles=np.deg2rad(1.0),
        orientations=[(0.9238795, 0.0, 0.3826834, 0.0)],
    )
    distant_light.set_intensities(3000)
    create_planning_obstacles()

    robot = Ur10(prim_path=f"{env_path}/ur10", use_test_gripper=use_test_gripper)
    return robot


def configure_conveyor(env_path: str) -> None:
    """Configure conveyor physics after referenced roller meshes finish loading.

    Args:
        env_path: Root prim path of the palletizing workcell.

    Raises:
        RuntimeError: If the referenced roller prim is unavailable.
    """
    stage = stage_utils.get_current_stage(backend="usd")
    rollers = stage.GetPrimAtPath(f"{env_path}/conveyor/Rollers")
    if not rollers.IsValid():
        raise RuntimeError(f"Conveyor rollers are unavailable below {env_path!r}.")

    roller_meshes = [prim for prim in Usd.PrimRange(rollers) if prim.IsA(UsdGeom.Mesh)]
    if not roller_meshes:
        raise RuntimeError(f"Conveyor roller meshes are unavailable below {rollers.GetPath()}.")

    conveyor_node = create_conveyor_belt(stage, rollers)
    conveyor_node.GetAttribute("inputs:animateDirection").Set(Gf.Vec2f(0.0, 1.0))
    conveyor_node.GetAttribute("inputs:animateScale").Set(0.01)
    conveyor_node.GetAttribute("inputs:animateTexture").Set(True)
    conveyor_node.GetAttribute("inputs:direction").Set(Gf.Vec3f(1.0, 0.0, 0.0))
    conveyor_node.GetParent().GetAttribute("graph:variable:Velocity").Set(_CONVEYOR_SPEED)
