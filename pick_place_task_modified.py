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

"""Callback-free pick/place task shared by interactive examples."""

from __future__ import annotations

from collections.abc import Iterable

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot

from .controllers import (
    GripperCommand,
    JointGripperController,
    PickPlaceController,
    PickPlacePhase,
    SurfaceGripperController,
)
from .robots import JointGripperConfig, SurfaceGripperConfig
from .scenario import ManipulationScenario


class PickPlaceTask:
    """One reusable, callback-free pick/place task."""

    def __init__(
        self,
        robot_path: str = "/World/robot",
        cube_path: str = "/World/Cube",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cube_positions: list[tuple[float, float, float]] | None = None,
        place_position: list[tuple[float, float, float]] | None = None,
        robot_name: str = "franka",
    ) -> None:
        self.scenario = ManipulationScenario(robot_name, robot_prim_path=robot_path, offset=offset)
        self.cube_path = cube_path
        self.offset = np.asarray(offset, dtype=np.float32)
        if cube_positions is None:
            cube_positions = [(0.5, 0.0, 0.0258),(0.4, 0.4, 0.0258)]
        if place_position is None:
            place_positions = [(0.0, 0.5, 0.0258),(0.2, 0.5, 0.0258)]
        self.pick_positions = [self.offset + np.asarray(position, dtype=np.float32) for position in cube_positions]
        self.place_positions = [self.offset + np.asarray(place_position, dtype=np.float32) for place_position in place_positions]
        self.cubes: list[RigidPrim] = []
        self.cube_paths: list[str] = []
        self.controller: PickPlaceController | None = None
        self._time = 0.0
        self._needs_reset = True
        self._active_cube = 0 # 现在正在抓第几个cube
        self._goal_setpoint: mg.RobotState | None = None # 这次任务的目标是什么
        self._failure_reason: str | None = None # 如果失败，为什么失败
        self._done = False # 整个任务完成了吗
        self._grasp_checked = False # 有没有确认真的抓住
        self._lift_checked = False # 有没有确认真的把cuba抬起来
        self._settle_time = 0.0
        self._completion_time = 0.0
        self._planning_disabled_cube: int | None = None
        self._world_initialized = False
        self._surface_gripper_interface = None
        self._surface_gripper_path: str | None = None
        self.cube_colors: list[str] = ["blue", "red"]

    def setup_scene(self) -> None:
        self.scenario.setup_scene()
        for index, position in enumerate(self.pick_positions):
            path = self.cube_path if len(self.pick_positions) == 1 else f"{self.cube_path}_{index}"
            cube_shape = Cube(
                path,
                positions=position,
                sizes=1.0,
                scales=[0.0515, 0.0515, 0.0515],
                colors=self.cube_colors[index],
            )
            GeomPrim(cube_shape.paths, apply_collision_apis=True) # 增加collision
            cube = RigidPrim(cube_shape.paths) # 把它作为一个刚体
            cube.set_default_state(
                positions=np.asarray([position], dtype=np.float32),
                orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                linear_velocities=np.zeros((1, 3), dtype=np.float32),
                angular_velocities=np.zeros((1, 3), dtype=np.float32),
            )
            self.cubes.append(cube)
            self.cube_paths.append(path)

    def initialize(self, exclude_prim_paths: Iterable[str] = ()) -> None: # 建立真正的controller
        self.scenario.initialize_world_binding(exclude_prim_paths)
        self._world_initialized = True
        config = self.scenario.robot_config
        cumotion_robot = load_cumotion_supported_robot(config.name)
        site_space = cumotion_robot.robot_description.tool_frame_names()
        tool_frame = config.tool.controller_frame
        if tool_frame not in site_space:
            raise RuntimeError(
                f"cuMotion configuration for {config.name!r} does not support configured tool frame {tool_frame!r}."
            )
        arm = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=self.scenario.world_interface,
            robot_joint_space=self.scenario.joint_space,
            robot_site_space=site_space,
            tool_frame=tool_frame,
        )
        gripper = config.gripper
        if isinstance(gripper, JointGripperConfig): # 代码支持两种gripper
            common = {
                "robot_joint_space": self.scenario.joint_space,
                "joint_names": gripper.joint_names,
                "open_positions": gripper.open_positions,
                "closed_positions": gripper.closed_positions,
                "position_tolerance": gripper.position_tolerance,
                "velocity_tolerance": gripper.velocity_tolerance,
                "minimum_close_fraction": gripper.minimum_close_fraction,
            }
            open_gripper = JointGripperController(**common, command=GripperCommand.OPEN)
            close_gripper = JointGripperController(**common, command=GripperCommand.CLOSE)
        elif isinstance(gripper, SurfaceGripperConfig): # 代码支持两种gripper
            from isaacsim.robot.surface_gripper import _surface_gripper

            gripper_path = f"{self.scenario.robot_prim_path}/{gripper.relative_path}"
            self._surface_gripper_interface = _surface_gripper.acquire_surface_gripper_interface()
            self._surface_gripper_path = gripper_path
            open_gripper = SurfaceGripperController(
                gripper_path=gripper_path,
                command=GripperCommand.OPEN,
                interface=self._surface_gripper_interface,
            )
            close_gripper = SurfaceGripperController(
                gripper_path=gripper_path,
                command=GripperCommand.CLOSE,
                interface=self._surface_gripper_interface,
            )
        else:
            raise TypeError(f"Unsupported gripper configuration: {type(gripper).__name__}")
        self.controller = PickPlaceController( # 把它们组装成pickplacecontroller，也就是支持RmpFlowController和GripperController
            arm_controller=arm,
            gripper_open_controller=open_gripper,
            gripper_close_controller=close_gripper,
            robot_site_space=site_space,
            tool_frame=tool_frame,
            controller_to_grasp_position=config.tool.controller_to_grasp_position,
            controller_to_grasp_orientation=config.tool.controller_to_grasp_orientation,
            grasp_orientation=config.grasp_orientation,
            phase_timeouts={PickPlacePhase.GRASP: 3.0} if isinstance(gripper, SurfaceGripperConfig) else None,
            approach_height=0.30,
            grasp_position_tolerance=0.025 if isinstance(gripper, SurfaceGripperConfig) else None,
        )
        self.reset()

    def reset(self) -> None:
        self._release_attachment()
        self._restore_planning_collision()
        self._time = 0.0
        self._needs_reset = True
        self._active_cube = 0
        self._goal_setpoint = None
        self._failure_reason = None
        self._done = False
        self._grasp_checked = False
        self._lift_checked = False
        self._settle_time = 0.0
        self._completion_time = 0.0

    def reset_robot(self) -> None:
        self._release_attachment()
        self.scenario.articulation.reset_to_default_state()
        for cube in self.cubes:
            cube.reset_to_default_state()
        self.reset()

    def _release_attachment(self) -> None:
        if self._surface_gripper_interface is not None and self._surface_gripper_path is not None:
            self._surface_gripper_interface.open_gripper(self._surface_gripper_path)

    def _set_active_planning_enabled(self, enabled: bool) -> None:
        if not self._world_initialized:
            return
        if enabled:
            if self._planning_disabled_cube is None:
                return
            path = self.cube_paths[self._planning_disabled_cube]
            self.scenario.set_planning_obstacles_enabled([path], True)
            self._planning_disabled_cube = None
        elif self._planning_disabled_cube is None:
            self.scenario.set_planning_obstacles_enabled([self.cube_paths[self._active_cube]], False)
            self._planning_disabled_cube = self._active_cube

    def _restore_planning_collision(self) -> None:
        self._set_active_planning_enabled(True)

    def _sync_active_planning_collision(self) -> None:
        if self.controller is None:
            return
        disabled_phases = {
            PickPlacePhase.DESCEND_PICK,
            PickPlacePhase.GRASP,
            PickPlacePhase.LIFT,
            PickPlacePhase.APPROACH_PLACE,
            PickPlacePhase.DESCEND_PLACE,
            PickPlacePhase.RELEASE,
        }
        if isinstance(self.scenario.robot_config.gripper, SurfaceGripperConfig):
            disabled_phases.add(PickPlacePhase.APPROACH_PICK)
        self._set_active_planning_enabled(self.controller.phase not in disabled_phases)

    def _capture_setpoint(self) -> mg.RobotState: # 决定抓哪里，放哪里
        pick = self.pick_positions[self._active_cube]
        if self.cubes:
            pick = self.cubes[self._active_cube].get_world_poses()[0].numpy()[0]

        # 当前 cube 对应自己的放置位置
        place = self.place_positions[self._active_cube].copy()

        if isinstance(self.scenario.robot_config.gripper, SurfaceGripperConfig):
            pick = pick.copy()
            pick[2] += 0.02575
            place[2] += 0.02575
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["pick", "place"],
                positions=(
                    ["pick", "place"],
                    wp.array([pick, place], dtype=wp.float32, device="cpu"),
                ),
            )
        )

    def _cube_position(self) -> np.ndarray:
        return np.asarray(self.cubes[self._active_cube].get_world_poses()[0].numpy()[0], dtype=np.float64)

    def _grasp_position(self, estimated: mg.RobotState) -> np.ndarray:
        tool_frame = self.scenario.robot_config.tool.controller_frame
        if estimated.sites is None or estimated.sites.positions is None or estimated.sites.orientations is None:
            raise RuntimeError(f"RobotState is missing measured tool site {tool_frame!r}.")
        sites = estimated.sites
        if tool_frame not in sites.position_names or tool_frame not in sites.orientation_names:
            raise RuntimeError(f"RobotState is missing measured tool site {tool_frame!r}.")
        controller_position = sites.positions.numpy()[sites.position_names.index(tool_frame)]
        controller_orientation = sites.orientations.numpy()[sites.orientation_names.index(tool_frame)]
        return transform_utils.transform_local_to_world(
            self.scenario.robot_config.tool.controller_to_grasp_position,
            controller_position,
            controller_orientation,
            dtype=wp.float64,
            device="cpu",
        ).numpy()

    def _validate_lift(self, estimated: mg.RobotState) -> bool:
        if self._goal_setpoint is None or self._goal_setpoint.sites is None:
            self._failure_reason = "Pick/place goal is unavailable."
            return False
        goal_sites = self._goal_setpoint.sites
        pick = np.asarray(goal_sites.positions.numpy()[goal_sites.position_names.index("pick")], dtype=np.float64)
        cube = self._cube_position()
        if cube[2] - pick[2] < 0.04:
            self._failure_reason = f"Cube {self._active_cube} was not lifted."
            self._restore_planning_collision()
            return False
        if np.linalg.norm(cube - self._grasp_position(estimated)) > 0.12:
            self._failure_reason = f"Cube {self._active_cube} is not held by the gripper."
            self._restore_planning_collision()
            return False
        self._lift_checked = True
        return True

    def _validate_grasp(self, estimated: mg.RobotState) -> bool:
        cube_path = self.cube_paths[self._active_cube]
        if self._surface_gripper_interface is not None and self._surface_gripper_path is not None:
            gripped = self._surface_gripper_interface.get_gripped_objects(self._surface_gripper_path)
            if not any(str(path) == cube_path or str(path).startswith(cube_path + "/") for path in gripped):
                self._failure_reason = f"Surface gripper did not attach cube {self._active_cube}."
                self._restore_planning_collision()
                return False
        elif np.linalg.norm(self._cube_position() - self._grasp_position(estimated)) > 0.12:
            self._failure_reason = f"Cube {self._active_cube} is not within the gripper."
            self._restore_planning_collision()
            return False
        self._grasp_checked = True
        return True

    def _finish_active_goal(self, dt: float) -> bool:
        if self._goal_setpoint is None or self._goal_setpoint.sites is None:
            self._failure_reason = "Pick/place goal is unavailable."
            return False
        place = self.place_positions[self._active_cube].astype(np.float64).copy()
        position_error = np.linalg.norm(self._cube_position() - place)
        linear_velocity = self.cubes[self._active_cube].get_velocities()[0].numpy()[0]
        settled = position_error <= 0.03 and np.linalg.norm(linear_velocity) <= 0.10
        self._completion_time += dt
        self._settle_time = self._settle_time + dt if settled else 0.0
        if self._settle_time < 0.15:
            if self._completion_time >= 2.0:
                self._failure_reason = (
                    f"Cube {self._active_cube} did not settle at its place goal "
                    f"(position error {position_error:.3f} m)."
                )
                self._restore_planning_collision()
                return False
            return True
        if self._active_cube + 1 == len(self.cubes):
            self._done = True
            return True
        self._active_cube += 1
        self._time = 0.0
        self._needs_reset = True
        self._goal_setpoint = None
        self._grasp_checked = False
        self._lift_checked = False
        self._settle_time = 0.0
        self._completion_time = 0.0
        return True

    def step(self, dt: float) -> bool:
        if self.controller is None or self._done or self._failure_reason is not None:
            return False
        self.scenario.sync_world()
        estimated = self.scenario.read_robot_state() # robot state / proprioception
        if self._needs_reset: # 第一次运行：创建goal
            self._goal_setpoint = self._capture_setpoint()
            if not self.controller.reset(estimated, self._goal_setpoint, self._time):
                return False
            self._needs_reset = False
            self._sync_active_planning_collision()
        elif self.controller.is_done:
            return self._finish_active_goal(dt)
        else:
            self._time += dt # 真正产生机器人动作的地方
            desired = self.controller.forward(estimated, self._goal_setpoint, self._time) # estimated=机器人现在在哪里，goal setpoint=cube在哪里+要放哪里，time=当前执行时间
            self.scenario.apply_robot_state(desired)
            self._sync_active_planning_collision()
            if not self._grasp_checked and self.controller.phase is PickPlacePhase.LIFT:
                if not self._validate_grasp(estimated):
                    return False
            if (
                not self._lift_checked
                and self.controller.phase
                in {
                    PickPlacePhase.APPROACH_PLACE,
                    PickPlacePhase.DESCEND_PLACE,
                    PickPlacePhase.RELEASE,
                    PickPlacePhase.RETREAT,
                    PickPlacePhase.DONE,
                }
                and not self._validate_lift(estimated)
            ):
                return False
        return not self.failed

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def failed(self) -> bool:
        if self._failure_reason is not None:
            return True
        if self._needs_reset:
            return False
        return self.controller is not None and self.controller.failed

    def status(self) -> dict[str, object]:
        if self.controller is None:
            return {"error": "Controller not initialized"}
        return {
            "phase": self.controller.phase.name,
            "active_cube": self._active_cube,
            "done": self.is_done,
            "failed": self.failed,
            "failure_reason": self._failure_reason or self.controller.failure_reason,
        }

    def cleanup(self) -> None:
        self._release_attachment()
        self._restore_planning_collision()
        self.controller = None
        self.cubes.clear()
        self.cube_paths.clear()
        self._goal_setpoint = None
        self.scenario.cleanup()
        self._world_initialized = False
        self._surface_gripper_interface = None
        self._surface_gripper_path = None
