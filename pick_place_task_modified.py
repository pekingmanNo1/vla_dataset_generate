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

import os
import json
from pathlib import Path

class PickPlaceTask:
    """One reusable, callback-free pick/place task."""

    def __init__(
        self,
        robot_path: str = "/World/robot",
        cube_path: str = "/World/Cube",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cube_positions: list[tuple[float, float, float]] | None = None,
        place_positions: list[tuple[float, float, float]] | None = None,
        robot_name: str = "franka",
    ) -> None:
        self.scenario = ManipulationScenario(robot_name, robot_prim_path=robot_path, offset=offset)
        self.cube_path = cube_path
        self.offset = np.asarray(offset, dtype=np.float32)
        if cube_positions is None:
            cube_positions = [(0.5, 0.0, 0.0258),(0.4, 0.4, 0.0258)]
        if place_positions is None:
            place_positions = [
                (0.0, 0.5, 0.0258),
                (0.2, 0.5, 0.0258),
            ]

        self.place_positions = [
            self.offset + np.asarray(position, dtype=np.float32)
            for position in place_positions
        ]

            
        self.pick_positions = [self.offset + np.asarray(position, dtype=np.float32) for position in cube_positions]
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

        # 准备eposide数据收集
        self._episode_index = 0
        self._episode_frames = []
        self._max_episodes = 100
        self._dataset_root = Path("vla_dataset")
        self._dataset_root.mkdir(parents=True, exist_ok=True)

        # 判断是否卡住
        self._episode_time = 0.0
        self._episode_timeout = 40.0

    def _record_step(self, estimated: mg.RobotState, desired: mg.RobotState | None) -> None:
        frame = {
            "time": float(self._time),
            "active_cube": int(self._active_cube),
            "phase": self.controller.phase.name,
            "cube_position": self._cube_position().copy(),
            "place_position": self.place_positions[self._active_cube].copy(),
        }

        # Robot joint state
        if estimated.joints is not None:
            if estimated.joints.positions is not None:
                frame["joint_positions"] = (
                    estimated.joints.positions.numpy().copy()
                )

            if estimated.joints.velocities is not None:
                frame["joint_velocities"] = (
                    estimated.joints.velocities.numpy().copy()
                )

        # Robot action state
        if desired is not None and desired.joints is not None:
            if desired.joints.positions is not None:
                frame["action_joint_position"] = (
                    desired.joints.positions.numpy().copy()
                )

            if desired.joints.velocities is not None:
                frame["action_joint_velocity"] = (
                    desired.joints.velocities.numpy().copy()
                )

        self._episode_frames.append(frame)

    def _save_episode(self) -> None:
        if not self._episode_frames:
            return

        episode_dir = self._dataset_root / f"episode_{self._episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        times = np.asarray(
            [frame["time"] for frame in self._episode_frames],
            dtype=np.float32,
        )

        active_cubes = np.asarray(
            [frame["active_cube"] for frame in self._episode_frames],
            dtype=np.int32,
        )

        cube_positions = np.stack(
            [frame["cube_position"] for frame in self._episode_frames]
        )

        place_positions = np.stack(
            [frame["place_position"] for frame in self._episode_frames]
        )

        joint_positions = np.stack(
            [frame["joint_positions"] for frame in self._episode_frames]
        )

        joint_velocities = np.stack(
            [frame["joint_velocities"] for frame in self._episode_frames]
        )

        phases = np.asarray(
            [frame["phase"] for frame in self._episode_frames]
        )

        action_joint_positions = np.stack(
            [frame["action_joint_position"] for frame in self._episode_frames]
        )

        action_joint_velocities = np.stack(
            [frame["action_joint_velocity"] for frame in self._episode_frames]
        )

        np.savez_compressed(
            episode_dir / "trajectory.npz",
            timestamp=times,
            active_cube=active_cubes,
            cube_position=cube_positions,
            place_position=place_positions,
            joint_position=joint_positions,
            joint_velocity=joint_velocities,
            action_joint_position=action_joint_positions,
            action_joint_velocity=action_joint_velocities,
            phase=phases,
        )

        metadata = {
            "episode_index": self._episode_index,
            "success": self._done,
            "failure_reason": self._failure_reason,
            "num_frames": len(self._episode_frames),
            "instruction": "pick and place the cubes",
        }

        with open(
            episode_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[Dataset] Saved episode {self._episode_index}: "
            f"{len(self._episode_frames)} frames"
        )

        self._episode_index += 1
        self._episode_frames.clear()

    def _start_next_episode(self) -> None:
        # 重新开始任务状态
        self._done = False
        self._failure_reason = None

        self._episode_time = 0.0

        self._active_cube = 0
        self._time = 0.0

        self._needs_reset = True
        self._goal_setpoint = None

        self._grasp_checked = False
        self._lift_checked = False

        self._settle_time = 0.0
        self._completion_time = 0.0

        # 机器人回默认状态
        self.scenario.articulation.reset_to_default_state()

        # 所有 cube 回默认状态
        for cube in self.cubes:
            cube.reset_to_default_state()

        # 每个 episode 随机化
        self._randomize_episode()

        print(f"[Dataset] Starting episode {self._episode_index}")

    def _randomize_episode(self) -> None:
        # 随机 cube 位置
        for i, cube in enumerate(self.cubes):
            x = np.random.uniform(0.40, 0.55)
            y = np.random.uniform(-0.15, 0.15)
            z = 0.0258

            position = np.asarray([x, y, z], dtype=np.float32)

            self.pick_positions[i] = position

            cube.set_world_poses(
                positions=np.asarray([position], dtype=np.float32)
            )

        # 随机 place 位置
        for i in range(len(self.place_positions)):
            x = np.random.uniform(-0.15, 0.15)
            y = np.random.uniform(0.35, 0.55)
            z = 0.0258

            self.place_positions[i] = np.asarray(
                [x, y, z],
                dtype=np.float32
            )

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
        self._episode_time = 0.0
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

    def _handle_episode_failure(self, reason: str) -> bool:
        self._failure_reason = reason

        print(
            f"[Dataset] Episode {self._episode_index} failed: "
            f"{self._failure_reason}"
        )

        # 保存失败 episode
        self._save_episode()

        # 达到目标数量就结束
        if self._episode_index >= self._max_episodes:
            print("[Dataset] Collection complete.")
            return False

        # 否则进入下一 episode
        self._start_next_episode()

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

            # 保存当前 episode
            self._save_episode()

            # 已经达到最大 episode 数
            if self._episode_index >= self._max_episodes:
                print("[Dataset] Collection complete.")
                return True

            # 开始下一 episode
            self._start_next_episode()

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
        if self.controller is None:
            return False

        # episode total time
        self._episode_time += dt

        # timeout check
        if self._episode_time >= self._episode_timeout:
            return self._handle_episode_failure(
                f"Episode timeout at phase {self.controller.phase.name}"
            )
    
        self.scenario.sync_world()
        estimated = self.scenario.read_robot_state() # robot state / proprioception

        if self._needs_reset: # 第一次运行：创建goal
            self._goal_setpoint = self._capture_setpoint()
            if not self.controller.reset(
                estimated,
                self._goal_setpoint,
                self._time
            ):
                return self._handle_episode_failure(
                    self.controller.failure_reason
                    or "Controller reset failed"
                )
            self._needs_reset = False
            self._sync_active_planning_collision()
        elif self.controller.is_done:
            result = self._finish_active_goal(dt)

            if self._failure_reason is not None:
                return self._handle_episode_failure(
                    self._failure_reason
                )

            return result
        else:
            self._time += dt # 真正产生机器人动作的地方

            desired = self.controller.forward(estimated, self._goal_setpoint, self._time) # estimated=机器人现在在哪里，goal setpoint=cube在哪里+要放哪里，time=当前执行时间

            self._record_step(
                estimated=estimated,
                desired=desired,
            )
            
            self.scenario.apply_robot_state(desired)
            self._sync_active_planning_collision()
            if (
                not self._grasp_checked
                and self.controller.phase is PickPlacePhase.LIFT
            ):
                if not self._validate_grasp(estimated):
                    return self._handle_episode_failure(
                        self._failure_reason or "Grasp validation failed"
                    )
                
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
            ):
                if not self._validate_lift(estimated):
                    return self._handle_episode_failure(
                        self._failure_reason or "Lift validation failed"
                    )
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
