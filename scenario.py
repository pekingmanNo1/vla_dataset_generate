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

"""Shared scene-to-controller boundary for manipulation examples."""

from __future__ import annotations

from collections.abc import Iterable

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.physics.tensors as physics_tensors
import warp as wp
from isaacsim.core.experimental.objects import Cone, Cylinder, DomeLight, GroundPlane, Mesh
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from isaacsim.core.experimental.utils import backend as backend_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.robot_motion.cumotion import CumotionWorldInterface
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf

from .robots import RobotConfig, get_robot_config


class ManipulationScenario:
    """Load a configured robot and translate between simulation and ``RobotState``.

    Call :meth:`setup_scene`, add task objects, then call :meth:`initialize_world_binding`
    before constructing a motion controller. During simulation, synchronize the world,
    read measured state, run the controller, and apply its desired state.

    Args:
        robot: Supported robot name or explicit robot configuration.
        robot_prim_path: Absolute USD path where the robot is loaded.
        robot_usd_path: Optional robot USD path that overrides the configured asset.
        offset: World-space translation applied to the robot.

    Raises:
        ValueError: If the robot name, prim path, or offset values are invalid.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import ManipulationScenario

        >>> scenario = ManipulationScenario("franka")
        >>> scenario.setup_scene()  # doctest: +SKIP
    """

    def __init__(
        self,
        robot: str | RobotConfig = "franka",
        robot_prim_path: str = "/World/robot",
        robot_usd_path: str | None = None,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if (
            not Sdf.Path.IsValidPathString(robot_prim_path)
            or not Sdf.Path(robot_prim_path).IsAbsolutePath()
            or not Sdf.Path(robot_prim_path).IsPrimPath()
        ):
            raise ValueError("robot_prim_path must be a valid absolute USD prim path.")

        self._robot_config = get_robot_config(robot) if isinstance(robot, str) else robot
        self._robot_prim_path = robot_prim_path
        self._robot_usd_path = robot_usd_path
        self._offset = tuple(float(value) for value in offset)
        self._articulation: Articulation | None = None
        self._measurement_prim: RigidPrim | None = None
        self._world_binding: mg.WorldBinding | None = None
        self._tracked_collision_paths: tuple[str, ...] = ()

    @property
    def robot_config(self) -> RobotConfig:
        """Get the selected robot configuration.

        Returns:
            Robot configuration used by the scenario.

        Example:

        .. code-block:: python

            >>> scenario.robot_config.name  # doctest: +SKIP
            'franka'
        """
        return self._robot_config

    @property
    def robot_prim_path(self) -> str:
        """Get the absolute USD path of the robot.

        Returns:
            Robot prim path.

        Example:

        .. code-block:: python

            >>> scenario.robot_prim_path  # doctest: +SKIP
            '/World/robot'
        """
        return self._robot_prim_path

    @property
    def articulation(self) -> Articulation:
        """Get the initialized robot articulation.

        Returns:
            Robot articulation created by :meth:`setup_scene`.

        Raises:
            RuntimeError: If the scene has not been set up.

        Example:

        .. code-block:: python

            >>> scenario.articulation  # doctest: +SKIP
        """
        if self._articulation is None:
            raise RuntimeError("Call setup_scene() before accessing the articulation.")
        return self._articulation

    @property
    def joint_space(self) -> list[str]:
        """Get the ordered robot joint names.

        Returns:
            Joint names reported by the initialized articulation.

        Raises:
            RuntimeError: If the scene has not been set up.

        Example:

        .. code-block:: python

            >>> scenario.joint_space  # doctest: +SKIP
        """
        return list(self.articulation.dof_names)

    @property
    def site_space(self) -> list[str]:
        """Get the spatial sites exposed to motion controllers.

        Returns:
            List containing the configured controller frame.

        Example:

        .. code-block:: python

            >>> scenario.site_space  # doctest: +SKIP
            ['panda_hand']
        """
        return [self._robot_config.tool.controller_frame]

    @property
    def world_interface(self) -> CumotionWorldInterface:
        """Get the cuMotion collision-world interface.

        Returns:
            World interface owned by the initialized binding.

        Raises:
            RuntimeError: If the world binding has not been initialized.

        Example:

        .. code-block:: python

            >>> scenario.world_interface  # doctest: +SKIP
        """
        return self._get_world_binding().get_world_interface()

    def setup_scene(self) -> Articulation:
        """Add the configured robot, ground plane, and light to the current stage.

        Returns:
            Initialized robot articulation.

        Raises:
            RuntimeError: If assets cannot be resolved or the robot lacks configured joints or tool prims.

        Example:

        .. code-block:: python

            >>> articulation = scenario.setup_scene()  # doctest: +SKIP
        """
        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath("/World/ground_plane").IsValid():
            GroundPlane("/World/ground_plane")
        if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
            DomeLight("/World/DomeLight").set_intensities(1000)

        asset_path = self._robot_usd_path
        variants: list[tuple[str, str]] = []
        if asset_path is None:
            assets_root = get_assets_root_path()
            if assets_root is None:
                raise RuntimeError("Unable to resolve the Isaac Sim assets root.")
            asset_path = assets_root + self._robot_config.asset_path
            variants = list(self._robot_config.variants)

        stage_utils.add_reference_to_stage(asset_path, self._robot_prim_path, variants=variants)
        return self._bind_robot_prims()

    def bind_existing_robot(self) -> Articulation:
        """Bind the scenario to a robot that already exists on the current stage.

        This method preserves the robot's authored world transform and joint defaults.

        Returns:
            Articulation wrapper for the existing robot.

        Raises:
            RuntimeError: If the robot prim, configured joints, or measurement prim is missing.

        Example:

        .. code-block:: python

            >>> articulation = scenario.bind_existing_robot()  # doctest: +SKIP
        """
        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath(self._robot_prim_path).IsValid():
            raise RuntimeError(f"Stage is missing robot prim {self._robot_prim_path}.")
        self._articulation = Articulation(self._robot_prim_path)
        self._bind_measurement_prim()
        return self._articulation

    def refresh_robot_prims(self) -> Articulation:
        """Recreate robot wrappers invalidated while USD references were loading.

        Returns:
            Current valid robot articulation.
        """
        if (
            self._articulation is None
            or not self._articulation.valid
            or not self._articulation.is_physics_tensor_entity_valid()
        ):
            return self._bind_robot_prims()
        return self._articulation

    def _bind_robot_prims(self) -> Articulation:
        XformPrim(
            self._robot_prim_path,
            positions=self._offset,
            orientations=(1.0, 0.0, 0.0, 0.0),
            reset_xform_op_properties=True,
        )
        self._articulation = Articulation(self._robot_prim_path)
        self._bind_measurement_prim()
        dof_indices = {name: index for index, name in enumerate(self._articulation.dof_names)}
        valid_dof_indices = [
            index
            for index, dof_type in enumerate(self._articulation.dof_types)
            if dof_type != physics_tensors.DofType.Invalid
        ]
        default_positions = np.zeros(self._articulation.num_dofs, dtype=np.float32)
        with backend_utils.use_backend("usd", raise_on_fallback=True):
            default_positions[valid_dof_indices] = self._articulation.get_dof_position_targets(
                dof_indices=valid_dof_indices
            ).numpy()[0]
        for name, value in self._robot_config.default_joint_positions:
            default_positions[dof_indices[name]] = value
        self._articulation.set_default_state(
            positions=self._offset,
            orientations=(1.0, 0.0, 0.0, 0.0),
            dof_positions=default_positions,
            dof_velocities=np.zeros_like(default_positions),
            dof_efforts=np.zeros_like(default_positions),
        )

        return self._articulation

    def _bind_measurement_prim(self) -> None:
        stage = stage_utils.get_current_stage()
        dof_names = set(self.articulation.dof_names)
        missing = [name for name, _ in self._robot_config.default_joint_positions if name not in dof_names]
        if missing:
            raise RuntimeError(f"Robot asset is missing configured joints: {', '.join(missing)}.")
        measurement_path = f"{self._robot_prim_path}/{self._robot_config.tool.measurement_prim_path}"
        if not stage.GetPrimAtPath(measurement_path).IsValid():
            raise RuntimeError(f"Robot asset is missing measurement prim {measurement_path}.")
        self._measurement_prim = RigidPrim(measurement_path)

    def initialize_world_binding(self, exclude_prim_paths: Iterable[str] = ()) -> mg.WorldBinding:
        """Initialize collision-world tracking around the robot.

        Call this method after adding task objects so their collision geometry can be
        discovered. The robot and requested paths are excluded from tracking.

        Args:
            exclude_prim_paths: Prim paths excluded from collision-world tracking.

        Returns:
            Initialized collision-world binding.

        Raises:
            RuntimeError: If the robot scene has not been set up.

        Example:

        .. code-block:: python

            >>> scenario.initialize_world_binding(["/World/target"])  # doctest: +SKIP
        """
        robot_positions, robot_orientations = self.articulation.get_world_poses()
        objects = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_positions.numpy()[0],
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[self._robot_prim_path, *exclude_prim_paths],
        )
        strategy = mg.ObstacleStrategy()
        strategy.set_default_configuration(Mesh, mg.ObstacleConfiguration("obb", 0.01))
        strategy.set_default_configuration(Cone, mg.ObstacleConfiguration("obb", 0.01))
        strategy.set_default_configuration(Cylinder, mg.ObstacleConfiguration("obb", 0.01))
        world_interface = CumotionWorldInterface(device="cpu")
        binding = mg.WorldBinding(
            world_interface=world_interface,
            obstacle_strategy=strategy,
            tracked_prims=objects,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        binding.initialize()
        world_interface.update_world_to_robot_root_transforms((robot_positions, robot_orientations))
        self._world_binding = binding
        self._tracked_collision_paths = tuple(objects)
        return binding

    def set_planning_obstacles_enabled(self, prim_paths: Iterable[str], enabled: bool) -> None:
        """Enable or disable tracked collisions below the requested prim paths.

        The update is applied only after every requested root matches at least one tracked
        collision path.

        Args:
            prim_paths: Root prim paths whose tracked collisions are updated.
            enabled: Whether the matching collisions participate in motion planning.

        Raises:
            RuntimeError: If the world binding has not been initialized.
            ValueError: If no roots are provided or a root has no tracked collisions.

        Example:

        .. code-block:: python

            >>> scenario.set_planning_obstacles_enabled(["/World/cube"], False)  # doctest: +SKIP
        """
        roots = tuple(str(path).rstrip("/") for path in prim_paths)
        if not roots:
            raise ValueError("At least one planning obstacle path is required.")
        matches = {
            root: tuple(
                tracked
                for tracked in self._tracked_collision_paths
                if tracked == root or tracked.startswith(root + "/")
            )
            for root in roots
        }
        missing = [root for root, tracked in matches.items() if not tracked]
        if missing:
            raise ValueError(f"No tracked planning obstacles matched: {', '.join(missing)}.")
        paths = list(dict.fromkeys(path for root in roots for path in matches[root]))
        self.world_interface.update_obstacle_enables(
            prim_paths=paths,
            enabled_array=wp.array([enabled] * len(paths), dtype=wp.bool, device="cpu"),
        )

    def sync_world(self) -> None:
        """Synchronize robot-root and obstacle transforms with the planning world.

        Raises:
            RuntimeError: If the scene or world binding has not been initialized.

        Example:

        .. code-block:: python

            >>> scenario.sync_world()  # doctest: +SKIP
        """
        binding = self._get_world_binding()
        binding.get_world_interface().update_world_to_robot_root_transforms(self.articulation.get_world_poses())
        binding.synchronize_transforms()

    def read_tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the configured controller-frame pose in world coordinates.

        Returns:
            Batched position and WXYZ orientation arrays with shapes ``(1, 3)`` and ``(1, 4)``.

        Raises:
            RuntimeError: If the scene is uninitialized or the measured pose is not finite.
        """
        if self._measurement_prim is None:
            raise RuntimeError("Call setup_scene() before reading the tool pose.")
        positions, orientations = self._measurement_prim.get_world_poses()
        measurement_position = positions.numpy()[0]
        measurement_orientation = orientations.numpy()[0]
        tool = self._robot_config.tool
        controller_position = transform_utils.transform_local_to_world(
            tool.measurement_to_controller_position,
            measurement_position,
            measurement_orientation,
            dtype=wp.float32,
            device="cpu",
        ).numpy()
        controller_orientation = transform_utils.quaternion_multiplication(
            measurement_orientation,
            tool.measurement_to_controller_orientation,
            dtype=wp.float32,
            device="cpu",
        ).numpy()
        if not np.isfinite((*controller_position, *controller_orientation)).all():
            raise RuntimeError("Measured controller pose is not finite.")
        return controller_position.reshape(1, 3), controller_orientation.reshape(1, 4)

    def read_robot_state(self) -> mg.RobotState:
        """Read measured joints and the controller-frame pose.

        Returns:
            Robot state containing named joint and spatial measurements.

        Raises:
            RuntimeError: If the scene is uninitialized or the measured pose is not finite.

        Example:

        .. code-block:: python

            >>> estimated_state = scenario.read_robot_state()  # doctest: +SKIP
        """
        controller_positions, controller_orientations = self.read_tool_pose()
        joints = self.joint_space
        tool_frame = self._robot_config.tool.controller_frame
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=joints,
                positions=(joints, self.articulation.get_dof_positions()),
                velocities=(joints, self.articulation.get_dof_velocities()),
            ),
            sites=mg.SpatialState.from_name(
                spatial_space=[tool_frame],
                positions=(
                    [tool_frame],
                    wp.array(controller_positions, dtype=wp.float32, device="cpu"),
                ),
                orientations=(
                    [tool_frame],
                    wp.array(controller_orientations, dtype=wp.float32, device="cpu"),
                ),
            ),
        )

    def apply_robot_state(self, desired_state: mg.RobotState | None) -> None:
        """Apply the available joint commands from a desired robot state.

        A missing desired state or joint state produces no command.

        Args:
            desired_state: Controller output containing optional position, velocity, and effort commands.

        Raises:
            RuntimeError: If a command is present before the scene is initialized.

        Example:

        .. code-block:: python

            >>> scenario.apply_robot_state(desired_state)  # doctest: +SKIP
        """
        if desired_state is None or desired_state.joints is None:
            return
        joints = desired_state.joints
        if joints.positions is not None:
            self.articulation.set_dof_position_targets(joints.positions, dof_indices=joints.position_indices)
        if joints.velocities is not None:
            self.articulation.set_dof_velocity_targets(joints.velocities, dof_indices=joints.velocity_indices)
        if joints.efforts is not None:
            self.articulation.set_dof_efforts(joints.efforts, dof_indices=joints.effort_indices)

    def cleanup(self) -> None:
        """Release articulation, measurement, and world-binding references.

        Example:

        .. code-block:: python

            >>> scenario.cleanup()
        """
        self._world_binding = None
        self._tracked_collision_paths = ()
        self._measurement_prim = None
        self._articulation = None

    def _get_world_binding(self) -> mg.WorldBinding:
        """Get the initialized collision-world binding."""
        if self._world_binding is None:
            raise RuntimeError("Call initialize_world_binding() before accessing the world binding.")
        return self._world_binding


__all__ = ["ManipulationScenario"]
