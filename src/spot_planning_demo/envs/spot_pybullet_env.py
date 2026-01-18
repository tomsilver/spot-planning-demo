"""PyBullet simulator and environment for Spot."""

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeAlias

import gymnasium
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_collisions_with_held_object,
    inverse_kinematics,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from spot_planning_demo.structs import (
    BANISH_POSE,
    CARDBOARD_TABLE_OBJECT,
    HUMAN_OBJECT,
    ROBOT_OBJECT,
    TIGER_TOY_OBJECT,
    TYPE_FEATURES,
    HandOver,
    MoveBase,
    Pick,
    Place,
    SpotAction,
)

RenderFrame: TypeAlias = Any


class ActionFailure(BaseException):
    """Raised in step() if raise_error_on_action_failures=True."""


@dataclass(frozen=True)
class SpotPybulletSimSpec:
    """Scene description for SpotPyBulletSim()."""

    # Robot, with pose taken from real.
    robot_base_pose: Pose = Pose.from_rpy((2.287, -0.339, -0.33), (0, 0, 1.421))

    # Floor.
    floor_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    floor_half_extents: tuple[float, float, float] = (3, 3, 0.001)
    floor_pose: Pose = Pose(
        (
            robot_base_pose.position[0],
            robot_base_pose.position[1],
            robot_base_pose.position[2] - floor_half_extents[2],
        )
    )

    # Table. Thin table positioned where the robot arm can reach.
    table_half_extents: tuple[float, float, float] = (0.15, 0.15, 0.005)
    table_pose: Pose = Pose((2.55, 0.1, 0.38))
    table_color: tuple[float, float, float, float] = (0.6, 0.3, 0.1, 1.0)

    # Shelf ceiling, forcing a side grasp and possibly forcing removal of obstacles.
    # This is currently disabled (with a very high pose.)
    shelf_ceiling_half_extents: tuple[float, float, float] = (
        table_half_extents[0],
        table_half_extents[1],
        0.001,
    )
    shelf_ceiling_pose: Pose = Pose(
        (
            table_pose.position[0],
            table_pose.position[1],
            # Comment back to make this active again.
            # table_pose.position[2] + table_half_extents[2] + 0.25,
            1000,
        ),
        table_pose.orientation,
    )
    shelf_ceiling_color: tuple[float, float, float, float] = (0.6, 0.3, 0.1, 0.5)

    # Purple block. On the table surface. Small size for easier grasping.
    purple_block_half_extents: tuple[float, float, float] = (0.02, 0.02, 0.03)
    _table_surface_z: float = table_pose.position[2] + table_half_extents[2]
    purple_block_init_pose: Pose = Pose(
        (
            table_pose.position[0] - table_half_extents[0] + 0.03,
            table_pose.position[1] - table_half_extents[1] + 0.03,
            _table_surface_z + purple_block_half_extents[2],
        )
    )
    purple_block_color: tuple[float, float, float, float] = (
        170 / 255,
        121 / 255,
        222 / 255,
        1.0,
    )

    # Green block. Currently disabled (banish pose).
    green_block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.05)
    green_block_init_pose: Pose = BANISH_POSE
    # Uncomment to enable:
    # green_block_init_pose: Pose = Pose(
    #     (
    #         table_pose.position[0] - table_half_extents[0] / 2,
    #         table_pose.position[1] + table_half_extents[1] / 2,
    #         _table_surface_z + green_block_half_extents[2],
    #     )
    # )
    green_block_color: tuple[float, float, float, float] = (
        10 / 255,
        222 / 255,
        10 / 255,
        1.0,
    )

    # Drop zone.
    drop_zone_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)
    # On the left of the initial pose.
    drop_zone_pose: Pose = Pose(
        (robot_base_pose.position[0] - 1.0, robot_base_pose.position[1], 0.75)
    )
    drop_zone_color: tuple[float, float, float, float] = (
        255 / 255,
        121 / 255,
        255 / 255,
        0.5,
    )

    # Hyperparameter for checking placement on surface success.
    placement_surface_threshold: float = 1e-1

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        return {
            "camera_target": (
                self.robot_base_pose.position[0],
                self.robot_base_pose.position[1],
                self.robot_base_pose.position[2] + 0.5,
            ),
            "camera_yaw": 180,
            "camera_distance": 1.5,
            "camera_pitch": -20,
            "background_rgb": (255, 255, 255),
            # Use for fast testing.
            # "image_width": 32,
            # "image_height": 32,
        }


class SpotPyBulletSim(gymnasium.Env[ObjectCentricState, SpotAction]):
    """PyBullet simulator for Spot demo environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        scene_description: SpotPybulletSimSpec = SpotPybulletSimSpec(),
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        raise_error_on_action_failures: bool = False,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scene_description = scene_description
        self.raise_error_on_action_failures = raise_error_on_action_failures

        # Create the PyBullet client.
        if use_gui:
            camera_info = self.scene_description.get_camera_kwargs()
            self.physics_client_id = create_gui_connection(**camera_info)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create the robot.
        robot = create_pybullet_robot(
            "spot",
            self.physics_client_id,
            fixed_base=False,
            base_pose=self.scene_description.robot_base_pose,
            control_mode="reset",
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create floor.
        self.floor_id = create_pybullet_block(
            self.scene_description.floor_color,
            self.scene_description.floor_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.floor_id, self.scene_description.floor_pose, self.physics_client_id
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_description.table_color,
            self.scene_description.table_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.table_id, self.scene_description.table_pose, self.physics_client_id
        )

        # Create a shelf ceiling.
        self.shelf_ceiling_id = create_pybullet_block(
            self.scene_description.shelf_ceiling_color,
            self.scene_description.shelf_ceiling_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.shelf_ceiling_id,
            self.scene_description.shelf_ceiling_pose,
            self.physics_client_id,
        )

        # Create purple block.
        self.purple_block_id = create_pybullet_block(
            self.scene_description.purple_block_color,
            self.scene_description.purple_block_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.purple_block_id,
            self.scene_description.purple_block_init_pose,
            self.physics_client_id,
        )

        # Create green block.
        self.green_block_id = create_pybullet_block(
            self.scene_description.green_block_color,
            self.scene_description.green_block_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.green_block_id,
            self.scene_description.green_block_init_pose,
            self.physics_client_id,
        )

        # Create drop zone.
        self.drop_zone_id = create_pybullet_block(
            self.scene_description.drop_zone_color,
            self.scene_description.drop_zone_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.drop_zone_id,
            self.scene_description.drop_zone_pose,
            self.physics_client_id,
        )

        # Create held object and transform.
        self._current_held_object_id: int | None = None
        self._current_held_object_transform: Pose | None = None

        # Designate obstacles for collision checking.
        # Note: table is excluded to allow arm to reach objects on it.
        self.obstacle_ids = {
            self.purple_block_id,
            self.green_block_id,
            self.shelf_ceiling_id,
        }

        # Map objects to pybullet IDs.
        self._object_to_pybullet_id = {
            TIGER_TOY_OBJECT: self.purple_block_id,
            CARDBOARD_TABLE_OBJECT: self.table_id,
            HUMAN_OBJECT: self.drop_zone_id,
        }

        # Indicate which features should update for which objects when the state is
        # manually set. This is importnat for the sim viz wrapper.
        self._object_to_state_set_features = {
            TIGER_TOY_OBJECT: ["x", "y", "z"],
            CARDBOARD_TABLE_OBJECT: ["x", "y"],
            HUMAN_OBJECT: ["x", "y"],
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:

        # Reset the robot.
        self.robot.set_base(self.scene_description.robot_base_pose)
        self.robot.close_fingers()

        # Reset the purple block.
        set_pose(
            self.purple_block_id,
            self.scene_description.purple_block_init_pose,
            self.physics_client_id,
        )

        # Reset the green block.
        set_pose(
            self.green_block_id,
            self.scene_description.green_block_init_pose,
            self.physics_client_id,
        )

        # Reset the held object and transform.
        self._current_held_object_id = None
        self._current_held_object_transform = None

        return self._get_obs(), {}

    def step(
        self, action: SpotAction
    ) -> tuple[ObjectCentricState, SupportsFloat, bool, bool, dict[str, Any]]:

        if isinstance(action, MoveBase):
            self._step_move_base(action.pose)

        elif isinstance(action, Pick):
            self._step_pick(action.object_name, action.end_effector_to_grasp_pose)

        elif isinstance(action, Place):
            self._step_place(action.surface_name, action.placement_pose)

        elif isinstance(action, HandOver):
            self._step_hand_over()

        else:
            raise NotImplementedError

        # Apply held object transform.
        if self._current_held_object_id is not None:
            assert self._current_held_object_transform is not None
            world_to_robot = self.robot.get_end_effector_pose()
            world_to_object = multiply_poses(
                world_to_robot, self._current_held_object_transform
            )
            set_pose(
                self._current_held_object_id,
                world_to_object,
                self.physics_client_id,
            )

        return self._get_obs(), 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None

    def set_state(self, state: ObjectCentricState) -> None:
        """Sync the simulation to the given state."""
        # Set the robot state.
        default_robot_base_pose = self.robot.get_base_pose()
        robot_base_x = state.get(ROBOT_OBJECT, "base_x")
        robot_base_y = state.get(ROBOT_OBJECT, "base_y")
        robot_base_rot = state.get(ROBOT_OBJECT, "base_rot")
        new_robot_base_pose = Pose.from_rpy(
            (robot_base_x, robot_base_y, default_robot_base_pose.position[2]),
            (
                default_robot_base_pose.rpy[0],
                default_robot_base_pose.rpy[1],
                robot_base_rot,
            ),
        )
        self.robot.set_base(new_robot_base_pose)
        # Set the object states.
        all_pose_feats = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        for obj, feats in self._object_to_state_set_features.items():
            pybullet_id = self._object_to_pybullet_id[obj]
            current_sim_pose = get_pose(pybullet_id, self.physics_client_id)
            feat_vals = list(current_sim_pose.position) + list(
                current_sim_pose.orientation
            )
            for feat in feats:
                feat_idx = all_pose_feats.index(feat)
                feat_vals[feat_idx] = state.get(obj, feat)
            pose = Pose(
                (feat_vals[0], feat_vals[1], feat_vals[2]),
                (feat_vals[3], feat_vals[4], feat_vals[5], feat_vals[6]),
            )
            set_pose(pybullet_id, pose, self.physics_client_id)

    def _get_obs(self) -> ObjectCentricState:

        # Get the robot state.
        robot_base_pose = self.robot.get_base_pose()
        robot_state_dict = {
            "base_x": robot_base_pose.position[0],
            "base_y": robot_base_pose.position[1],
            "base_rot": robot_base_pose.rpy[2],
        }

        # Finish the state.
        state_dict: dict[Object, dict[str, float]] = {ROBOT_OBJECT: robot_state_dict}
        for obj, pybullet_id in self._object_to_pybullet_id.items():
            pose = get_pose(pybullet_id, self.physics_client_id)
            state_dict[obj] = {
                "x": pose.position[0],
                "y": pose.position[1],
                "z": pose.position[2],
                "qx": pose.orientation[0],
                "qy": pose.orientation[1],
                "qz": pose.orientation[2],
                "qw": pose.orientation[3],
            }
        return create_state_from_dict(state_dict, TYPE_FEATURES)

    def _step_move_base(self, new_pose: Pose) -> None:
        # Store the current robot pose in case we need to change it back.
        current_robot_base_pose = self.robot.get_base_pose()
        # Tentatively update the base pose.
        self.robot.set_base(new_pose)
        # Check for collisions.
        ignore_ids = (
            {self._current_held_object_id} if self._current_held_object_id else set()
        )
        if self._collision_exists(ignore_ids=ignore_ids):
            if self.raise_error_on_action_failures:
                raise ActionFailure("Robot in collision")
            self.robot.set_base(current_robot_base_pose)

    def _step_pick(
        self, object_name: str, end_effector_to_grasp_pose: Pose | None
    ) -> None:
        # Can only pick if hand is empty.
        if self._current_held_object_id is not None:
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot pick while holding something")
            return
        # Can only pick if object is reachable.
        # NOTE: currently assuming that reachable => gazeable.
        # Run inverse kinematics to determine grasp joint positions.
        object_id = self._object_name_to_id(object_name)
        target_object_pose = get_pose(object_id, self.physics_client_id)
        # Auto-compute grasp pose if not provided (top-down grasp).
        if end_effector_to_grasp_pose is None:
            end_effector_to_grasp_pose = self._compute_top_down_grasp_pose(object_name)
        target_end_effector_pose = multiply_poses(
            target_object_pose,
            end_effector_to_grasp_pose.invert(),
        )
        if not self._step_reach_end_effector_pose(
            target_end_effector_pose, collision_ignore_ids={object_id}
        ):
            return
        # Pick succeeds.
        self._current_held_object_id = object_id
        self._current_held_object_transform = end_effector_to_grasp_pose

    def _step_place(self, surface_name: str, pose: Pose) -> None:
        # Need to be holding something for place to be possible.
        if self._current_held_object_id is None:
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot place when hand is empty")
            return
        assert self._current_held_object_transform is not None
        target_end_effector_pose = multiply_poses(
            pose,
            self._current_held_object_transform.invert(),
        )
        if not self._step_reach_end_effector_pose(
            target_end_effector_pose,
            collision_ignore_ids={self._current_held_object_id},
        ):
            return
        # Tentatively set the object and measure distance to surface.
        current_held_object_pose = get_pose(
            self._current_held_object_id, self.physics_client_id
        )
        set_pose(self._current_held_object_id, pose, self.physics_client_id)
        surface_id = self._object_name_to_id(surface_name)
        # Check distance between held object and surface.
        closest_points = p.getClosestPoints(
            self._current_held_object_id,
            surface_id,
            distance=self.scene_description.placement_surface_threshold,
            physicsClientId=self.physics_client_id,
        )
        if not closest_points:
            set_pose(
                self._current_held_object_id,
                current_held_object_pose,
                self.physics_client_id,
            )
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot place when hand is empty")
            return
        # Placement succeeded.
        self._current_held_object_id = None
        self._current_held_object_transform = None

    def _step_hand_over(self) -> None:
        # Need to be holding something for handover to be possible.
        if self._current_held_object_id is None:
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot hand over when hand is empty")
            return
        # Hand over succeeds.
        set_pose(self._current_held_object_id, BANISH_POSE, self.physics_client_id)
        self._current_held_object_id = None
        self._current_held_object_transform = None

    def _step_reach_end_effector_pose(
        self, target_end_effector_pose: Pose, collision_ignore_ids: set[int]
    ) -> bool:
        """Attempt to reach end effector pose.

        Returns True on success.
        """
        current_robot_joints = self.robot.get_joint_positions()
        try:
            robot_joints = inverse_kinematics(self.robot, target_end_effector_pose)
        except InverseKinematicsError:
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot reach target")
            return False
        # Check for collisions.
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            robot_joints,
        )
        if self._collision_exists(ignore_ids=collision_ignore_ids):
            # Reset to original before returning.
            set_robot_joints_with_held_object(
                self.robot,
                self.physics_client_id,
                self._current_held_object_id,
                self._current_held_object_transform,
                current_robot_joints,
            )
            if self.raise_error_on_action_failures:
                raise ActionFailure("End effector would collide when reaching")
            return False
        # Reset to original.
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            current_robot_joints,
        )
        return True

    def _compute_top_down_grasp_pose(self, object_name: str) -> Pose:
        """Compute a top-down grasp pose for the given object.

        Returns the pose of the object relative to the end effector frame.
        """
        # Get object half extents to determine grasp offset.
        if object_name == "purple block":
            half_extents = self.scene_description.purple_block_half_extents
        elif object_name == "green block":
            half_extents = self.scene_description.green_block_half_extents
        else:
            # Default small offset for unknown objects.
            half_extents = (0.025, 0.025, 0.05)
        # Grasp from above: object is below the gripper (negative z in EE frame).
        # Add small offset above the object's top surface.
        grasp_offset_z = half_extents[2] + 0.02
        # The gripper points along its x-axis, so for a top-down grasp we need
        # to rotate so the gripper x-axis points down (world -z).
        # This is a 90-degree rotation about the y-axis.
        return Pose.from_rpy((0, 0, -grasp_offset_z), (0, np.pi / 2, 0))

    def _object_name_to_id(self, name: str) -> int:
        if name == "purple block":
            return self.purple_block_id
        if name == "green block":
            return self.green_block_id
        if name == "table":
            return self.table_id
        if name == "floor":
            return self.floor_id
        raise NotImplementedError

    def _collision_exists(self, ignore_ids: set[int] | None = None) -> bool:
        """Check for collisions between the robot (and held object) and obstacles."""
        ignore_ids = ignore_ids or set()
        collision_bodies = set(self.obstacle_ids) - ignore_ids
        return check_collisions_with_held_object(
            self.robot,
            collision_bodies,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            self.robot.get_joint_positions(),
        )
