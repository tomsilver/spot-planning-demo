"""PyBullet simulator and environment for Spot."""

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeAlias

import gymnasium
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
    TYPE_FEATURES,
    HandOver,
    MoveBase,
    Pick,
    Place,
    SpotAction,
    ROBOT_OBJECT,
)

RenderFrame: TypeAlias = Any


class ActionFailure(BaseException):
    """Raised in step() if raise_error_on_action_failures=True."""


@dataclass(frozen=True)
class SpotPybulletSimSpec:
    """Scene description for SpotPyBulletSim()."""

    # Robot, with pose taken from real.
    robot_base_pose: Pose = Pose.from_rpy((2.287, -0.339, 0), (0, 0, 1.421))

    # Floor.
    floor_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    floor_half_extents: tuple[float, float, float] = (3, 3, 0.001)
    floor_pose: Pose = Pose(
        (
            robot_base_pose.position[0],
            robot_base_pose.position[1],
            -floor_half_extents[2],
        )
    )

    # Table.
    table_half_extents: tuple[float, float, float] = (0.2, 0.1, 0.05)
    table_pose: Pose = Pose((2.3, 0.7, table_half_extents[2]))
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

    # Purple block.
    purple_block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.05)
    purple_block_init_pose: Pose = Pose(
        (
            table_pose.position[0] - table_half_extents[0] / 2,
            table_pose.position[1] - table_half_extents[1] / 2,
            table_pose.position[2]
            + table_half_extents[2]
            + purple_block_half_extents[2],
        )
    )
    purple_block_color: tuple[float, float, float, float] = (
        170 / 255,
        121 / 255,
        222 / 255,
        1.0,
    )

    # Green block.
    # This is currently disabled (banish pose).
    green_block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.05)
    green_block_init_pose: Pose = BANISH_POSE
    # Comment this back to make it active again.
    # green_block_init_pose: Pose = Pose(
    #     (
    #         table_pose.position[0] - table_half_extents[0] / 2,
    #         table_pose.position[1] + table_half_extents[1] / 2,
    #         table_pose.position[2]
    #         + table_half_extents[2]
    #         + green_block_half_extents[2],
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
            "camera_yaw": 0,
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

        # Designate obstacles.
        self.obstacle_ids = {
            self.purple_block_id,
            self.green_block_id,
            self.table_id,
            self.shelf_ceiling_id,
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
            self._step_hand_over(action.pose)

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

    def _get_obs(self) -> ObjectCentricState:

        # Get the robot state.
        robot_base_pose = self.robot.get_base_pose()
        robot_state_dict = {
            "base_x": robot_base_pose.position[0],
            "base_y": robot_base_pose.position[1],
            "base_rot": robot_base_pose.rpy[2],
        }

        # Finish the state.
        state_dict: dict[Object, dict[str, float]] = {
            ROBOT_OBJECT: robot_state_dict
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

    def _step_pick(self, object_name: str, end_effector_to_grasp_pose: Pose) -> None:
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
        target_end_effector_pose = multiply_poses(
            target_object_pose,
            end_effector_to_grasp_pose.invert(),
        )
        self._step_reach_end_effector_pose(
            target_end_effector_pose, collision_ignore_ids={object_id}
        )
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
        assert self._current_held_object_transform is not None
        target_end_effector_pose = multiply_poses(
            pose,
            self._current_held_object_transform.invert(),
        )
        self._step_reach_end_effector_pose(
            target_end_effector_pose,
            collision_ignore_ids={self._current_held_object_id},
        )
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

    def _step_hand_over(self, pose: Pose) -> None:
        # Need to be holding something for handover to be possible.
        if self._current_held_object_id is None:
            if self.raise_error_on_action_failures:
                raise ActionFailure("Cannot hand over when hand is empty")
            return
        # Check reachability. The pose is in the object space, so transform to ee.
        assert self._current_held_object_transform is not None
        target_end_effector_pose = multiply_poses(
            pose,
            self._current_held_object_transform.invert(),
        )
        self._step_reach_end_effector_pose(
            target_end_effector_pose,
            collision_ignore_ids={self._current_held_object_id},
        )
        # Hand over succeeds.
        set_pose(self._current_held_object_id, BANISH_POSE, self.physics_client_id)
        self._current_held_object_id = None
        self._current_held_object_transform = None

    def _step_reach_end_effector_pose(
        self, target_end_effector_pose: Pose, collision_ignore_ids: set[int]
    ) -> None:
        current_robot_joints = self.robot.get_joint_positions()
        try:
            robot_joints = inverse_kinematics(self.robot, target_end_effector_pose)
        except InverseKinematicsError:
            if self.raise_error_on_action_failures:

                # Uncomment to debug.
                # from pybullet_helpers.gui import visualize_pose
                # current_ee = self.robot.get_end_effector_pose()
                # visualize_pose(current_ee, self.physics_client_id)
                # visualize_pose(target_end_effector_pose, self.physics_client_id)
                # while True:
                #     p.getMouseEvents(self.physics_client_id)

                raise ActionFailure("Cannot reach target")
        # Check for collisions.
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            robot_joints,
        )
        if self._collision_exists(ignore_ids=collision_ignore_ids):
            if self.raise_error_on_action_failures:

                # Uncomment to debug.
                # from pybullet_helpers.gui import visualize_pose
                # current_ee = self.robot.get_end_effector_pose()
                # visualize_pose(current_ee, self.physics_client_id)
                # visualize_pose(target_end_effector_pose, self.physics_client_id)
                # while True:
                #     p.getMouseEvents(self.physics_client_id)

                raise ActionFailure("End effector would collide when reaching")
        # Reset to original.
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            current_robot_joints,
        )

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
