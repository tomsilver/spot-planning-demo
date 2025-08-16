"""PyBullet simulator and environment for Spot."""

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeAlias

import gymnasium
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from spot_planning_demo.structs import BANISH_POSE, HandOver, MoveBase, Pick, SpotAction

ObsType: TypeAlias = Any  # coming soon
RenderFrame: TypeAlias = Any


class ActionFailure(BaseException):
    """Raised in step() if raise_error_on_action_failures=True."""


@dataclass(frozen=True)
class SpotPybulletSimSpec:
    """Scene description for SpotPyBulletSim()."""

    # Robot.
    robot_base_pose: Pose = Pose.identity()
    end_effector_to_grasp_pose: Pose = Pose((0.2, 0.0, 0.0))

    # Floor.
    floor_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    floor_half_extents: tuple[float, float, float] = (3, 3, 0.001)
    floor_pose: Pose = Pose((0, 0, -floor_half_extents[2]))

    # Table.
    table_half_extents: tuple[float, float, float] = (0.3, 0.4, 0.3)
    table_pose: Pose = Pose((0.9, 0.0, table_half_extents[2]))
    table_color: tuple[float, float, float, float] = (0.6, 0.3, 0.1, 1.0)

    # Block.
    block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)
    block_init_pose: Pose = Pose(
        (
            table_pose.position[0] - table_half_extents[0] / 2,
            table_pose.position[1],
            table_pose.position[2] + table_half_extents[2] + block_half_extents[2],
        )
    )
    block_color: tuple[float, float, float, float] = (
        170 / 255,
        121 / 255,
        222 / 255,
        1.0,
    )

    # Drop zone.
    drop_zone_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)
    drop_zone_pose: Pose = Pose((-0.75, -0.75, 0.75))
    drop_zone_color: tuple[float, float, float, float] = (
        255 / 255,
        121 / 255,
        255 / 255,
        0.5,
    )

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


class SpotPyBulletSim(gymnasium.Env[ObsType, SpotAction]):
    """PyBullet simulator for Spot demo environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

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

        # Create block.
        self.block_id = create_pybullet_block(
            self.scene_description.block_color,
            self.scene_description.block_half_extents,
            self.physics_client_id,
        )
        set_pose(
            self.block_id,
            self.scene_description.block_init_pose,
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
        self.obstacle_ids = {self.block_id, self.table_id}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        # Reset the robot.
        self.robot.set_base(self.scene_description.robot_base_pose)
        self.robot.close_fingers()

        # Reset the block.
        set_pose(
            self.block_id,
            self.scene_description.block_init_pose,
            self.physics_client_id,
        )

        # Reset the held object and transform.
        self._current_held_object_id = None
        self._current_held_object_transform = None

        return None, {}

    def step(
        self, action: SpotAction
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if isinstance(action, MoveBase):
            # Store the current robot pose in case we need to change it back.
            current_robot_base_pose = self.robot.get_base_pose()
            # Tentatively update the base pose.
            self.robot.set_base(action.pose)
            # Check for collisions.
            if self._collision_exists():
                if self.raise_error_on_action_failures:
                    raise ActionFailure("Robot in collision")
                self.robot.set_base(current_robot_base_pose)

        elif isinstance(action, Pick):
            # TODO: check gaze and reachability and hand empty
            self._current_held_object_id = self._object_name_to_id(action.object_name)
            self._current_held_object_transform = (
                self.scene_description.end_effector_to_grasp_pose
            )

        elif isinstance(action, HandOver):
            # TODO check reachability and held object
            assert self._current_held_object_id is not None
            set_pose(self._current_held_object_id, BANISH_POSE, self.physics_client_id)
            self._current_held_object_id = None
            self._current_held_object_transform = None

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

        return None, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None

    def _object_name_to_id(self, name: str) -> int:
        if name == "block":
            return self.block_id
        raise NotImplementedError

    def _collision_exists(self) -> bool:
        """Check for collisions between the robot (and held object) and obstacles."""
        collision_bodies = set(self.obstacle_ids)
        if self._current_held_object_id is not None:
            collision_bodies.discard(self._current_held_object_id)
        return check_collisions_with_held_object(
            self.robot,
            collision_bodies,
            self.physics_client_id,
            self._current_held_object_id,
            self._current_held_object_transform,
            self.robot.get_joint_positions(),
        )
