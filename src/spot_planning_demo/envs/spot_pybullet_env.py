"""PyBullet simulator and environment for Spot."""

from typing import Any, SupportsFloat, TypeAlias
from pybullet_helpers.gui import create_gui_connection
import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from dataclasses import dataclass

import gymnasium

ObsType: TypeAlias = Any  # coming soon
ActType: TypeAlias = Any  # coming soon
RenderFrame: TypeAlias = Any  # coming soon


@dataclass(frozen=True)
class SpotPybulletSimSpec:
    """Scene description forSpotPyBulletSim()."""

    robot_base_pose: Pose = Pose.identity()

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        return {
            "camera_target": (
                self.robot_base_pose.position[0],
                self.robot_base_pose.position[1],
                self.robot_base_pose.position[2] + 0.5
            ),
            "camera_yaw": 0,
            "camera_distance": 1.5,
            "camera_pitch": -20,
            "background_rgb": (255, 255, 255),
            # Use for fast testing.
            # "image_width": 32,
            # "image_height": 32,
        }


class SpotPyBulletSim(gymnasium.Env[ObsType, ActType]):
    """PyBullet simulator for Spot demo environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self,
                 scene_description: SpotPybulletSimSpec | None = SpotPybulletSimSpec(),
                 render_mode: str | None = "rgb_array",
        use_gui: bool = False):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scene_description = scene_description

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
            base_pose=self.scene_description.robot_base_pose,
            control_mode="reset",
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Coming soon."""
        return None, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Coming soon."""
        return None, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None
