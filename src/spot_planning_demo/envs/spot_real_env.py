"""Real environment for Spot that mirrors spot_pybullet_env.

# TODO: CHANGE SIM TO MATCH THE BELOW CONVENTION

NOTE: the origin (0, 0, 0) is Spot facing towards the table, and:
    - +x is facing right
    - +y is facing forward
    - +z is facing up
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat, TypeAlias

import gymnasium
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.util import authenticate
from pybullet_helpers.geometry import Pose

from spot_planning_demo.spot_utils.skills.spot_navigation import (
    navigate_to_absolute_pose,
)
from spot_planning_demo.spot_utils.spot_localization import SpotLocalizer
from spot_planning_demo.spot_utils.utils import verify_estop
from spot_planning_demo.structs import HandOver, MoveBase, Pick, Place, SpotAction

ObsType: TypeAlias = Any  # coming soon
RenderFrame: TypeAlias = Any


@dataclass(frozen=True)
class SpotRealEnvSpec:
    """Scene description for SpotRealEnv()."""

    graph_nav_map: Path = (
        Path(__file__).parents[3] / "graph_nav_maps" / "prpl_fwing_test_map"
    )
    sdk_client_name: str = "SpotPlanningDemoClient"
    map_to_world_frame_tf: SE2Pose = SE2Pose(2.287, -0.339, 1.421)

    def __post_init__(self) -> None:
        assert (
            self.graph_nav_map.exists()
        ), f"Graph nav map directory not found: {self.graph_nav_map}"


class SpotRealEnv(gymnasium.Env[ObsType, SpotAction]):
    """Real environment for Spot that mirrors spot_pybullet_env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        scene_description: SpotRealEnvSpec = SpotRealEnvSpec(),
        render_mode: str | None = "rgb_array",
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scene_description = scene_description

        # Create the interface to the spot robot.
        sdk = create_standard_sdk(self.scene_description.sdk_client_name)
        if "BOSDYN_IP" not in os.environ:
            raise KeyError("BOSDYN_IP not found in os.environ")
        hostname = os.environ.get("BOSDYN_IP")
        self.robot = sdk.create_robot(hostname)
        authenticate(self.robot)
        verify_estop(self.robot)
        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(
            lease_client, must_acquire=True, return_at_exit=True
        )

        # Create the localizer.
        self.localizer = SpotLocalizer(
            self.robot,
            self.scene_description.graph_nav_map,
            lease_client,
            lease_keepalive,
        )
        self.robot.time_sync.wait_for_sync()
        self.localizer.localize()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        # TODO remove
        print("Current robot pose:", self._get_robot_pose())

        return None, {}

    def step(
        self, action: SpotAction
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

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

        # TODO remove
        print("Current robot pose:", self._get_robot_pose())

        return None, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None

    def _step_move_base(self, new_pose: Pose) -> None:
        desired_world_pose_se2 = SE2Pose(
            new_pose.position[0], new_pose.position[1], new_pose.rpy[2]
        )
        desired_localizer_pose_se2 = (
            self.scene_description.map_to_world_frame_tf * desired_world_pose_se2
        )
        navigate_to_absolute_pose(
            self.robot, self.localizer, desired_localizer_pose_se2
        )

    def _step_pick(self, object_name: str, end_effector_to_grasp_pose: Pose) -> None:
        pass

    def _step_place(self, surface_name: str, pose: Pose) -> None:
        pass

    def _step_hand_over(self, pose: Pose) -> None:
        pass

    def _get_robot_pose(self) -> Pose:
        self.localizer.localize()
        localizer_se3_pose = self.localizer.get_last_robot_pose()
        localizer_se2_pose = localizer_se3_pose.get_closest_se2_transform()
        world_se2_pose = (
            localizer_se2_pose * self.scene_description.map_to_world_frame_tf.inverse()
        )
        world_pose = Pose.from_rpy(
            (world_se2_pose.x, world_se2_pose.y, 0), (world_se2_pose.angle, 0, 0)
        )
        return world_pose
