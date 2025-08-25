"""Real environment for Spot that mirrors spot_pybullet_env.

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
import numpy as np
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.util import authenticate
from prpl_perception_utils.object_detection_2d.base_object_detector_2d import (
    ObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import (
    GeminiObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.render_wrapper import (
    RenderWrapperObjectDetector2D,
)
from prpl_perception_utils.pose_detection_6d.simple_2d_pose_detector_6d import (
    Simple2DPoseDetector6D,
)
from prpl_perception_utils.structs import (
    CameraInfo,
    LanguageObjectDetectionID,
    RGBDImage,
)
from pybullet_helpers.geometry import Pose
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict
from spatialmath import SE3

from spot_planning_demo.spot_utils.perception.spot_cameras import capture_images
from spot_planning_demo.spot_utils.skills.spot_navigation import (
    navigate_to_absolute_pose,
)
from spot_planning_demo.spot_utils.spot_localization import SpotLocalizer
from spot_planning_demo.spot_utils.utils import verify_estop
from spot_planning_demo.structs import (
    CARDBOARD_TABLE_OBJECT,
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


@dataclass(frozen=True)
class SpotRealEnvSpec:
    """Scene description for SpotRealEnv()."""

    graph_nav_map: Path = (
        Path(__file__).parents[3] / "graph_nav_maps" / "prpl_fwing_test_map"
    )
    sdk_client_name: str = "SpotPlanningDemoClient"

    object_detector_artifact_path: Path | None = (
        Path(__file__).parents[3] / "object_detections"
    )

    def __post_init__(self) -> None:
        assert (
            self.graph_nav_map.exists()
        ), f"Graph nav map directory not found: {self.graph_nav_map}"
        if self.object_detector_artifact_path is not None:
            self.object_detector_artifact_path.mkdir(exist_ok=True)


class SpotRealEnv(gymnasium.Env[ObjectCentricState, SpotAction]):
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

        # Create the object pose detector.
        detector_2d: ObjectDetector2D = GeminiObjectDetector2D()
        if self.scene_description.object_detector_artifact_path is not None:
            detector_2d = RenderWrapperObjectDetector2D(
                detector_2d, self.scene_description.object_detector_artifact_path
            )
        self._pose_detector = Simple2DPoseDetector6D(detector_2d)
        self._pose_detector_object_ids = [
            LanguageObjectDetectionID(TIGER_TOY_OBJECT.name),
            LanguageObjectDetectionID(CARDBOARD_TABLE_OBJECT.name),
        ]

        # Track objects.
        self._last_known_object_poses: dict[Object, Pose] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:

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

        return self._get_obs(), 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon.

        Make sure to reuse the images captured in _get_obs().
        """
        return None

    def _get_obs(self) -> ObjectCentricState:

        # Get the robot state.
        robot_base_pose = self._get_robot_pose()
        robot_state_dict = {
            "base_x": robot_base_pose.position[0],
            "base_y": robot_base_pose.position[1],
            "base_rot": robot_base_pose.rpy[2],
        }

        # Capture images and get the object state.
        rgbds_with_context = capture_images(self.robot, self.localizer)

        # Detect objects.
        cam_to_rgbd: dict[CameraInfo, RGBDImage] = {}
        for camera_name, rgbd_with_context in rgbds_with_context.items():
            fx = rgbd_with_context.camera_model.intrinsics.focal_length.x
            fy = rgbd_with_context.camera_model.intrinsics.focal_length.y
            cx = rgbd_with_context.camera_model.intrinsics.principal_point.x
            cy = rgbd_with_context.camera_model.intrinsics.principal_point.y
            cam_info = CameraInfo(
                camera_name,
                (fx, fy),
                (cx, cy),
                rgbd_with_context.depth_scale,
                SE3(rgbd_with_context.world_tform_camera.to_matrix()),
            )
            cam_to_rgbd[cam_info] = RGBDImage(
                rgbd_with_context.rgb, rgbd_with_context.depth
            )
        detections = self._pose_detector.detect(
            cam_to_rgbd, self._pose_detector_object_ids
        )
        objects_to_track = {TIGER_TOY_OBJECT, CARDBOARD_TABLE_OBJECT}
        object_to_detected_poses: dict[Object, list[Pose]] = {
            o: [] for o in objects_to_track
        }
        object_id_to_object = {o.name: o for o in objects_to_track}
        for camera_detections in detections.values():
            for detection in camera_detections:
                assert isinstance(detection.object_id, LanguageObjectDetectionID)
                obj = object_id_to_object[detection.object_id.language_id]
                pose = Pose.from_rpy(
                    (detection.pose.x, detection.pose.y, detection.pose.z),
                    detection.pose.rpy(),
                )
                object_to_detected_poses[obj].append(pose)
        # Average poses or use the last known pose if none are found.
        for obj in objects_to_track:
            detected_poses = object_to_detected_poses[obj]
            if not detected_poses:
                assert obj in self._last_known_object_poses
                pose = self._last_known_object_poses[obj]
            else:
                mean_pos = np.mean([pose.position for pose in detected_poses], axis=0)
                mean_quat = np.mean(
                    [pose.orientation for pose in detected_poses], axis=0
                )
                pose = Pose(tuple(mean_pos), tuple(mean_quat))
            self._last_known_object_poses[obj] = pose

        # Finish the state.
        state_dict: dict[Object, dict[str, float]] = {ROBOT_OBJECT: robot_state_dict}
        for obj, pose in self._last_known_object_poses.items():
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
        desired_pose_se2 = SE2Pose(
            new_pose.position[0], new_pose.position[1], new_pose.rpy[2]
        )
        navigate_to_absolute_pose(self.robot, self.localizer, desired_pose_se2)

    def _step_pick(self, object_name: str, end_effector_to_grasp_pose: Pose) -> None:
        pass

    def _step_place(self, surface_name: str, pose: Pose) -> None:
        pass

    def _step_hand_over(self, pose: Pose) -> None:
        pass

    def _get_robot_pose(self) -> Pose:
        self.localizer.localize()
        world_se3_pose = self.localizer.get_last_robot_pose()
        world_se2_pose = world_se3_pose.get_closest_se2_transform()
        world_pose = Pose.from_rpy(
            (world_se2_pose.x, world_se2_pose.y, 0), (0, 0, world_se2_pose.angle)
        )
        return world_pose
