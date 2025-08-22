"""Small utility functions for spot."""

import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from bosdyn.api import estop_pb2, robot_state_pb2
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray

# Pose for the hand (relative to the body) that looks down in front.
DEFAULT_HAND_LOOK_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 6)
)
DEFAULT_HAND_DROP_OBJECT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)
DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3)
)
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE_HIGH = math_helpers.SE3Pose(
    x=0.65, y=0.0, z=0.32, rot=math_helpers.Quat.from_pitch(np.pi / 2.5)
)
DEFAULT_HAND_PRE_DUMP_LIFT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.3, rot=math_helpers.Quat.from_pitch(2 * np.pi / 3)
)
DEFAULT_HAND_PRE_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80,
    y=0.0,
    z=0.25,
    rot=math_helpers.Quat.from_pitch(np.pi / 2)
    * math_helpers.Quat.from_yaw(np.pi / 1.1),
)
DEFAULT_HAND_POST_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)


def get_graph_nav_dir(map_room_name: str) -> Path:
    """Get the path to the graph nav directory."""
    upload_dir = Path(__file__).parent / "graph_nav_maps"
    return upload_dir / map_room_name


def verify_estop(robot: Robot) -> None:
    """Verify the robot is not estopped."""
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = (
            "Robot is estopped. Please use an external"
            + " E-Stop client, such as the estop SDK example, to"
            + " configure E-Stop."
        )
        robot.logger.error(error_message)
        raise Exception(error_message)


def get_pixel_from_user(rgb: NDArray[np.uint8]) -> Tuple[int, int]:
    """Use open CV GUI to select a pixel on the given image."""
    image_click: Optional[Tuple[int, int]] = None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _callback(event: int, x: int, y: int, flags: int, param: None) -> None:
        """Callback for the click-to-grasp functionality with the Spot API's grasping
        interface."""
        del flags, param
        nonlocal image_click
        if event == cv2.EVENT_LBUTTONUP:
            image_click = (x, y)

    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, _callback)
    cv2.imshow(image_title, bgr)

    while image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            # Quit and terminate the process (if you're panicking.)
            sys.exit()

    cv2.destroyAllWindows()

    return image_click


def get_pixel_from_grounded_sam(
    rgb: NDArray[np.uint8], text_prompt: str, endpoint_url: str
) -> Optional[Tuple[int, int]]:
    """Pick a pixel in rgb that matches text_prompt.

    This function queries a server hosting a GroundedSAM instance (e.g.
    https://github.com/chsahit/GroundedSAMEndpoint),
    in order to get an image mask corresponding to the given text.
    It then computes the center of this mask and returns this pixel.
    This assumes the object matching text_prompt contains its center.
    """
    _, encoded_image = cv2.imencode(".png", rgb)
    image_bytes = io.BytesIO(encoded_image.tobytes())
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"text_prompt": text_prompt}
    response = requests.post(endpoint_url, files=files, data=data, timeout=100)

    # Check the response
    if response.status_code == 200:
        # Convert the received boolean masks back into NumPy arrays
        masks = [np.array(mask, dtype=bool) for mask in response.json()]
        y_coords, x_coords = np.where(masks[0])
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        pixel = (center_x, center_y)
        return pixel
    print(f"Error: {response.status_code}, {response.text}")
    return None


def get_relative_se2_from_se3(
    robot_pose: math_helpers.SE3Pose,
    target_pose: math_helpers.SE3Pose,
    target_offset_distance: float = 0.0,
    target_offset_angle: float = 0.0,
) -> math_helpers.SE2Pose:
    """Given a current se3 pose and a target se3 pose on the same plane, return a
    relative se2 pose for moving from the current to the target.

    Also add an angle and distance offset to the target pose. The returned se2 pose is
    facing toward the target. Typical use case: we know the current se3 pose for the
    body of the robot and the se3 pose for a table, and we want to move in front of the
    table.
    """
    dx = np.cos(target_offset_angle) * target_offset_distance
    dy = np.sin(target_offset_angle) * target_offset_distance
    x = target_pose.x + dx
    y = target_pose.y + dy
    # Face towards the target.
    rot = target_offset_angle + np.pi
    target_se2 = math_helpers.SE2Pose(x, y, rot)
    robot_se2 = robot_pose.get_closest_se2_transform()
    return robot_se2.inverse() * target_se2


def get_robot_state(
    robot: Robot, timeout_per_call: float = 20, num_retries: int = 10
) -> robot_state_pb2.RobotState:
    """Get the robot state."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    for _ in range(num_retries):
        try:
            robot_state = robot_state_client.get_robot_state(timeout=timeout_per_call)
            return robot_state
        except (TimedOutError, ProxyConnectionError):
            print("WARNING: get robot state failed once, retrying...")
    raise RuntimeError("get_robot_state() failed permanently.")


def get_robot_gripper_open_percentage(robot: Robot) -> float:
    """Get the current state of how open the gripper is."""
    robot_state = get_robot_state(robot)
    return float(robot_state.manipulator_state.gripper_open_percentage)
