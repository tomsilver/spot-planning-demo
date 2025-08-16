"""Common data structures."""

from dataclasses import dataclass

from pybullet_helpers.geometry import Pose

BANISH_POSE = Pose((-10000, -10000, -10000))


class SpotAction:
    """An executable action."""


@dataclass(frozen=True)
class MoveBase(SpotAction):
    """An action representing moving the robot base."""

    pose: Pose  # absolute pose in world frame


@dataclass(frozen=True)
class Pick(SpotAction):
    """Pick an object."""

    object_name: str  # unique object name
    end_effector_to_grasp_pose: Pose


@dataclass(frozen=True)
class HandOver(SpotAction):
    """Pick an object."""

    pose: Pose  # absolute pose in world frame
