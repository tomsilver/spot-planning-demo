"""Common data structures."""

from dataclasses import dataclass

from pybullet_helpers.geometry import Pose
from relational_structs import Type

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
class Place(SpotAction):
    """Place a held object onto a surface."""

    surface_name: str  # unique object name
    placement_pose: Pose  # absolute object pose


@dataclass(frozen=True)
class HandOver(SpotAction):
    """Pick an object."""

    pose: Pose  # absolute pose in world frame


# Object types.
RobotType = Type("robot")
MovableObjectType = Type("movable_object")
TYPE_FEATURES = {
    RobotType: ["base_x", "base_y", "base_rot"],
    MovableObjectType: ["x", "y", "z", "qx", "qy", "qz", "qw"],
}
