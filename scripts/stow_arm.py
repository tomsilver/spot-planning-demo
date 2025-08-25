"""Stow the spot arm."""

from spot_planning_demo.spot_utils.skills.spot_hand_move import stow_arm
from spot_planning_demo.spot_utils.utils import initialize_robot_with_lease

if __name__ == "__main__":
    robot, _, _ = initialize_robot_with_lease()
    stow_arm(robot)
