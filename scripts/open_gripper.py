"""Open the spot gripper."""

from spot_planning_demo.spot_utils.skills.spot_hand_move import open_gripper
from spot_planning_demo.spot_utils.utils import initialize_robot_with_lease

if __name__ == "__main__":
    robot, _, _ = initialize_robot_with_lease()
    open_gripper(robot)
