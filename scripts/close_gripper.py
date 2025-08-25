"""Close the spot gripper."""

from spot_planning_demo.spot_utils.skills.spot_hand_move import close_gripper
from spot_planning_demo.spot_utils.utils import initialize_robot_with_lease

if __name__ == "__main__":
    robot, _, _ = initialize_robot_with_lease()
    close_gripper(robot)
