"""Basic usage of the Spot simulation environment with skills.

This example demonstrates:
- Creating and resetting the PyBullet simulation environment
- Moving the robot base (MoveBase action)
- Querying the object-centric state

Note: The simulation environment uses PyBullet for physics. Pick and Place
actions require valid IK solutions which depend on robot positioning.
"""

from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_pybullet_env import SpotPyBulletSim
from spot_planning_demo.structs import ROBOT_OBJECT, TIGER_TOY_OBJECT, MoveBase


def main() -> None:
    """Demonstrate basic usage of the Spot simulation environment."""
    # Create the simulation environment (use_gui=True opens a PyBullet window)
    env = SpotPyBulletSim(use_gui=True)
    # Extract the constant robot z position.
    robot_z = env.scene_description.robot_base_pose.position[2]

    # Reset to initial state
    obs, _ = env.reset()
    print("Initial state:")
    print(obs.pretty_str())

    # Query specific object properties from the state
    robot_x = obs.get(ROBOT_OBJECT, "base_x")
    robot_y = obs.get(ROBOT_OBJECT, "base_y")
    tiger_x = obs.get(TIGER_TOY_OBJECT, "x")
    tiger_y = obs.get(TIGER_TOY_OBJECT, "y")
    print(f"\nRobot position: ({robot_x:.2f}, {robot_y:.2f})")
    print(f"Tiger toy position: ({tiger_x:.2f}, {tiger_y:.2f})")

    # Move the robot to a new position
    target_pose = Pose((2.0, -0.2, robot_z))
    move_action = MoveBase(target_pose)
    obs, _, _, _, _ = env.step(move_action)
    print(f"\nAfter MoveBase to x={target_pose.position[0]}:")
    print(f"  Robot x: {obs.get(ROBOT_OBJECT, 'base_x'):.2f}")
    print(f"  Robot y: {obs.get(ROBOT_OBJECT, 'base_y'):.2f}")
    input("Press enter to continue")

    # Move to another position
    second_pose = Pose((2.3, 0.0, robot_z))
    obs, _, _, _, _ = env.step(MoveBase(second_pose))
    print(f"\nAfter second MoveBase to x={second_pose.position[0]}:")
    print(f"  Robot x: {obs.get(ROBOT_OBJECT, 'base_x'):.2f}")
    print(f"  Robot y: {obs.get(ROBOT_OBJECT, 'base_y'):.2f}")
    input("Press enter to continue")

    # Return to original position
    original_pose = env.scene_description.robot_base_pose
    obs, _, _, _, _ = env.step(MoveBase(original_pose))
    print("\nReturned to original position:")
    print(obs.pretty_str())
    input("Press enter to continue")

    env.close()  # type: ignore[no-untyped-call]
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
