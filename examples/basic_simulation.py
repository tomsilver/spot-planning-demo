"""Basic usage of the Spot simulation environment with skills.

This example demonstrates:
- Creating and resetting the PyBullet simulation environment
- Moving the robot base (MoveBase action)
- Picking up objects (Pick action)
- Placing objects on surfaces (Place action)
- Handing over objects (HandOver action)
- Querying the object-centric state
"""

from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_pybullet_env import SpotPyBulletSim
from spot_planning_demo.structs import (
    ROBOT_OBJECT,
    TIGER_TOY_OBJECT,
    HandOver,
    MoveBase,
    Pick,
    Place,
)


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
    input("Press enter to continue")

    # Move the robot to a new position
    target_pose = Pose((2.0, -0.2, robot_z))
    move_action = MoveBase(target_pose)
    obs, _, _, _, _ = env.step(move_action)
    print(f"\nAfter MoveBase to x={target_pose.position[0]}:")
    print(f"  Robot x: {obs.get(ROBOT_OBJECT, 'base_x'):.2f}")
    print(f"  Robot y: {obs.get(ROBOT_OBJECT, 'base_y'):.2f}")
    input("Press enter to continue")

    # Return to original position for picking
    original_pose = env.scene_description.robot_base_pose
    obs, _, _, _, _ = env.step(MoveBase(original_pose))
    print("\nReturned to original position")
    input("Press enter to continue")

    # Pick up the purple block (tiger toy)
    obs, _, _, _, _ = env.step(Pick("purple block"))
    print("\nAfter Pick:")
    print(
        f"  Tiger toy position: ({obs.get(TIGER_TOY_OBJECT, 'x'):.2f}, "
        f"{obs.get(TIGER_TOY_OBJECT, 'y'):.2f}, {obs.get(TIGER_TOY_OBJECT, 'z'):.2f})"
    )
    input("Press enter to continue")

    # Place the block at a new location on the table
    table_surface_z = (
        env.scene_description.table_pose.position[2]
        + env.scene_description.table_half_extents[2]
    )
    place_z = table_surface_z + env.scene_description.purple_block_half_extents[2]
    place_pose = Pose((2.55, 0.1, place_z))
    obs, _, _, _, _ = env.step(Place("table", place_pose))
    print("\nAfter Place:")
    print(
        f"  Tiger toy position: ({obs.get(TIGER_TOY_OBJECT, 'x'):.2f}, "
        f"{obs.get(TIGER_TOY_OBJECT, 'y'):.2f}, {obs.get(TIGER_TOY_OBJECT, 'z'):.2f})"
    )
    input("Press enter to continue")

    # Pick it up again
    obs, _, _, _, _ = env.step(Pick("purple block"))
    print("\nPicked up again")
    input("Press enter to continue")

    # Hand over the object (simulates giving to a human)
    obs, _, _, _, _ = env.step(HandOver())
    print("\nAfter HandOver:")
    print(obs.pretty_str())
    input("Press enter to continue")

    env.close()  # type: ignore[no-untyped-call]
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
