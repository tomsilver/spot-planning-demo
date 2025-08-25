"""Tests for spot_real_env.py.

NOTE: all these tests are disabled in CI.
"""

import pytest
from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_real_env import SpotRealEnv
from spot_planning_demo.envs.spot_sim_viz_wrapper import SpotSimVizWrapper
from spot_planning_demo.structs import TIGER_TOY_OBJECT, MoveBase, Pick

runreal = pytest.mark.skipif("not config.getoption('runreal')")


@runreal
def test_spot_real_custom_action_sequence():
    """Tests a custom action sequence."""
    env = SpotRealEnv()
    spec = env.scene_description
    env.reset(seed=123)

    # Create a GUI visualizer.
    env = SpotSimVizWrapper(env)

    # Pick the toy tiger. Assumes that it is in view.
    pick_tiger = Pick(TIGER_TOY_OBJECT.name)
    env.step(pick_tiger)

    # Move back to the origin.
    move_to_origin = MoveBase(spec.robot_base_pose)
    env.step(move_to_origin)


@runreal
def test_spot_real_move_base():
    """Tests for MoveBase()."""
    env = SpotRealEnv()
    spec = env.scene_description
    env.reset(seed=123)

    # Create a GUI visualizer.
    env = SpotSimVizWrapper(env)

    # Move backward a little bit.
    move_backward = MoveBase(
        Pose(
            (
                spec.robot_base_pose.position[0],
                spec.robot_base_pose.position[1] - 0.5,
                spec.robot_base_pose.position[2],
            ),
            spec.robot_base_pose.orientation,
        )
    )
    env.step(move_backward)

    # Move to the origin.
    move_to_origin = MoveBase(spec.robot_base_pose)
    env.step(move_to_origin)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env._sim.physics_client_id)


@runreal
def test_spot_real_pick():
    """Tests for Pick()."""
    env = SpotRealEnv()
    env.reset(seed=123)

    # Create a GUI visualizer.
    env = SpotSimVizWrapper(env)

    # Pick the toy tiger. Assumes that it is in view.
    pick_tiger = Pick(TIGER_TOY_OBJECT.name)
    env.step(pick_tiger)
