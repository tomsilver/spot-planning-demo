"""Tests for spot_real_env.py.

NOTE: all these tests are disabled in CI.
"""

import numpy as np
import pytest
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_real_env import SpotRealEnv
from spot_planning_demo.envs.spot_sim_viz_wrapper import SpotSimVizWrapper
from spot_planning_demo.structs import (
    HUMAN_OBJECT,
    ROBOT_OBJECT,
    TIGER_TOY_OBJECT,
    HandOver,
    MoveBase,
    Pick,
)

runreal = pytest.mark.skipif("not config.getoption('runreal')")


@runreal
def test_spot_real_custom_action_sequence():
    """Tests a custom action sequence."""
    env = SpotRealEnv()
    spec = env.scene_description
    obs, _ = env.reset(seed=123)

    # Create a GUI visualizer.
    env = SpotSimVizWrapper(env)

    # Pick the toy tiger. Assumes that it is in view.
    pick_tiger = Pick(TIGER_TOY_OBJECT.name)
    obs, _, _, _, _ = env.step(pick_tiger)

    # Move back to the origin.
    move_to_origin = MoveBase(spec.robot_base_pose)
    obs, _, _, _, _ = env.step(move_to_origin)

    # For fun: move to human.
    robot_x = obs.get(ROBOT_OBJECT, "base_x")
    robot_y = obs.get(ROBOT_OBJECT, "base_y")
    human_x = obs.get(HUMAN_OBJECT, "x")
    human_y = obs.get(HUMAN_OBJECT, "y")
    d = 1.0  # standoff distance [m]
    theta = np.arctan2(human_y - robot_y, human_x - robot_x)
    x = human_x - d * np.cos(theta)
    y = human_y - d * np.sin(theta)
    rot = wrap_angle(theta)
    base_pose_facing_human = Pose.from_rpy((x, y, 0), (0, 0, rot))
    move_to_human = MoveBase(base_pose_facing_human)
    env.step(move_to_human)

    # Hand over.
    env.step(HandOver())


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
