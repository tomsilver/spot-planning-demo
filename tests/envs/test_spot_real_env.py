"""Tests for spot_real_env.py.

NOTE: all these tests are disabled in CI.
"""

import pytest
from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_real_env import SpotRealEnv
from spot_planning_demo.envs.spot_sim_viz_wrapper import SpotSimVizWrapper
from spot_planning_demo.structs import MoveBase

runreal = pytest.mark.skipif("not config.getoption('runreal')")
import numpy as np


@runreal
def test_spot_pybullet_move_base():
    """Tests for MoveBase()."""
    env = SpotRealEnv()
    env.reset(seed=123)

    # Create a GUI visualizer.
    env = SpotSimVizWrapper(env)

    # Moving to the origin should be possible.
    move_to_origin = MoveBase(Pose.identity())
    env.step(move_to_origin)

    # Turn counterclockwise.
    move_to_face_left = MoveBase(Pose.from_rpy((0, 0, 0), (0, 0, np.pi / 2)))
    env.step(move_to_face_left)

    # Uncomment to debug.
    import pybullet as p
    while True:
        p.getMouseEvents(env._sim.physics_client_id)
