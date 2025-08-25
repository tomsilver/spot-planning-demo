"""Tests for spot_real_env.py.

NOTE: all these tests are disabled in CI.
"""

import pytest
from pybullet_helpers.geometry import Pose

from spot_planning_demo.envs.spot_real_env import SpotRealEnv
from spot_planning_demo.structs import MoveBase

runreal = pytest.mark.skipif("not config.getoption('runreal')")


@runreal
def test_spot_pybullet_move_base():
    """Tests for MoveBase()."""
    env = SpotRealEnv()
    env.reset(seed=123)

    # Moving to the origin should be possible.
    move_to_origin = MoveBase(Pose.identity())
    env.step(move_to_origin)
