"""Tests for spot_pybullet_env.py."""

from spot_planning_demo.envs.spot_pybullet_env import SpotPyBulletSim


def test_spot_pybullet_sim():
    """Tests for SpotPyBulletSim()."""
    sim = SpotPyBulletSim()
    assert isinstance(sim, SpotPyBulletSim)
