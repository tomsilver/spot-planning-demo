"""Tests for spot_pybullet_env.py."""

from spot_planning_demo.envs.spot_pybullet_env import SpotPyBulletSim


def test_spot_pybullet_sim():
    """Tests for SpotPyBulletSim()."""
    sim = SpotPyBulletSim(use_gui=False)  # TODO change to False
    assert isinstance(sim, SpotPyBulletSim)

    # TODO remove
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # from pybullet_helpers.geometry import Pose
    # visualize_pose(Pose.identity(), sim.physics_client_id)
    # while True:
    #     p.getMouseEvents(sim.physics_client_id)
