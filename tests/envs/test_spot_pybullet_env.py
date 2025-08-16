"""Tests for spot_pybullet_env.py."""

from spot_planning_demo.envs.spot_pybullet_env import SpotPyBulletSim
from spot_planning_demo.structs import MoveBase, Pick
from pybullet_helpers.geometry import Pose
import numpy as np


def test_spot_pybullet_sim():
    """Tests for SpotPyBulletSim()."""
    sim = SpotPyBulletSim(use_gui=True)  # change use_gui to True for debugging
    assert isinstance(sim, SpotPyBulletSim)

    sim.reset(seed=123)
    # drop_zone_pose = get_pose(sim.drop_zone_id, sim.physics_client_id)

    # Test a sequence of actions.
    action_sequence = [
        Pick("block"),
        MoveBase(Pose.from_rpy((-1.0, 0.0, 0.0), (0.0, 0.0, -np.pi / 2))),
        # HandOver(drop_zone_pose)
    ]

    # For now, just make sure this doesn't crash.
    for action in action_sequence:
        sim.step(action)

        # Uncomment to debug.
        import pybullet as p
        import time
        for _ in range(1000):
            p.getMouseEvents(sim.physics_client_id)
            time.sleep(0.001)


    # Uncomment to debug.
    import pybullet as p
    from pybullet_helpers.gui import visualize_pose
    visualize_pose(Pose.identity(), sim.physics_client_id)
    while True:
        p.getMouseEvents(sim.physics_client_id)
