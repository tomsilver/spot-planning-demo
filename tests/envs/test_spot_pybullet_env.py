"""Tests for spot_pybullet_env.py."""

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose, get_pose

from spot_planning_demo.envs.spot_pybullet_env import ActionFailure, SpotPyBulletSim
from spot_planning_demo.structs import HandOver, MoveBase, Pick


def test_spot_pybullet_sim():
    """Tests for SpotPyBulletSim()."""
    sim = SpotPyBulletSim(
        use_gui=False, raise_error_on_action_failures=True
    )  # change use_gui to True for debugging
    sim.reset(seed=123)
    drop_zone_pose = get_pose(sim.drop_zone_id, sim.physics_client_id)

    # Test a sequence of actions.
    action_sequence = [
        Pick("block", Pose.from_rpy((0, 0, 0.1), (0, np.pi, 0))),
        MoveBase(Pose.from_rpy((-1.0, 0.0, 0.0), (0.0, 0.0, -np.pi / 2))),
        HandOver(drop_zone_pose),
    ]

    # For now, just make sure this doesn't crash.
    for action in action_sequence:
        sim.step(action)

        # Uncomment to debug.
        # import pybullet as p
        # import time
        # for _ in range(1000):
        #     p.getMouseEvents(sim.physics_client_id)
        #     time.sleep(0.001)

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(Pose.identity(), sim.physics_client_id)
    # while True:
    #     p.getMouseEvents(sim.physics_client_id)


def test_spot_pybullet_move_base():
    """Tests for MoveBase()."""
    sim = SpotPyBulletSim(
        use_gui=False, raise_error_on_action_failures=True
    )  # change use_gui to True for debugging
    sim.reset(seed=123)

    # Moving to the origin should be possible.
    move_to_origin = MoveBase(Pose.identity())
    sim.step(move_to_origin)

    # Moving into the table should not be possible because of collisions.
    table_pose = get_pose(sim.table_id, sim.physics_client_id)
    move_into_table = MoveBase(
        Pose((table_pose.position[0], table_pose.position[1], 0))
    )
    with pytest.raises(ActionFailure):
        sim.step(move_into_table)


def test_spot_pybullet_pick():
    """Tests for Pick()."""
    sim = SpotPyBulletSim(
        use_gui=False, raise_error_on_action_failures=True
    )  # change use_gui to True for debugging
    sim.reset(seed=123)

    # Picking the block from the origin should be possible.
    sim.robot.set_base(Pose.identity())
    sim.step(Pick("block", Pose.from_rpy((0, 0, 0.1), (0, np.pi, 0))))
    assert sim._current_held_object_id is not None  # pylint: disable=protected-access

    # Picking from the origin with a side grasp should also be possible.
    sim.reset(seed=123)
    sim.robot.set_base(Pose.identity())
    sim.step(Pick("block", Pose.from_rpy((0, 0, 0.1), (-np.pi / 2, -np.pi / 2, 0))))
    assert sim._current_held_object_id is not None  # pylint: disable=protected-access

    # Picking the block from further back should not be possible.
    sim.reset(seed=123)
    sim.robot.set_base(Pose((-1.0, 0.0, 0.0)))

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(Pose.identity(), sim.physics_client_id)
    # while True:
    #     p.getMouseEvents(sim.physics_client_id)

    with pytest.raises(ActionFailure):
        sim.step(Pick("block", Pose.from_rpy((0, 0, 0.1), (0, np.pi, 0))))


def test_spot_pybullet_handover():
    """Tests for HandOver()."""
    sim = SpotPyBulletSim(
        use_gui=False, raise_error_on_action_failures=True
    )  # change use_gui to True for debugging
    sim.reset(seed=123)

    # It should be possible to hand over back to the pose where the block started, with
    # a little bit of padding added to avoid issues with table collisions.
    init_block_pose = get_pose(sim.block_id, sim.physics_client_id)
    good_handover_pose = Pose(
        (
            init_block_pose.position[0],
            init_block_pose.position[1],
            init_block_pose.position[2] + 1e-1,
        ),
        init_block_pose.orientation,
    )
    sim.robot.set_base(Pose.identity())
    sim.step(Pick("block", Pose.from_rpy((0, 0, 0.1), (-np.pi / 2, -np.pi / 2, 0))))
    sim.step(HandOver(good_handover_pose))

    # It should be impossible to hand over to a far-away pose.
    far_pose = Pose((1000, 1000, 0))
    sim.reset(seed=123)
    sim.robot.set_base(Pose.identity())
    sim.step(Pick("block", Pose.from_rpy((0, 0, 0.1), (-np.pi / 2, -np.pi / 2, 0))))
    with pytest.raises(ActionFailure):
        sim.step(HandOver(far_pose))
