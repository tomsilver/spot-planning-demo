"""A wrapper that updates a PyBullet GUI every time an observation is received."""

import gymnasium
from relational_structs import ObjectCentricState

from spot_planning_demo.envs.spot_pybullet_env import SpotAction


class SpotSimVizWrapper(
    gymnasium.ObservationWrapper[ObjectCentricState, SpotAction, ObjectCentricState]
):
    """A wrapper that updates a PyBullet GUI every time an observation is received."""

    def observation(self, observation: ObjectCentricState) -> ObjectCentricState:
        return observation
