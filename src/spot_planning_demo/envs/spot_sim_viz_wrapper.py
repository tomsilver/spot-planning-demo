"""A wrapper that updates a PyBullet GUI every time an observation is received."""

import gymnasium
from relational_structs import ObjectCentricState

from spot_planning_demo.envs.spot_pybullet_env import SpotAction, SpotPyBulletSim
from spot_planning_demo.envs.spot_real_env import SpotRealEnv


class SpotSimVizWrapper(
    gymnasium.ObservationWrapper[ObjectCentricState, SpotAction, ObjectCentricState]
):
    """A wrapper that updates a PyBullet GUI every time an observation is received."""

    def __init__(self, env: SpotRealEnv | SpotPyBulletSim) -> None:
        super().__init__(env)
        self._sim = SpotPyBulletSim(use_gui=True)

    def observation(self, observation: ObjectCentricState) -> ObjectCentricState:
        print("Visualizing state:")
        print(observation.pretty_str())
        self._sim.set_state(observation)
        return observation
