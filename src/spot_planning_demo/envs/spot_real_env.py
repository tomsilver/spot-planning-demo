"""Real environment for Spot that mirrors spot_pybullet_env."""

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeAlias

import gymnasium
from pybullet_helpers.geometry import Pose

from spot_planning_demo.structs import BANISH_POSE, HandOver, MoveBase, Pick, SpotAction

ObsType: TypeAlias = Any  # coming soon
RenderFrame: TypeAlias = Any


@dataclass(frozen=True)
class SpotRealEnvSpec:
    """Scene description for SpotRealEnv()."""



class SpotRealEnv(gymnasium.Env[ObsType, SpotAction]):
    """Real environment for Spot that mirrors spot_pybullet_env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        scene_description: SpotRealEnvSpec = SpotRealEnvSpec(),
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scene_description = scene_description

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        return None, {}

    def step(
        self, action: SpotAction
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if isinstance(action, MoveBase):
            self._step_move_base(action.pose)

        elif isinstance(action, Pick):
            self._step_pick(action.object_name, action.end_effector_to_grasp_pose)

        elif isinstance(action, HandOver):
            self._step_hand_over(action.pose)

        else:
            raise NotImplementedError

        return None, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None

    def _step_move_base(self, new_pose: Pose) -> None:
        pass


    def _step_pick(self, object_name: str, end_effector_to_grasp_pose: Pose) -> None:
        pass

    def _step_hand_over(self, pose: Pose) -> None:
        pass
