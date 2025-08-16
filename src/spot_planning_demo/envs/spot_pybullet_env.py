"""PyBullet simulator and environment for Spot."""

from typing import Any, SupportsFloat, TypeAlias

import gymnasium

ObsType: TypeAlias = Any  # coming soon
ActType: TypeAlias = Any  # coming soon
RenderFrame: TypeAlias = Any  # coming soon


class SpotPyBulletSim(gymnasium.Env[ObsType, ActType]):
    """PyBullet simulator for Spot demo environment."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Coming soon."""
        return None, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Coming soon."""
        return None, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Coming soon."""
        return None
