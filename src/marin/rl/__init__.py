"""Marin RL public interface.

Only re-export *types* that are intended for public consumption.  Keeping
`__all__` small makes the import graph more manageable and avoids pulling in
heavy dependencies outside of training scripts.
"""

# Configs now live in .config
from .config import AbstractEnvConfig, MarinRlConfig, RlTrainingConfig

# Example environment configs
from .envs.hello import HelloEnvConfig
from .envs.math_env import MathEnvConfig
from .envs.openai_echo import ChatEchoEnvConfig
from .types import Rollout, RolloutGroup, RolloutSink, Turn

__all__ = [
    "AbstractEnvConfig",
    "ChatEchoEnvConfig",
    "HelloEnvConfig",
    "MarinRlConfig",
    "MathEnvConfig",
    "RlTrainingConfig",
    "Rollout",
    "RolloutGroup",
    "RolloutSink",
    "Turn",
]
