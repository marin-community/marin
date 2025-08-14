"""Marin RL public interface.

Only re-export *types* that are intended for public consumption.  Keeping
`__all__` small makes the import graph more manageable and avoids pulling in
heavy dependencies outside of training scripts.
"""

# Configs now live in .config
from marin.rl.config import AbstractEnvConfig, MarinRlConfig, RlTrainingConfig
from marin.rl.datatypes import Rollout, RolloutGroup, RolloutSink, Turn

from marin.rl.env import AbstractMarinEnv
from marin.rl.envs.hello import HelloEnvConfig
from marin.rl.envs.openai_echo import ChatEchoEnvConfig

__all__ = [
    "AbstractEnvConfig",
    "AbstractMarinEnv",
    "ChatEchoEnvConfig",
    "HelloEnvConfig",
    "MarinRlConfig",
    "RlTrainingConfig",
    "Rollout",
    "RolloutGroup",
    "RolloutSink",
    "Turn",
]
