# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
