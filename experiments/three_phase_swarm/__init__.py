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

"""Three-phase data mixture swarm experiments.

This package implements RegMix-style proxy model training with three training phases
(pretrain, midtrain, SFT), where mixture weights are sampled independently for each phase
using Dirichlet distributions.
"""

from experiments.three_phase_swarm.weight_sampler import (
    ThreePartitionWeightConfig,
    ThreePartitionWeightSampler,
)
from experiments.three_phase_swarm.datasets import (
    ALL_COMPONENTS,
    PRETRAIN_COMPONENTS,
    MIDTRAIN_COMPONENTS,
    SFT_COMPONENTS,
)

__all__ = [
    "ThreePartitionWeightConfig",
    "ThreePartitionWeightSampler",
    "ALL_COMPONENTS",
    "PRETRAIN_COMPONENTS",
    "MIDTRAIN_COMPONENTS",
    "SFT_COMPONENTS",
]
