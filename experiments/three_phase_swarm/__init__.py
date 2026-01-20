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

"""N-domain, n-phase data mixture experiments.

This package provides infrastructure for running data mixture experiments with
arbitrary numbers of domains and training phases, using RegMix-style Dirichlet
sampling for mixture weights.

Key components:
- config.py: Core configuration classes (Domain, PhaseSchedule, WeightConfig, etc.)
- domains.py: Registry of reusable domain definitions
- weight_sampler.py: Dirichlet-based weight sampling
- experiment.py: MixtureExperiment base class for running experiments

Example experiments:
- three_phase_experiment.py: 3-domain, 3-phase experiment (pretrain/midtrain/SFT)
"""

from experiments.three_phase_swarm.config import (
    DatasetComponent,
    Domain,
    PhaseConfig,
    PhaseSchedule,
    WeightConfig,
    ExperimentConfig,
)
from experiments.three_phase_swarm.domains import (
    register_domain,
    get_domain,
    list_domains,
    get_three_partition_domains,
    get_two_partition_domains,
    NEMOTRON_HQ_DOMAIN,
    NEMOTRON_FULL_DOMAIN,
    FINEWEB_EDU_DOMAIN,
    MATH_SFT_DOMAIN,
    GENERAL_SFT_DOMAIN,
)
from experiments.three_phase_swarm.weight_sampler import (
    DirichletSamplingParams,
    WeightSampler,
)
from experiments.three_phase_swarm.experiment import MixtureExperiment

__all__ = [
    "FINEWEB_EDU_DOMAIN",
    "GENERAL_SFT_DOMAIN",
    "MATH_SFT_DOMAIN",
    "NEMOTRON_FULL_DOMAIN",
    "NEMOTRON_HQ_DOMAIN",
    "DatasetComponent",
    "DirichletSamplingParams",
    "Domain",
    "ExperimentConfig",
    "MixtureExperiment",
    "PhaseConfig",
    "PhaseSchedule",
    "WeightConfig",
    "WeightSampler",
    "get_domain",
    "get_three_partition_domains",
    "list_domains",
    "register_domain",
]
