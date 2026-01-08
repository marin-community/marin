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

"""Scaling recipes: model-specific hyperparameter bundles for scaling law experiments.

A ScalingRecipe defines the interface for scaling experiments. Concrete implementations
provide model-specific decisions for:
- Architecture formula (how to compute architecture from target param count)
- Model config building (returns LlamaConfig or subclass)
- Optimizer config building
- Candidate generation for isoflop sweeps

Orchestration logic (generating train args, predicting optimal configs) lives in
the library functions in isoflop_analysis.py, not in recipes.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

from levanter.models.llama import LlamaConfig
from levanter.optim.config import OptimizerConfig

if TYPE_CHECKING:
    from marin.scaling_laws.isoflop_analysis import CandidateConfig

# Default constants
DEFAULT_SEQ_LEN = 4096
DEFAULT_STEPS_PER_RUN = 2**16  # Reference step count for hyperparameter tuning
DEFAULT_FLOP_TOLERANCE = 0.01  # Relative error tolerance for FLOP budget


class ScalingRecipe(Protocol):
    """Protocol defining the interface for scaling law recipes.

    Concrete implementations (e.g., Marin2025Recipe) should implement these
    model-specific methods. Orchestration logic (generating training args,
    predicting optimal configs) is handled by library functions that use
    these core methods.
    """

    name: str
    """Name identifying this recipe (e.g., 'marin-2025')."""

    def build_model_config(self, target_params: int, vocab_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build a model config for a target parameter count."""
        ...

    def estimate_memory_bytes(self, model_config: LlamaConfig, batch_size: int, vocab_size: int) -> int:
        """Estimate memory usage in bytes for training with this model config."""
        ...

    def build_optimizer_config(self, candidate: "CandidateConfig", vocab_size: int) -> OptimizerConfig:
        """Build optimizer config for a candidate."""
        ...

    def candidate_configs(
        self,
        budget: float,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "Iterator[CandidateConfig]":
        """Yield candidate configurations within the FLOP budget."""
        ...
