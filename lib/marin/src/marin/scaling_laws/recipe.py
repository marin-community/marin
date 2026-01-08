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
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Protocol

from levanter.models.llama import LlamaConfig
from levanter.optim.config import OptimizerConfig

if TYPE_CHECKING:
    from marin.scaling_laws.isoflop_analysis import CandidateConfig, IsoFlopTrainArgs

# Default constants
DEFAULT_SEQ_LEN = 4096
DEFAULT_STEPS_PER_RUN = 2**16  # Reference step count for hyperparameter tuning
DEFAULT_FLOP_TOLERANCE = 0.01  # Relative error tolerance for FLOP budget


class ScalingRecipe(Protocol):
    """Protocol defining the interface for scaling law recipes.

    Concrete implementations (e.g., Marin2025Recipe) should implement all methods
    with their specific hyperparameters and formulas.
    """

    name: str
    """Name identifying this recipe (e.g., 'marin-2025')."""

    def compute_num_layers(self, hidden_size: int) -> int:
        """Compute number of layers from hidden size using the recipe's depth-width formula."""
        ...

    def hidden_size_for_params(self, target_params: int, vocab_size: int) -> int:
        """Find the hidden size that gives approximately target_params."""
        ...

    def build_model_config(self, target_params: int, vocab_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build a model config for a target parameter count."""
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

    def generate_isoflop_train_args(
        self,
        budgets: Sequence[float],
        experiment_name: str,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "list[IsoFlopTrainArgs]":
        """Generate training arguments for each candidate in an isoflop sweep."""
        ...

    def predict_optimal_config(
        self,
        scaling_fits: "dict[str, tuple[float, float]]",
        target_flops: float,
        label: str,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "CandidateConfig | None":
        """Predict optimal training config for a target compute budget using fitted scaling laws."""
        ...
