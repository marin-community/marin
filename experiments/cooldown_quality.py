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

"""Methods to evaluate the quality of data for model cooldown.

This experiment fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% Target candidate dataset split (variable per experiment)
- 15% Dolma/FLAN dataset

The goal is to systematically compare different candidate datasets
and determine their relative contributions to model performance.
"""

from dataclasses import dataclass


from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.pretraining_datasets import tokenize_dolma_steps
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config, PermutationType
from marin.resources import TpuPodConfig


@dataclass(frozen=True)
class QualityAblationConfig:
    """Configuration for quality ablation experiments."""

    # Dataset components and weights
    baseline_component: TokenizerStep = dclm_components_llama3["dclm_baseline"]
    mcq_component: TokenizerStep = tokenize_dolma_steps()["dolma/flan"]
    baseline_weight: float = 0.7
    mcq_weight: float = 0.15
    candidate_weight: float = 0.15

    # Training parameters
    num_anneal_tokens: int = 50_000_000_000
    resources: TpuPodConfig = TpuPodConfig(tpu_type="v5litepod-128")  # noqa: RUF009

    # Naming
    model_name_prefix: str = "8b-quality-eval"

    permutation_type: PermutationType = "feistel"

    def get_dataset_config(self, candidate_tokenized: TokenizerStep):
        """Creates the dataset configuration for the ablation."""
        return lm_mixture_data_config(
            components={
                "baseline": self.baseline_component,
                "mcq": self.mcq_component,
                "candidate": candidate_tokenized,
            },
            weights={
                "baseline": self.baseline_weight,
                "mcq": self.mcq_weight,
                "candidate": self.candidate_weight,
            },
            permutation_type=self.permutation_type,
        )

    def get_anneal_config(self, candidate_tokenized: TokenizerStep) -> AnnealConfig:
        """Creates the anneal configuration for the ablation."""
        return AnnealConfig(
            dataset_config=self.get_dataset_config(candidate_tokenized),
            num_anneal_training_tokens=self.num_anneal_tokens,
            resources=self.resources,
        )

    def get_model_name(self, candidate_tokenized: TokenizerStep) -> str:
        """Generates the model name based on the candidate dataset."""
        name = candidate_tokenized.name.replace("/", "-")
        return f"{self.model_name_prefix}-{name}"


def default_quality_ablation(
    candidate_tokenized: TokenizerStep, config: QualityAblationConfig | None = None
) -> ExecutorStep:
    """
    Evaluates the quality of the candidate_tokenized dataset by annealing the learning rate of a Tootsie run
    on a data mixture designed to highlight the possible benefits of the candidate on downstream benchmarks.

    Args:
        candidate_tokenized: The tokenized candidate dataset to evaluate
        config: Configuration for the quality ablation (optional)

    Returns:
        Annealed model on the ablation mixture
    """
    if config is None:
        config = QualityAblationConfig(permutation_type="linear")

    hq_anneal_config = config.get_anneal_config(candidate_tokenized)
    model_name = config.get_model_name(candidate_tokenized)

    hq_anneal_model = default_anneal(
        name=model_name,
        anneal_config=hq_anneal_config,
    )
    return hq_anneal_model
