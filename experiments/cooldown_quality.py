"""Methods to evaluate the quality of data for model cooldown.

This experiment fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% Target candidate dataset split (variable per experiment)
- 15% Dolma/FLAN dataset

The goal is to systematically compare different candidate datasets
and determine their relative contributions to model performance.
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional, Any, cast

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config, DataConfig


@dataclass
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
    tpu_type: str = "v5litepod-128"

    # Naming
    model_name_prefix: str = "8b-quality-eval"

    # Optional data mix name for when evaluating a data mix
    data_mix_name: Optional[str] = None

    def get_dataset_config(self, candidate: Union[TokenizerStep, DataConfig]) -> DataConfig:
        """Creates the dataset configuration for the ablation.

        For TokenizerStep inputs:
        - Creates a standard ablation mix (70% baseline, 15% MCQ, 15% candidate)

        For DataConfig inputs:
        - Merges the components from the data mix with baseline and MCQ
          components, preserving proportional weights within the candidate portion (15%)
        """
        if isinstance(candidate, DataConfig):
            # We're using a data mix
            # Extract components and weights from the data mix
            data_mix_components = cast(DataConfig, candidate).components
            data_mix_weights = cast(DataConfig, candidate).weights

            # Normalize the data mix weights to sum to 1.0
            total_weight = sum(data_mix_weights.values())
            normalized_weights = {k: v / total_weight for k, v in data_mix_weights.items()}

            # Scale the normalized weights by the candidate weight (15%)
            scaled_weights = {k: v * self.candidate_weight for k, v in normalized_weights.items()}

            # Create the combined components dict
            combined_components = {"baseline": self.baseline_component, "mcq": self.mcq_component, **data_mix_components}

            # Create the combined weights dict
            combined_weights = {"baseline": self.baseline_weight, "mcq": self.mcq_weight, **scaled_weights}

            # Return the combined data config
            return lm_mixture_data_config(components=combined_components, weights=combined_weights)
        else:
            # Original behavior for single dataset
            return lm_mixture_data_config(
                components={
                    "baseline": self.baseline_component,
                    "mcq": self.mcq_component,
                    "candidate": candidate,
                },
                weights={
                    "baseline": self.baseline_weight,
                    "mcq": self.mcq_weight,
                    "candidate": self.candidate_weight,
                },
            )

    def get_anneal_config(self, candidate: Union[TokenizerStep, DataConfig]) -> AnnealConfig:
        """Creates the anneal configuration for the ablation."""
        return AnnealConfig(
            dataset_config=self.get_dataset_config(candidate),
            num_anneal_training_tokens=self.num_anneal_tokens,
            tpu_type=self.tpu_type,
        )

    def get_model_name(self, candidate: Union[TokenizerStep, DataConfig]) -> str:
        """Generates the model name based on the candidate dataset or mix.

        Automatically detects the type of candidate:
        - For TokenizerStep: Uses the name of the candidate dataset
        - For DataConfig: Uses the provided data_mix_name if available, or a default name
        """
        if isinstance(candidate, DataConfig):
            # For data mixes, use the provided data_mix_name or a default
            mix_name = self.data_mix_name if self.data_mix_name else "data-mix"
            return f"{self.model_name_prefix}-{mix_name}"
        else:
            # For single datasets, use the original behavior
            name = candidate.name.replace("/", "-")
            return f"{self.model_name_prefix}-{name}"


def default_quality_ablation(
    candidate_tokenized: Union[TokenizerStep, DataConfig],
    config: Optional[QualityAblationConfig] = None,
    mix_name: Optional[str] = None,
) -> ExecutorStep:
    """
    Evaluates the quality of either a candidate dataset or a complete data mix by annealing the learning rate
    of a Tootsie run on a data mixture designed to highlight the possible benefits on downstream benchmarks.

    This function supports both individual datasets and complete data mixes:
    - For individual datasets: Creates a standard ablation mix (70% baseline, 15% MCQ, 15% candidate)
    - For data mixes: Merges the components of the data mix with baseline and MCQ components,
      preserving proportional weights within the candidate portion (15%)

    Args:
        candidate_tokenized: Either a TokenizerStep (single dataset) or a DataConfig (data mix)
        config: Configuration for the quality ablation (optional)
        mix_name: Optional name to use for the data mix (only if evaluating a mix)

    Returns:
        Annealed model on the ablation mixture
    """
    if config is None:
        config = QualityAblationConfig()

    # If mix_name is provided and we're using a DataConfig, set it in the config
    if mix_name is not None and isinstance(candidate_tokenized, DataConfig):
        config.data_mix_name = mix_name

    hq_anneal_config = config.get_anneal_config(candidate_tokenized)
    model_name = config.get_model_name(candidate_tokenized)

    hq_anneal_model = default_anneal(
        name=model_name,
        anneal_config=hq_anneal_config,
    )
    return hq_anneal_model
