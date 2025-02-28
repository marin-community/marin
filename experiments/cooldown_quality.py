"""Methods to evaluate the quality of data for model cooldown.

This experiment fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% Target candidate dataset split (variable per experiment)
- 15% Dolma/FLAN dataset

The goal is to systematically compare different candidate datasets
and determine their relative contributions to model performance.
"""

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config

NUM_ANNEAL_TOKENS: int = 50_000_000_000


def default_quality_ablation(candidate_tokenized: TokenizerStep, tpu_type: str = "v5litepod-128") -> ExecutorStep:
    """
    Evaluates the quality of the candidate_tokenized dataset by annealing the learning rate of a Tootsie run
    on a data mixture designed to highlight the possible benefits of the candidate on dowstream benchmarks.

    Args:
        candidate_tokenized: The tokenized candidate dataset to evaluate
        tpu_type: The TPU type to use for training

    Returns:
        Annealed model on the ablation mixture
    """
    dolmino_dclm = dclm_components_llama3["dclm_baseline"]
    flan = tokenize_dolma_steps["dolma/flan"]
    name = candidate_tokenized.name.replace("/", "-")
    dataset_config = lm_mixture_data_config(
        components={
            "dclm": dolmino_dclm,
            "flan": flan,
            "candidate": candidate_tokenized,
        },
        weights={
            "dclm": 0.7,
            "flan": 0.15,
            "candidate": 0.15,
        },
    )
    hq_anneal_config = AnnealConfig(
        dataset_config=dataset_config, num_anneal_training_tokens=NUM_ANNEAL_TOKENS, tpu_type=tpu_type
    )

    hq_anneal_model = default_anneal(
        name=f"8b-quality-eval-{name}",
        anneal_config=hq_anneal_config,
    )
    return hq_anneal_model
