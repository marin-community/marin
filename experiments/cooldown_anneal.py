"""An experiment to cooldown a 8B model on a 30/70 mixture of high-quality sources and DCLM.

This is our initial guess of a good cooldown mixture dataset which is similar to the Olmo-2
cooldown dataset, but swapping their synthetic math datasets with finemath, and using different
wiki/pes2o/stackexchange splits from Dolma instead of Dolmino.
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step_llama3("dclm")

steps = {}

# High-quality sources
dolma_splits = [
    "dolma/algebraic-stack",
    "dolma/arxiv",
    "dolma/megawika",
    "dolma/open-web-math",
    "dolma/pes2o",
    "dolma/stackexchange",
    "dolma/wiki",
]
all_dolma_steps = tokenize_dolma_steps()
steps.update({dataset: step for dataset, step in all_dolma_steps.items() if dataset in dolma_splits})
steps["finemath_3_plus"] = finemath_3_plus_tokenized
steps["fineweb_edu"] = fineweb_edu_tokenized
steps["dclm"] = dolmino_dclm


# Dolma counts are done with llama tokens (https://docs.google.com/spreadsheets/d/1ykVJ1EGJvA1zwF67FZGFBzlm7P0ZBIMuCpBW9Pqp7cY/edit?gid=0#gid=0)
# This is slightly different from standard olmo tokenizer token counts
# The first number is the number of tokens in the dataset, the second is the desired mixing portion
high_quality_token_counts = {
    "dolma/algebraic-stack": 11.5 * 1.0,
    "dolma/arxiv": 27.9 * 1.0,
    "dolma/megawika": 4.44 * 1.0,
    "dolma/open-web-math": 5.06 * 1.0,
    "dolma/pes2o": 58.1 * 1.0,
    "dolma/stackexchange": 17.1 * 1.0,
    "dolma/wiki": 3.65 * 1.0,
    "finemath_3_plus": 34.0 * 1.0,  # https://huggingface.co/datasets/HuggingFaceTB/finemath
    "fineweb_edu": 0.0 * 1.0,  # 1.3T tokens total (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
}

total_high_quality_token_count = sum(high_quality_token_counts.values())

# reweight data so that 30% are high-quality sources and 70% are dclm
cooldown_mixture_weights = {
    **{
        dataset: 30 * token_count / total_high_quality_token_count
        for dataset, token_count in high_quality_token_counts.items()
    },
    "dclm": 70,
}

checkpoint_step = 668224
checkpoint_path = f"gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-{checkpoint_step}"

anneal_config = AnnealConfig(
    initialize_from_checkpoint_path=checkpoint_path,
    dataset_config=lm_mixture_data_config(
        components=steps,
        weights=cooldown_mixture_weights,
    ),
    num_anneal_training_tokens=100_000_000_000,
)

annealed_model = default_anneal(name="llama-8b-tootsie-cooldown", anneal_config=anneal_config)

if __name__ == "__main__":
    executor_main(
        steps=[
            annealed_model,
        ],
        description="Cooldown the model for Phase 2 using a 30/70 mixture of high-quality sources and DCLM.",
    )
