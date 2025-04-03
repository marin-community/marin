"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/784

Evaluation Criteria:
-------------------
Success metrics tracked in the issue:
- MMLU (overall performance)
- Overall Paloma

Both metrics are part of the default_run evaluation loop.
"""

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

# Constants
DCLM_BASELINE = dclm_components_llama3["dclm_baseline"]
ANNEAL_TOKENS = 50_000_000_000
TPU_TYPE = "v5litepod-128"
DCLM_WEIGHT = 0.7
HIGH_QUALITY_WEIGHT = 0.3
FLAN_WEIGHT_15 = 0.15
BASE_MODEL_NAME = "llama-8b-anneal"

# Define high-quality Dolma datasets
DOLMA_HIGH_QUALITY_DATASETS = [
    "dolma/flan",
    "dolma/algebraic-stack",
    "dolma/arxiv",
    "dolma/megawika",
    "dolma/open-web-math",
    "dolma/pes2o",
]

# Token counts for high quality datasets (in billions)
HIGH_QUALITY_TOKEN_COUNTS = {
    "dolma/algebraic-stack": 11.5,
    "dolma/arxiv": 27.9,
    "dolma/megawika": 4.44,
    "dolma/open-web-math": 5.06,
    "dolma/pes2o": 58.1,
    "dolma/stackexchange": 17.1,
    "dolma/wiki": 3.65,
}

TOTAL_HIGH_QUALITY_TOKENS = sum(HIGH_QUALITY_TOKEN_COUNTS.values())

# Control model is 100% Dolmino DCLM dataset
control_dataset_config = lm_mixture_data_config(
    components={"dclm_baseline": DCLM_BASELINE},
    weights={"dclm_baseline": 1.0},
)

control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config, num_anneal_training_tokens=ANNEAL_TOKENS, tpu_type=TPU_TYPE
)

control_model = default_anneal(name=f"{BASE_MODEL_NAME}-control-eu-long", anneal_config=control_anneal_config)

# Initialize experiment steps list with control model
experiment_steps = [control_model]

# Get tokenized Dolma datasets
dolma_tokenized = tokenize_dolma_steps()
high_quality_datasets = [dolma_tokenized[dataset] for dataset in DOLMA_HIGH_QUALITY_DATASETS]

# Create experiments for individual high-quality datasets
for hq_dataset in high_quality_datasets:
    dataset_name = hq_dataset.name.replace("/", "-")

    # Experiment 1: 30% high-quality, 70% DCLM
    dataset_config = lm_mixture_data_config(
        components={"dclm": DCLM_BASELINE, "high_quality": hq_dataset},
        weights={"dclm": DCLM_WEIGHT, "high_quality": HIGH_QUALITY_WEIGHT},
    )

    anneal_config = AnnealConfig(
        dataset_config=dataset_config, num_anneal_training_tokens=ANNEAL_TOKENS, tpu_type=TPU_TYPE
    )

    model = default_anneal(
        name=f"{BASE_MODEL_NAME}-eu-{dataset_name}-long",
        anneal_config=anneal_config,
    )
    experiment_steps.append(model)

    # Experiment 2: 15% high-quality, 15% FLAN, 70% DCLM
    dataset_config_with_flan = lm_mixture_data_config(
        components={"dclm": DCLM_BASELINE, "high_quality": hq_dataset, "flan": dolma_tokenized["dolma/flan"]},
        weights={"dclm": DCLM_WEIGHT, "high_quality": FLAN_WEIGHT_15, "flan": FLAN_WEIGHT_15},
    )

    anneal_config_with_flan = AnnealConfig(
        dataset_config=dataset_config_with_flan, num_anneal_training_tokens=ANNEAL_TOKENS, tpu_type=TPU_TYPE
    )

    model_with_flan = default_anneal(
        name=f"{BASE_MODEL_NAME}-eu-{dataset_name}-long-w-flan15",
        anneal_config=anneal_config_with_flan,
    )
    experiment_steps.append(model_with_flan)

# Create mixture experiment with all high-quality datasets (15% all HQ, 15% FLAN, 70% DCLM)
mixture_weights_15pct_flan = {
    **{
        dataset: 15 * token_count / TOTAL_HIGH_QUALITY_TOKENS
        for dataset, token_count in HIGH_QUALITY_TOKEN_COUNTS.items()
    },
    "flan": 15,
    "dclm": 70,
}

all_hq_datasets = {dataset: dolma_tokenized[dataset] for dataset in HIGH_QUALITY_TOKEN_COUNTS.keys()}
all_hq_dataset_config = lm_mixture_data_config(
    components={"dclm": DCLM_BASELINE, "flan": dolma_tokenized["dolma/flan"], **all_hq_datasets},
    weights=mixture_weights_15pct_flan,
)

all_hq_anneal_config = AnnealConfig(
    dataset_config=all_hq_dataset_config, num_anneal_training_tokens=ANNEAL_TOKENS, tpu_type=TPU_TYPE
)

all_hq_model = default_anneal(
    name=f"{BASE_MODEL_NAME}-eu-all-dolma-hq-w-flan15",
    anneal_config=all_hq_anneal_config,
)
experiment_steps.append(all_hq_model)

# Create mixture experiment with all high-quality datasets (25% all HQ, 5% FLAN, 70% DCLM)
mixture_weights_5pct_flan = {
    **{
        dataset: 25 * token_count / TOTAL_HIGH_QUALITY_TOKENS
        for dataset, token_count in HIGH_QUALITY_TOKEN_COUNTS.items()
    },
    "flan": 5,
    "dclm": 70,
}

all_hq_dataset_config_5pct = lm_mixture_data_config(
    components={"dclm": DCLM_BASELINE, "flan": dolma_tokenized["dolma/flan"], **all_hq_datasets},
    weights=mixture_weights_5pct_flan,
)

all_hq_anneal_config_5pct = AnnealConfig(
    dataset_config=all_hq_dataset_config_5pct, num_anneal_training_tokens=ANNEAL_TOKENS, tpu_type=TPU_TYPE
)

all_hq_model_5pct = default_anneal(
    name=f"{BASE_MODEL_NAME}-eu-all-dolma-hq-w-flan5",
    anneal_config=all_hq_anneal_config_5pct,
)
experiment_steps.append(all_hq_model_5pct)

if __name__ == "__main__":
    executor_main(
        steps=experiment_steps,
    )
