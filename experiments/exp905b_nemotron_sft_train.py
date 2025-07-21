"""
#1237: Starling SFT

SFT the Deeper Starling Iteration of Tootsie 8B Model using the Reasoning + Tulu SFT Mixture.
This is to produce our release candidate for Marin's launch given the strength of the base model!

GitHub Issue: https://github.com/marin-community/marin/issues/1237
"""

import dataclasses

from experiments.defaults import default_sft
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_8b
from experiments.tootsie.exp600_tootsie import tootsie_8b_deeper_starling
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

import logging
logger = logging.getLogger("ray")

# Experiment specific settings
EXPERIMENT_NAME = "sft/mixture_sft_deeper_starling_with_nemotron_and_openthoughts3"

SFT_CONFIG = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
    initialize_from_checkpoint_path=tootsie_8b_deeper_starling.cd("checkpoints/step-1419967").nonblocking(),
)

MODEL_CONFIG = llama_8b

EXPERIMENT_TAGS = [
    "llama",
    "8b",
    "tootsie",
    "sft",
    "starling",
    "mixture",
    "exp905b",
    "nemotron+openthoughts3-1.2m",
]

# Training parameters
BATCH_SIZE = 64
EPOCHS = 3


# Dataset configurations
from experiments.exp905a_nemotron_sft_dstc import DATASETS, create_tokenization_step
tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

# Mixture weights should be read from the json file written by exp905a
mixture_weights = {
    "acecode_89k": 26032149,
    "smoltalk": 883494479,
    "verifiable_math_problems": 382056624,
    "dolphin_r1_nonreasoning": 319820708,
    "dolphin_r1_reasoning": 508743187,
    "bespoke_stratos_17k": 85724829,
    "openthoughts_114k_math": 72964948,
    "tulu_3_sft_mixture": 749008790,
    "natural_reasoning": 966484170,
    "nemotron_sft": 34739443205,
    "openthoughts3": 17449811417
}

# Calculate the number of training steps from computed values
total_tokens = sum(mixture_weights.values())
num_steps = total_tokens // (BATCH_SIZE * MODEL_CONFIG.seq_len) * EPOCHS + 1419967



if __name__ == "__main__":
    sft_mixture_llama3 = lm_mixture_data_config(
        tokenized_datasets,
        mixture_weights, # Edit in create_experiment_config_step, not here.
        shuffle=True,
        missing_weights_are_validation=True,
    )

    _sft_config = dataclasses.replace(
        SFT_CONFIG,
        num_train_steps=num_steps,  # Using the values in the config file
        train_batch_size=BATCH_SIZE,# Using the values in the config file
    )

    sft_step = default_sft(
        EXPERIMENT_NAME,
        tokenized=sft_mixture_llama3,
        model_config=MODEL_CONFIG,
        sft_config=_sft_config,
        tags=EXPERIMENT_TAGS,
    )

    # Now run the SFT step
    executor_main(
        [
            sft_step,
            *default_sft_eval(sft_step),
        ],
        description="Run SFT training step",
    )
