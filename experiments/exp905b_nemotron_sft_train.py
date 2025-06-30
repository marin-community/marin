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


# Dataset configurations
from exp905a_nemotron_sft_dstc import DATASETS, create_tokenization_step
# Dataset weights set with the naive baseline of the number of documents per dataset
mixture_weights = {
    "acecode_89k": 87149,
    "smoltalk": 1043917,
    "verifiable_math_problems": 777457,
    "dolphin_r1_nonreasoning": 214318,
    "dolphin_r1_reasoning": 585418,
    "bespoke_stratos_17k": 16710,
    "openthoughts_114k_math": 89120,
    "tulu_3_sft_mixture": 939343,
    "natural_reasoning": 1145824,
    "nemotron_sft": 32955418,
    "openthoughts3": 1200000
}

tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

sft_experiments = []
deeper_sft_config = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    num_train_steps=10228,
    train_batch_size=128,
    model_name_or_path=tootsie_8b_deeper_starling,
)

sft_mixture_llama3 = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=True,
    missing_weights_are_validation=True,
)

mixture_sft_deeper_starling2 = default_sft(
    name="sft/mixture_sft_deeper_starling2",
    tokenized=sft_mixture_llama3,
    model_config=llama_8b,
    sft_config=deeper_sft_config,
    tags=[
        "llama",
        "8b",
        "tootsie",
        "sft",
        "starling",
        "mixture",
        "exp905b",
        "nemotron+openthoughts3-1.2m",
    ],
).with_output_path("checkpoints/sft/mixture_sft_deeper_starling2")


if __name__ == "__main__":
    executor_main(
        [
            mixture_sft_deeper_starling2,
            *default_sft_eval(mixture_sft_deeper_starling2),
        ],
        description="SFT for Deeper Starling Model with addition of Nemotron and OpenThoughts3-1.2M",
    )
