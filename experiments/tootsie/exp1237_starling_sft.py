# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
#1237: Starling SFT

SFT the Deeper Starling Iteration of Tootsie 8B Model using the Reasoning + Tulu SFT Mixture.
This is to produce our release candidate for Marin's launch given the strength of the base model!

GitHub Issue: https://github.com/marin-community/marin/issues/1237
"""

import dataclasses

from experiments.defaults import default_sft
from experiments.evals.evals import default_sft_eval
from experiments.exp808_sft_mixture import mixture_config as sft_mixture_llama3
from experiments.llama import llama_8b
from experiments.tootsie.exp600_tootsie import tootsie_8b_deeper_starling
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main

sft_experiments = []
deeper_sft_config = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    num_train_steps=10228,
    train_batch_size=128,
    initialize_from_hf=tootsie_8b_deeper_starling,
)


mixture_sft_deeper_starling = default_sft(
    name="sft/mixture_sft_deeper_starling",
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
    ],
).with_output_path("checkpoints/sft/mixture_sft_deeper_starling")


if __name__ == "__main__":
    executor_main(
        [
            mixture_sft_deeper_starling,
            *default_sft_eval(mixture_sft_deeper_starling),
        ],
        description="SFT for Deeper Starling Model",
    )
