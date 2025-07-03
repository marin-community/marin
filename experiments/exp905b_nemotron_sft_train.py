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
# Different from exp808, this dict records the number of tokens, not rows.
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
BATCH_SIZE = 128
EPOCHS = 3
SEQ_LEN = 4096 # Should change to load from model config

tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

TOTAL_TOKENS = sum(mixture_weights.values())
NUM_STEPS = TOTAL_TOKENS // (BATCH_SIZE * SEQ_LEN) * EPOCHS

sft_experiments = []
deeper_sft_config = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    num_train_steps=10228*8, # Num rows is now 8x larger, so we need to train for 8x more steps. We need to be more principled
    train_batch_size=BATCH_SIZE,
    initialize_from_checkpoint_path=tootsie_8b_deeper_starling.cd("checkpoints/step-1399999").nonblocking(),
)

sft_mixture_llama3 = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=True,
    missing_weights_are_validation=True,
)

mixture_sft_deeper_starling_with_nemotron_and_openthoughts3 = default_sft(
    name="sft/mixture_sft_deeper_starling_with_nemotron_and_openthoughts3",
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
).with_output_path("checkpoints/sft/mixture_sft_deeper_starling_with_nemotron_and_openthoughts3")


if __name__ == "__main__":
    executor_main(
        [
            mixture_sft_deeper_starling_with_nemotron_and_openthoughts3,
            *default_sft_eval(mixture_sft_deeper_starling_with_nemotron_and_openthoughts3),
        ],
        description="SFT for Deeper Starling Model with addition of Nemotron and OpenThoughts3-1.2M",
    )
