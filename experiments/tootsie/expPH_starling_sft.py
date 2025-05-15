"""
#945 : Spoonbill SFT Learning Rate Sweep

This experiment sweeps through different learning rates for SFT of the spoonbill zloss model.
We try learning rates of [1e-5, 2e-5, 3e-5, 5e-5, 1e-4] to find the optimal rate.
"""

import dataclasses

from experiments.defaults import default_sft
from experiments.exp808_sft_mixture import mixture_config as sft_mixture_llama3
from experiments.exp606_sft import tulu3_llama_tokenize_step
from experiments.llama import llama_8b
from experiments.tootsie.exp977_phoenix_cooldown import tootsie_8b_sensible_starling
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main

sft_experiments = []
deeper_sft_config = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    num_train_steps=10228,
    reinit_tokens=True,
    model_name_or_path="gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999",
)


deeper_mixture_experiment = default_sft(
    name=f"sft/deeper_mixture_sft_starling_1e-4-longer-reinit",
    tokenized=sft_mixture_llama3,
    model_config=llama_8b,
    sft_config=deeper_sft_config,
    tags=["llama", "8b", "tootsie", "sft", "starling", "mixture", "reinit"],
).with_output_path(f"checkpoints/sft/deeper_mixture_sft_starling_1e-4-longer-reinit")
sft_experiments.append(deeper_mixture_experiment)


if __name__ == "__main__":
    executor_main(
        sft_experiments,
        description="SFT mixture sweep for starling model",
    )
