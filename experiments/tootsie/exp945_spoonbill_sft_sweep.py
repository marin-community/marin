"""
#945 : Spoonbill SFT Learning Rate Sweep

This experiment sweeps through different learning rates for SFT of the spoonbill zloss model.
We try learning rates of [1e-5, 2e-5, 3e-5, 5e-5, 1e-4] to find the optimal rate.
"""

import dataclasses

from experiments.defaults import default_sft
from experiments.exp606_sft import tulu3_llama_tokenize_step
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import llama_8b_fp32_attn, spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main

# Define the learning rates to sweep through
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4]

# Create SFT configs for each learning rate
sft_configs = []
for lr in LEARNING_RATES:
    sft_config = dataclasses.replace(
        spoonbill_zloss_tulu3_sft_config,
        learning_rate=lr,
    )
    sft_configs.append(sft_config)

# Create SFT experiments for each learning rate
sft_experiments = []
for lr, sft_config in zip(LEARNING_RATES, sft_configs, strict=True):
    experiment = default_sft(
        name=f"sft/tulu3_sft_spoonbill_945_lr_{lr:.0e}",
        tokenized=tulu3_llama_tokenize_step,
        model_config=llama_8b_fp32_attn,
        sft_config=sft_config,
        tags=["llama", "8b", "exp945", "tootsie", "sft", "spoonbill"],
    ).with_output_path(f"checkpoints/sft/tulu3_sft_spoonbill_945_lr_{lr:.0e}")
    sft_experiments.append(experiment)

if __name__ == "__main__":
    executor_main(
        sft_experiments,
        description="SFT learning rate sweep for spoonbill zloss model",
    )
