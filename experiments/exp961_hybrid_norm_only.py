"""
Experiment 961: Hybrid Norm and Input Embedding Norm Ablation
Link: https://github.com/stanford-crfm/marin/issues/961

In Gemma, a hybrid norm is applied on every layer, with one layernorm at the input and one at the output of each layer.
Another similar operation is applying Layernorm with input embedding.

This may be useful for controlling the norm of the weight.
The norms of the features are now decided by the Layernorm scaling factor and the model becomes more scale invariant.
If larger weight norm is bad for SFT, the model with more normalization may be easier to finetune.
This experiment will ablate with this hypothesis.

We conduct an experiment with the same setting as #950 (large cosine learning rate) but with a modified architecture.

The experiment compares:
- Combined: Llama with both hybrid norm and input embedding norm

Each model is trained on a DCLM mixture followed by supervised fine-tuning with the Tulu SFT configuration.

Author: Kaiyue Wen
"""

import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_sft, default_train
from experiments.exp606_sft import tulu3_llama_tokenize_step, tulu_sft_config
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main, output_path_of

# Base training config with cosine schedule and high learning rate
base_train_config = dataclasses.replace(
    llama_1_4b_train_config,
    num_train_steps=238418,  # 4096 * 1024 * 238418 = 1T tokens
    weight_decay=0.05,
    learning_rate=1e-3,
    lr_schedule="cosine",
    decay=None,
    ema_beta=0.995,
    z_loss_weight=1e-4,
)

# Create model configs with different normalization settings
llama_1_4b_hybrid_norm = dataclasses.replace(llama_1_4b, hybrid_norm=True)

# Train models with hybrid normalization settings
dclm_mix_model_hybrid = default_train(
    name="norm_ablation_hybrid",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b_hybrid_norm,
    train_config=base_train_config,
)

# SFT configurations
sft_model_hybrid = default_sft(
    name="sft/tulu_sft_hybrid",
    tokenized=tulu3_llama_tokenize_step,
    model_config=llama_1_4b_hybrid_norm,
    sft_config=dataclasses.replace(
        tulu_sft_config,
        model_name_or_path=output_path_of(dclm_mix_model_hybrid, "hf/238417/"),
    ),
    tags=["llama", "1.4b", "exp961", "hybrid_norm", "sft", "z_loss"],
).with_output_path("checkpoints/sft/tulu_sft_hybrid")

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_mix_model_hybrid,
            sft_model_hybrid,
        ],
        description=(
            "Train 1.4B models with a hybrid normalization setting on dclm, "
            "then SFT the resulting models to test the impact of normalization on SFT performance."
        ),
    )
