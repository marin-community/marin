# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity check: run the canonical lr=1e-5 LoRA DPO config on v6e-8.

Matches the v5p-8 tune_lora run (smh_lr1em05_seed0_b64_v5p8) exactly except
for hardware. Uses per_device_parallelism=4 with 2x grad accum (validated
v6e-8 config from dpo-lora-claude.md).

Compare against existing v5p-8 run to check for the same v6e-8 vs v5p-8
training discrepancy observed in per-statement DPO experiments.
"""

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo
from experiments.dpo_bloom_speceval_v2 import tokenized_eval, tokenized_preferences, tokenized_train
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v6e-8", regions=["europe-west4", "us-east5", "us-east1"]),
    per_device_parallelism=4,
    per_device_eval_parallelism=4,
    train_batch_size=64,
    num_epochs=1.0,
    learning_rate=1e-5,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(
        r=64,
        alpha=64,
        dropout=0.0,
        zero_init_b=True,
        target_modules=None,
    ),
    reference=AdapterBaseReferenceConfig(),
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
    steps_per_checkpoint=200,
    steps_per_hf_export=200,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
)

training_step = default_dpo(
    name="dpo/tune_lora/lora_bloom_speceval_v2_marin_instruct_lora_beta0p1_lr1e5_seed0_b64_v6e8",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    tags=[
        "dpo",
        "lora-dpo",
        "bloom",
        "speceval-v2",
        "llama3",
        "marin-instruct",
        "executor",
        "tune-lora",
        "reference-logprob-cache",
        "beta0.1",
        "lr1e-05",
        "seed0",
        "b64",
        "r64",
        "v6e-8",
        "sanity-check",
    ],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step,
        ]
    )
