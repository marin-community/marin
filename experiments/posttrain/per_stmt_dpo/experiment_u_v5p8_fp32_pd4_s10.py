# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment U: rerun the bad `v5p-8` LoRA regime with full fp32 compute.

This is intentionally a near-copy of Experiment Q's `pd=4` branch:

- `v5p-8`
- LoRA `r=64`, `alpha=64`
- `AdapterBaseReferenceConfig`
- `bs=64`
- `steps=10`

The one scientific change is the mixed-precision policy:

- `p=f32,c=f32`

We also pin `per_device_parallelism=4` to keep peak HBM lower than the
original `pd=8` Experiment Q configuration while still staying in the known-bad
`v5p-8` regime.
"""

import os

import jmp
from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.per_stmt_dpo.common import (
    tokenized_full_val,
    tokenized_preferences,
    tokenized_stmt_val,
    tokenized_train,
)
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

TPU_TYPE = "v5p-8"
REGIONS = ["us-central1", "us-east5"]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "u-pd4-fp32")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None),
    reference=AdapterBaseReferenceConfig(),
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="disabled"),
    steps_per_checkpoint=9999,
    steps_per_hf_export=9999,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    mp=jmp.get_policy("p=f32,c=f32"),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    lr_schedule="cosine",
    warmup=0.1,
    seed=0,
    max_eval_batches=1,
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_u_r64_v5p8_pd{PER_DEVICE}_fp32_s10_{_DBG_RUN_TAG}",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    tags=[
        "dpo",
        "lora-dpo",
        "bloom",
        "per-stmt",
        "support-mental-health",
        "debug-accum",
        "experiment-u",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        "p-f32-c-f32",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
