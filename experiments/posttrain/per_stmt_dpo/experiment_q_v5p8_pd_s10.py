# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment Q: fixed-r64/alpha64 LoRA sweep on v5p-8.

This script exists to answer the narrower question left after Exp N, O, and P:

- matched-family LoRA on v5p-16 / v6e-16 with pd=2 matched closely (Exp N)
- full FT on v6e-16 stayed close when pd changed from 4 -> 2 (Exp O)
- LoRA on v5p-16 stayed close when pd changed from 2 -> 4 (Exp P)

So the remaining live question is whether something specifically pathological
about the v5p-8 regime remains once the LoRA recipe is frozen to the recent
good setup.

This script holds all training knobs fixed to the recent r=64/alpha=64 LoRA
debug runs and only changes `per_device_parallelism` via `EXPERIMENT_Q_PD`.

Intended sweep order:
1. `EXPERIMENT_Q_PD=8`
2. `EXPERIMENT_Q_PD=4`
3. optional follow-up: `EXPERIMENT_Q_PD=2`
"""

import os

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
DEFAULT_REGIONS = ["us-central1", "us-east5"]
DEFAULT_PD = 8
ALLOWED_PD = {2, 4, 8}

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_pd_raw = os.environ.get("EXPERIMENT_Q_PD", str(DEFAULT_PD))
PER_DEVICE = int(_pd_raw)
if PER_DEVICE not in ALLOWED_PD:
    raise ValueError(f"Experiment Q only supports per_device_parallelism in {sorted(ALLOWED_PD)}, got {PER_DEVICE}")

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"qpd{PER_DEVICE}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g" if PER_DEVICE <= 4 else "400g", regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
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
    seed=0,
    max_eval_batches=1,
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_q_r64_v5p8_pd{PER_DEVICE}_s10_{_DBG_RUN_TAG}",
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
        "experiment-q",
        "v5p-8",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
