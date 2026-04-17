# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment Z2: force fp32 precision on cross-chip collectives on v5p-8.

Clone of experiment_q_v5p8_pd_s10.py (bs=64, pd=4, LoRA r=64, ABRC,
seq_len=4096, 10 steps) with XLA_FLAGS set to force fp32 internal
precision on TPU collectives.

The flag tried is `--xla_allow_excess_precision=false`, which tells
XLA not to downcast intermediates to lower precision for speed.
Other candidate flags (try via EXPERIMENT_Z2_XLA_FLAGS if the default
doesn't recover):

  --xla_allow_excess_precision=false
  --xla_tpu_force_allreduce_f32=true                 (may not exist; safe to try)
  --xla_tpu_enable_all_reduce_sum_fusion=false       (disables fusion that may bf16)
  --xla_tpu_enable_latency_hiding_scheduler=false    (changes scheduling)

Outcomes:

- loss at step 2 drops to ~0.33 band → collective internal precision
  is the mechanism; found a production fix.
- loss at step 2 stays at ~0.685 → collective precision (per this
  flag) is not the mechanism; try a different flag or move to Z4
  (HLO diff).
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
DEFAULT_REGIONS = ["us-east5", "us-central1"]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "z2")

_DEFAULT_XLA_FLAGS = "--xla_allow_excess_precision=false"
XLA_FLAGS = os.environ.get("EXPERIMENT_Z2_XLA_FLAGS", _DEFAULT_XLA_FLAGS)

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
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
        "XLA_FLAGS": XLA_FLAGS,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_z2_r64_v5p8_pd{PER_DEVICE}_f32coll_s10_{_DBG_RUN_TAG}",
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
        "experiment-z2",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        "f32-collective",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
