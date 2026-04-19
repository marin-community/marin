# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment CF: CA at 100 steps to study trajectory shape.

Purpose
=======

CA (10 steps) reached 0.532 at step 9. Was this reflecting a plateau
(CA plateaus at 0.53) or a slow descent (CA continues to descend past
step 9 toward Exp N's 0.32)?

CF runs CA for 100 steps. Predictions:
- CF step 99 ≈ 0.32 → CA is fast-escape + slow-descent. The plateau
  at step 9 was premature; given more time, CA reaches Exp N level.
- CF step 99 ≈ 0.50 → CA is fast-escape with different end-state
  plateau. Residual mechanism is truly different, not just slower.

Env: all CA gates ON (Linear I/O/W, RmsNorm out, residual).

Note
----

New runs in this campaign should use the same-region GCS copy of
``Llama-3.1-8B-Instruct`` as the base model (see
``experiments.posttrain.per_stmt_dpo.base_model.LLAMA_3_1_8B_INSTRUCT_GCS_PATH``).
Loading is ~3-5x faster than the HuggingFace Hub CDN and Bug-1 has been
verified model-agnostic. Do NOT retroactively edit the ``model_name_or_path``
here - its historical value is tied to the run artifacts. This note is
for future forks of this experiment.
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

TPU_TYPE = "v5p-16"
REGIONS = ["us-central1", "us-east5"]
PER_DEVICE = 2

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_N_STEPS = int(os.environ.get("MARIN_DEBUG_NUM_STEPS", "100"))
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"cf-v5p16-fp32-ca-s{_N_STEPS}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=_N_STEPS,
    steps_per_eval=9999,
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
        "MARIN_DEBUG_ROUND_LINEAR_OUT": "1",
        "MARIN_DEBUG_ROUND_LINEAR_INPUT": "1",
        "MARIN_DEBUG_ROUND_LINEAR_WEIGHT": "1",
        "MARIN_DEBUG_ROUND_NORM_OUT": "1",
        "MARIN_DEBUG_ROUND_RESIDUAL": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_cf_r64_v5p16_pd{PER_DEVICE}_fp32_ca100_s{_N_STEPS}_{_DBG_RUN_TAG}",
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
        "experiment-cf",
        "class-c",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "ca-rounding",
        f"steps-{_N_STEPS}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
