# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment C1a: AC recipe + JAX_DEFAULT_MATMUL_PRECISION=default.

Purpose
=======

Based on BK (cast grads to bf16) NOT rescuing, and BH (CE kernel ref) NOT
rescuing, the Bug 2 mechanism is isolated to forward/backward compute
producing different gradients at c=f32 vs c=bf16. The most obvious
candidate compute difference is MATMUL PRECISION.

At c=bf16, every matmul in forward/backward is bf16xbf16→bf16 (with
f32 accumulation internally). Rounding every matmul output to bf16.
At c=f32, matmuls are f32xf32→f32 (no rounding to bf16).

C1a tests: **under c=f32, force matmul precision to DEFAULT (bf16 on TPU)**.
Activations are f32 but matmul intermediaries are bf16.

Via env var `JAX_DEFAULT_MATMUL_PRECISION=default`.

Decision rule
=============

- Step 2 loss ≤ 0.5 → bf16 matmul rounding is the critical compute
  difference that enables Exp N's fast escape. We can keep c=f32 (for
  numerical stability in activations) but use bf16 matmul internally.
- Step 2 loss stays ~0.69 → matmul precision alone isn't the driver.
  Dive into per-op precision differences.

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

_PRECISION = os.environ.get("MARIN_DEBUG_MATMUL_PRECISION", "default")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"c1a-v5p16-pd2-fp32-mm{_PRECISION}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
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
        "JAX_DEFAULT_MATMUL_PRECISION": _PRECISION,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_c1a_r64_v5p16_pd{PER_DEVICE}_fp32_mm{_PRECISION}_s10_{_DBG_RUN_TAG}",
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
        "experiment-c1a",
        "class-c",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        f"matmul-precision-{_PRECISION}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
