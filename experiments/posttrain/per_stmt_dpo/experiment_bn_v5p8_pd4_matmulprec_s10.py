# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BN: Exp Q recipe + matmul precision sweep on v5p-8 width-4.

Purpose
=======

Tests Lane 2 (Bug 1) hypothesis: does the local dot algorithm (matmul
precision) matter at width 4 on v5p-8?

AB (v5p-8 pd=4 c=f32) falsified "bf16 collective dtype is the mechanism" —
the all-reduces are f32 yet training still stalls. Bug 1's mechanism is
still unknown. BN tests whether the per-device matmul algorithm
(bf16+bf16→bf16, bf16+bf16→f32, or f32+f32→f32) contributes.

Controlled via env var `MARIN_DEBUG_MATMUL_PRECISION`, mapped to
`jax_default_matmul_precision`:
  - "default" (v5p default, typically bf16 with 1 pass)
  - "high" (bf16 with 3 passes, closer to f32)
  - "highest" (pure f32 matmul)

This is a Bug 1 probe — keeps c=bf16 (the default that triggers the stall)
and leaves width 4 intact. The single knob is local matmul precision.

Decision rule
=============

- "high" or "highest" rescues → local matmul precision is part of Bug 1.
- All three stall identically → local matmul precision not at fault.

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

import jax
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
REGIONS = ["us-east5", "us-central1"]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_PRECISION = os.environ.get("MARIN_DEBUG_MATMUL_PRECISION", "highest")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"bn-v5p8-pd4-matmul{_PRECISION}")

# Map precision name to jax.lax.Precision
_JAX_PRECISION_MAP = {
    "default": jax.lax.Precision.DEFAULT,
    "high": jax.lax.Precision.HIGH,
    "highest": jax.lax.Precision.HIGHEST,
}
if _PRECISION not in _JAX_PRECISION_MAP:
    raise ValueError(f"MARIN_DEBUG_MATMUL_PRECISION must be one of {list(_JAX_PRECISION_MAP)}, got {_PRECISION}")

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
    name=f"dpo/stmt_dpo/debug/experiment_bn_r64_v5p8_pd{PER_DEVICE}_matmul{_PRECISION}_s10_{_DBG_RUN_TAG}",
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
        "experiment-bn",
        "class-b",
        "lane-2",
        "v5p-8",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        f"matmul-precision-{_PRECISION}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
