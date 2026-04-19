# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BD: AC recipe + one-shot Gaussian noise on lora_B grad at step 1.

Purpose
=======

Tests whether a tiny perturbation of the first effective gradient update breaks
the c=f32 stall without changing the init or any model structure.

Canonical LoRA + c=f32 stalls near log(2) on v5p-16 pd=2 (AC). At step 0,
B=0 and dL/dB is the only nonzero grad. The first optimizer update is
B_1 = 0 - lr * dL/dB (step 0 → step 1). If this update is too symmetric or
aligned with an unstable direction, training might stall in the f32 regime.

BD keeps the init canonical but at step 1 adds small Gaussian noise only
to the lora_B grad before the optimizer consumes it. This is orthogonal to
both BA (init perturbation) and BC (factor geometry swap): we test
whether breaking the symmetry at the *gradient* level (not init level) is
enough to rescue c=f32.

Sweep
=====

Configure via env vars:
  MARIN_DEBUG_LORA_GRAD_NOISE_STD  — stddev (default 1e-5 per BD spec)
  MARIN_DEBUG_LORA_GRAD_NOISE_STEP — inject step (default 1)
  MARIN_DEBUG_LORA_GRAD_NOISE_TARGET — A, B, both (default B)

Sweep values: 1e-6, 1e-5, 1e-4.

Decision rule
=============

- Step-0 loss = log(2), step-2 loss ≤ 0.5 → noise rescues. Combined with BA,
  this localizes the trap to "needs a kick at step 1."
- No rescue at any scale → gradient-noise is not the rescue mechanism.
  Move to BH/BI/BK (kernel / accum / optimizer probes).

Prereqs
=======

- `MARIN_DEBUG_SKIP_HF_EXPORT=1` (standing rule).
- No env-gate needed (canonical invariants preserved; grad perturbation applied
  only at step N > 0).

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

_NOISE_STD = os.environ.get("MARIN_DEBUG_LORA_GRAD_NOISE_STD", "1e-5")
_NOISE_STEP = os.environ.get("MARIN_DEBUG_LORA_GRAD_NOISE_STEP", "1")
_NOISE_TARGET = os.environ.get("MARIN_DEBUG_LORA_GRAD_NOISE_TARGET", "B")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"bd-v5p16-pd2-fp32-n{_NOISE_STD}-{_NOISE_TARGET.lower()}")

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
        "MARIN_DEBUG_LORA_GRAD_NOISE_STD": _NOISE_STD,
        "MARIN_DEBUG_LORA_GRAD_NOISE_STEP": _NOISE_STEP,
        "MARIN_DEBUG_LORA_GRAD_NOISE_TARGET": _NOISE_TARGET,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_bd_r64_v5p16_pd{PER_DEVICE}_fp32_noise_s10_{_DBG_RUN_TAG}",
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
        "experiment-bd",
        "class-b",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        f"noise-{_NOISE_STD}",
        f"noise-target-{_NOISE_TARGET}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
