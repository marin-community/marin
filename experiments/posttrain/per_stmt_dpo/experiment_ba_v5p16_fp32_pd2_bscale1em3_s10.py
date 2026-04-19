# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BA: AC recipe + b_init_scale=1e-3.

Purpose
=======

Clean successor to AD v3 for Class-B Lane 1 (Bug 2 c=f32).

AC stalls at step-9 loss 0.66 (zero_init_b=True, c=f32, AdapterBase on v5p-16 pd=2).
AD v3 (zero_init_b=False → Kaiming-scale random B) descends from step-0 loss 7.46
to 5.11 by step 9 but doesn't reach the good basin. The descent under c=f32 is
real, but AD v3 confounds "light symmetry break" with "huge random init shock."

BA uses the new `b_init_scale=1e-3` knob (B0.1) to init B as N(0, 1e-3). This
is:
  - 4-5 orders of magnitude smaller stddev than AD v3's Kaiming init on B,
  - still breaks the exact `B=0` symmetry that might be load-bearing for the
    c=f32 stall,
  - keeps the step-0 state much closer to the policy=reference line, so we
    should see near-log(2) at step 0 and then either descend to ~0.33 like
    Exp N or stall like AC.

Decision rule
=============

- step-0 loss ≈ log(2) and step-2 loss ≤ 0.5 → H_B (light symmetry break
  rescues f32) supported. Run BB to map the rescue window.
- step-0 loss ≈ log(2) but step-2 loss ≈ 0.69 → the symmetry break didn't
  rescue at this scale. Larger-scale probe needed (BB with 1e-2).
- step-0 loss already far from log(2) → probe scale was too large; the run
  behaves like AD v3. Reduce b_init_scale to 1e-4 for a cleaner probe.

Prereq env vars
===============

- MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1 — BA breaks the
  AdapterBase+LoRA+adapter_out=0 validation (nonzero B means nonzero
  adapter_out at init).
- MARIN_DEBUG_SKIP_HF_EXPORT=1 — standing rule.

Config
======

Base: Exp N (debug_r64_matched_pd2_s10.py with TPU_TYPE=v5p-16).
Deltas from AC (which is Exp N + c=f32):
- b_init_scale=1e-3 via new LoraConfig knob (B0.1).

Reference, mesh, seed, data, LR, warmup, TPU family all unchanged.

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

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "ba-v5p16-pd2-fp32-bs1em3")
_BSCALE = float(os.environ.get("MARIN_DEBUG_B_INIT_SCALE", "1e-3"))

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(
        r=64,
        alpha=64,
        dropout=0.0,
        zero_init_b=True,
        target_modules=None,
        b_init_scale=_BSCALE,
    ),
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
    name=f"dpo/stmt_dpo/debug/experiment_ba_r64_v5p16_pd{PER_DEVICE}_fp32_bs{_BSCALE:.0e}_s10_{_DBG_RUN_TAG}",
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
        "experiment-ba",
        "class-b",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        f"b-init-scale-{_BSCALE:.0e}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
