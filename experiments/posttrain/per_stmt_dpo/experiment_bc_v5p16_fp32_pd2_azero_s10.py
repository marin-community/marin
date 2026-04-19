# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BC: AC recipe + a_init_mode='zero', b_init_scale=1e-3.

Purpose
=======

Factor-geometry probe. In canonical LoRA init (A=random, B=0), at step 0:
  - adapter_out = B @ A @ x = 0 (policy = reference). Loss = log(2). ✓ invariant held.
  - dL/dA = 0 (no grad flows into A because adapter_out=0 and lm_head·adapter_out=0
    but chain rule says dL/dA = (dL/dadapter_out) · B, and B=0 kills it).
  - dL/dB != 0.

So A stays at random init, only B moves. First effective update direction is
B_new = 0 - lr * dL/dB, then at step 2 dL/dA flips on.

BC reverses this: A=0 (new knob) + B=random_small (b_init_scale=1e-3). Now:
  - adapter_out = B @ 0 @ x = 0 (policy = reference). Loss = log(2). ✓ invariant STILL held.
  - dL/dB = 0 (by symmetric argument: A=0 kills the chain).
  - dL/dA != 0.

First effective update direction is A_new = 0 - lr * dL/dA, then at step 2 dL/dB
flips on. **The roles of A and B are swapped at step 0/1.**

If Bug 2 (c=f32 stall) is caused specifically by the A=random,B=0 factor
geometry (e.g. "update lands in the rank-direction spanned by A_init"), BC
should behave differently — either rescuing training (= geometry is the cause)
or stalling identically (= geometry is not the cause, something else is).

Decision rule
=============

- BC step-0 loss ≈ log(2) AND step-2 loss ≤ 0.5 → factor-geometry-swap rescues
  c=f32. Promote "first-updated-factor direction" as the leading Bug-2 mechanism.
  Next: BJ scalar comparison of step-0 grads for N / AC / BC.
- BC step-0 loss ≈ log(2) AND step-2 loss ≈ 0.69 (like AC) → factor geometry
  not load-bearing. Move to BD/BH/BK.
- BC step-0 loss far from log(2) → my invariant reasoning above is wrong, or
  a_init_mode='zero' has a bug. Debug.

Key distinction from BA
=======================

BA breaks `policy=reference` at init (nonzero B); BC preserves `policy=reference`
(zero A). Thus BC does NOT require the `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B`
env gate — the updated validation at train_dpo.py:369-394 checks `a_is_zero OR b_is_zero`.
BA and BC together disentangle "breaks invariant" from "different factor direction."

Config
======

Base: Exp N (debug_r64_matched_pd2_s10.py, v5p-16).
Deltas from AC (Exp N + c=f32):
- a_init_mode='zero' (new B0.1 knob).
- b_init_scale=1e-3 (matches BA scale).

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

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "bc-v5p16-pd2-fp32-azero")
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
        zero_init_b=False,
        target_modules=None,
        a_init_mode="zero",
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
    name=f"dpo/stmt_dpo/debug/experiment_bc_r64_v5p16_pd{PER_DEVICE}_fp32_azero_bs{_BSCALE:.0e}_s10_{_DBG_RUN_TAG}",
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
        "experiment-bc",
        "class-b",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "a-init-zero",
        f"b-init-scale-{_BSCALE:.0e}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
