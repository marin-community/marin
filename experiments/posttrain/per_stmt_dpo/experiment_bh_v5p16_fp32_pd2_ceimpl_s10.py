# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BH: AC recipe + override CE kernel implementation (reference path).

Purpose
=======

Tests whether the f32 LoRA DPO failure lives in the blocked XLA CE kernel path.
The default on TPU is `implementation="xla"` (the blocked XLA kernel in
`lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py`,
which tiles over [B, V] for memory efficiency). BH forces
`implementation="reference"` (the pure-JAX reference path, no blocking, no
Pallas).

Controlled by env var `MARIN_DEBUG_CE_IMPL=reference` (wired via B0.2+ hook
in `lib/levanter/src/levanter/models/loss.py`).

Decision rule
=============

- Step-2 loss ≤ 0.5 → the blocked XLA CE kernel has an f32-specific numerical
  flaw. Audit `linear_softmax_cross_entropy_loss_xla`.
- Step-2 loss stays near log(2) → CE kernel not load-bearing; the bug is
  elsewhere (forward embedding path, backward, or optimizer).

Note
====

Reference CE does a full logits matmul, which may OOM on 8B at 4096 seq
len with 64 batch. May need to reduce batch or rank. If it OOMs, we learn
nothing about Bug 2; consider pallas_tpu as an alternative.

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

_CE_IMPL = os.environ.get("MARIN_DEBUG_CE_IMPL", "reference")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"bh-v5p16-pd2-fp32-ce{_CE_IMPL}")

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
        "MARIN_DEBUG_CE_IMPL": _CE_IMPL,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_bh_r64_v5p16_pd{PER_DEVICE}_fp32_ce{_CE_IMPL}_s10_{_DBG_RUN_TAG}",
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
        "experiment-bh",
        "class-b",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        f"ce-impl-{_CE_IMPL}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
