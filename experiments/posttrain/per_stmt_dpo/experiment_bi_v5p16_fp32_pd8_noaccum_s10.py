# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BI: AC recipe + pd=8 (no microbatch accumulation).

Purpose
=======

Tests whether the c=f32 failure lives in the grad-accum reshard loop at
`lib/levanter/src/levanter/grad_accum.py`, or in the intra-step reshards
that are part of multi-microbatch grad accumulation.

AC on v5p-16 pd=2: microbatch=16, accum_steps=4 (bs=64 / (pd=2 * chips=8)).
BI on v5p-16 pd=8: microbatch=64, accum_steps=1 (bs=64 / (pd=8 * chips=8) = 1).
Same batch, same model, same mesh family, same c=f32 — only grad-accum
changes.

If BI rescues, we know Bug 2 requires microbatching. If BI stalls, the bug
is in the non-accum forward/backward/optimizer path.

Note on mesh sharding
=====================

With pd=8 on v5p-16, the mesh is `{data:8, model:1}` (since v5p-16 has 8
unique positions on the data axis — wait, v5p-16 has 16 chips. Checking:
for v5p, each chip is one "device". 16 chips. pd=8 means 8 data replicas
x 2 per-device, so data axis has 8 positions; model axis is implicit =1
or we shard model along the remaining 2. Let's just set pd=8 and see.)

Actually simpler: v5p-16 has 16 chips. pd=8 uses 8 chips for data, 2 for
model. But the intended FSDP recipe is pd=per_device_parallelism=8 means
each device does 8 examples per microbatch, and microbatch = pd x #devices
= 8 x 8 = 64 = train_batch_size. So grad_accum = 1.

Decision rule
=============

- Step-2 loss ≤ 0.5 → grad-accum path broken at c=f32.
- Step-2 loss stays near log(2) → accum not at fault; bug is in the
  single-microbatch path.

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
PER_DEVICE = 8  # BI key knob: pd=8 → microbatch=bs=64, no accum.

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "bi-v5p16-pd8-fp32-noaccum")

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
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_bi_r64_v5p16_pd{PER_DEVICE}_fp32_noaccum_s10_{_DBG_RUN_TAG}",
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
        "experiment-bi",
        "class-b",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "no-grad-accum",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
