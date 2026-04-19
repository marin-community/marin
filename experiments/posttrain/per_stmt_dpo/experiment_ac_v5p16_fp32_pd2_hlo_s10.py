# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment AC: AB recipe (`p=f32, c=f32`) on v5p-16 pd=2.

Purpose
=======

Identify the non-dtype mechanism causing the v5p-8 width-4 LoRA DPO
pathology.

AB (v5p-8, `c=f32`, HLO dump) proved that forcing f32 reductions does
NOT recover training. So bf16 non-associativity in the grad all-reduce
is not the mechanism. The `|data|=4` pathology must therefore be
driven by something structural about width 4 — tree/algorithm choice,
buffer layout, fusion boundaries, or reduction operand order — not
reduction dtype.

AC is the minimally-confounded comparison to isolate that structural
variable:

- Same TPU family as AB v5p-8 (v5p).
- Same microbatch structure as AB v5p-8:
  - v5p-8 pd=4: `microbatch_size = 4 * 4 = 16`, batch=64 → 4 accum steps.
  - v5p-16 pd=2: `microbatch_size = 2 * 8 = 16`, batch=64 → 4 accum steps.
- Same dtype (`p=f32, c=f32`).
- Same LoRA config (r=64, alpha=64, zero_init_b).
- Same data, same seed.
- **Only `|data|` differs: 8 (AC) vs 4 (AB v5p-8).**
- HLO dump so we can diff structurally against AB v5p-8 HLO.

Predictions
===========

- Training recovers (step-2 loss ≤ 0.5, step-10 delta_pi - delta_ref ≥ 5).
  If this fails, `c=f32` has a global issue we missed and AB's "stuck
  under f32" reading is wrong.
- HLO differs from AB v5p-8's HLO in at least `replica_groups` size
  and per-chip shard shapes. Any additional structural difference
  (fusion boundaries, reduce-scatter algorithm, scheduling) is a
  mechanism candidate.
- Step-0 LoRA grad values differ between AC (this run) and AB v5p-8:
  - At ~1e-7 absolute → fp32 non-associativity at width 4 is the
    load-bearing effect.
  - At ~1e-5 absolute (same magnitude Z1 saw at bf16) → a systematic
    non-precision bias exists at width 4.

Decision matrix after AC
========================

Four cases (AC trajectory / HLO diff vs AB v5p-8 / grad diff vs AB v5p-8):

1. recovers / identical-except-replica_groups / ~1e-7 absolute
   → width-4 reduction order + f32 associativity limit is the
   mechanism. Next: try an XLA flag to force an alternate reduction
   tree.

2. recovers / structural differences found / any grad diff
   → those structural differences ARE the mechanism candidates.
   Next: isolate which one matters.

3. recovers / identical-except-replica_groups / ~1e-5 or larger
   → systematic non-precision bias at width 4. Hunt in compile path.

4. stuck (step-2 loss ≈ 0.685)
   → f32 globally breaks training somehow. Surprise. Audit AB + U.

Launch directive (from top of logbook)
======================================

Launch one copy per available v5p region in parallel with distinct
MARIN_DEBUG_RUN_TAG values. For v5p-*: us-central1, us-east5.

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

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "ac-v5p16-pd2-fp32")
_DBG_UPLOAD_PREFIX = os.environ.get(
    "EXPERIMENT_AC_HLO_PREFIX",
    "gs://marin-us-central1/debug/xla_hlo",
)
_DBG_UPLOAD_DIR = f"{_DBG_UPLOAD_PREFIX}/{_DBG_RUN_TAG}/"

_XLA_FLAGS = "--xla_dump_to=/tmp/xla_hlo " "--xla_dump_hlo_as_text " "--xla_dump_hlo_module_re=.*train.*"

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
        "XLA_FLAGS": _XLA_FLAGS,
        "MARIN_DEBUG_HLO_UPLOAD_DIR": _DBG_UPLOAD_DIR,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_ac_r64_v5p16_pd{PER_DEVICE}_fp32_hlo_s10_{_DBG_RUN_TAG}",
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
        "experiment-ac",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "hlo-dump",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
