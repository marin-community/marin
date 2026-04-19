# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment AB: rerun of Experiment U (`p=f32, c=f32` on v5p-8 pd=4 LoRA),
with HLO dumping enabled so we can directly verify the dtype of the FSDP
gradient reductions.

Why AB exists
=============

AA v1-v5 tried to force f32 FSDP grad reductions by casting gradients inside
`grad_accum.py`'s accumulation loop. AA v5's diagnostic prints revealed that
the gradients reaching that loop are already `float32` — which means the
FSDP bf16 reduce-scatter happens *per-layer inside the backward pass of
`fn`*, not at the microbatch accumulation boundary. The entire AA
intervention site was wrong. See `debug_accum_tpu_type.md` AA v5 section.

Exp U was the experiment that *should* have produced f32 reductions, by
forcing all forward/backward tensors to f32 via the mixed-precision policy.
Exp U's trajectory was stuck in the bad basin, same as Exp Q. If Exp U's
HLO really did have f32 reductions, that falsifies the bf16-collective-
width-4 hypothesis and forces us to pivot to non-dtype mechanisms. If Exp
U's HLO still had bf16 reductions somewhere (despite `c=f32`), we've got a
hidden bf16 path to find and fix.

Exp U was run before the HLO-dump workflow was added (Z4 added it). AB is
the same recipe, this time with HLO dumping, so we can answer the question
directly.

Decision matrix (see logbook for full context)
==============================================

| HLO LoRA-grad reduce dtype | Training outcome | Conclusion |
|---|---|---|
| f32 | stuck (step-2 ≈ 0.685) | bf16-collective-width-4 HYPOTHESIS REJECTED; pivot to non-dtype |data|=4 investigation |
| f32 | recovers (step-2 < 0.5) | bf16-collective-width-4 confirmed; Exp U must have been mislogged |
| bf16 | stuck | hidden bf16 path forces bf16 reductions despite c=f32; find and fix it |
| bf16 | recovers | unexpected; investigate |

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

TPU_TYPE = "v5p-8"
REGIONS = ["us-central1", "us-east5"]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "ab-pd4-fp32")
_DBG_UPLOAD_PREFIX = os.environ.get(
    "EXPERIMENT_AB_HLO_PREFIX",
    "gs://marin-us-central1/debug/xla_hlo",
)
_DBG_UPLOAD_DIR = f"{_DBG_UPLOAD_PREFIX}/{_DBG_RUN_TAG}/"

_XLA_FLAGS = "--xla_dump_to=/tmp/xla_hlo " "--xla_dump_hlo_as_text " "--xla_dump_hlo_module_re=.*train.*"

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
    name=f"dpo/stmt_dpo/debug/experiment_ab_r64_v5p8_pd{PER_DEVICE}_fp32_hlo_s10_{_DBG_RUN_TAG}",
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
        "experiment-ab",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        "p-f32-c-f32",
        "hlo-dump",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
