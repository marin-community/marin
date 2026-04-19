# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment CP9: Bug 1 (v5p-8 pd=4 c=bf16) at lr=3e-5 to complete LR grid.

Purpose
=======

Completes the 3x3 LR-scaling comparison:

| config | lr=1e-6 | lr=1e-5 | lr=3e-5 |
| Exp N  | 0.32    | 0.26    | 0.125 (CP7)   |
| Bug 1  | 0.66    | 0.41 (CP6) | CP9 ← this |
| Bug 2  | 0.66    | 0.41    | CP8 (running) |

If CP9 step 9 ~= CP8 step 9 and CP7 step 9 = 0.125 → Bug 1 and Bug 2 scale
identically with LR, confirming Bug 1 == Bug 2 (pure LR slowdown).
If CP9 > CP8 → Bug 1 has a distinct ceiling; the v5p-8 topology imposes a
floor that LR alone cannot overcome.

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

_LR = float(os.environ.get("MARIN_DEBUG_LR", "3e-5"))
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"cp9-v5p8-pd4-lr{_LR:.0e}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=_LR,
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
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_cp9_r64_v5p8_pd{PER_DEVICE}_lr{_LR:.0e}_s10_{_DBG_RUN_TAG}",
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
        "experiment-cp9",
        "class-c",
        "lane-2-bug-1",
        "v5p-8",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        f"lr-{_LR:.0e}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
