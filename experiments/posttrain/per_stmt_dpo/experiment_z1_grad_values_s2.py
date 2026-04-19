# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment Z1: per-element grad slice dump on v5p-8 vs v6e-8.

Clone of experiment_y_sharding_probe_s2.py but with
MARIN_DEBUG_DUMP_GRAD_VALUES=1 enabled, which emits DEBUGJ GRAD_VAL
lines for each Q/K/V/O/gate/up/down lora_B gradient at 5 fixed indices.

Comparing the DEBUGJ GRAD_VAL output between v5p-8 and v6e-8 at step 0
directly shows whether the all-reduced gradient values differ element-
for-element (post-all-reduce divergence = collective behavior is the
source) or match bit-for-bit (ruling out collective as the source).

TPU type configurable via EXPERIMENT_Y_TPU=v5p-8 or v6e-8 (same
convention as Exp Y).

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

_ALLOWED_TPUS = {
    "v5p-8": ["us-east5", "us-central1"],
    "v6e-8": ["us-east5", "us-east1", "europe-west4"],
}

TPU_TYPE = os.environ.get("EXPERIMENT_Y_TPU", "v5p-8")
if TPU_TYPE not in _ALLOWED_TPUS:
    raise ValueError(f"EXPERIMENT_Y_TPU must be one of {sorted(_ALLOWED_TPUS)}, got {TPU_TYPE}")

DEFAULT_REGIONS = _ALLOWED_TPUS[TPU_TYPE]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"z1-{TPU_TYPE.replace('-', '')}")

_RAM = "250g" if TPU_TYPE == "v5p-8" else "400g"

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram=_RAM, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=64,
    num_train_steps=2,
    steps_per_eval=2,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
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
    seed=0,
    max_eval_batches=1,
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
        "MARIN_DEBUG_DUMP_SHARDING": "1",
        "MARIN_DEBUG_DUMP_GRAD_VALUES": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_z1_r64_{TPU_TYPE.replace('-', '')}_pd{PER_DEVICE}_gradvals_s2_{_DBG_RUN_TAG}",
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
        "experiment-z1",
        TPU_TYPE,
        "pd4",
        "r64-alpha64",
        "grad-values",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
