# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment Z4: HLO dump + GCS upload for v5p-8 vs v6e-8 diff.

Clone of experiment_y_sharding_probe_s2.py with XLA_FLAGS set to dump
the compiled HLO to /tmp/xla_hlo, plus MARIN_DEBUG_HLO_UPLOAD_DIR set
to a GCS path so the train_dpo atexit hook uploads the dump on exit.

After runs complete on both TPU families, pull the dumped HLO from
GCS and diff the all-reduce op signatures (algorithm, dtype,
replica_groups) to identify what XLA actually chose.

TPU type configurable via EXPERIMENT_Y_TPU (same as Y and Z1).
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

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"z4-{TPU_TYPE.replace('-', '')}")
_DBG_UPLOAD_PREFIX = os.environ.get(
    "EXPERIMENT_Z4_HLO_PREFIX",
    "gs://marin-us-central1/debug/xla_hlo",
)
_DBG_UPLOAD_DIR = f"{_DBG_UPLOAD_PREFIX}/{_DBG_RUN_TAG}/"

_RAM = "250g" if TPU_TYPE == "v5p-8" else "400g"

_XLA_FLAGS = "--xla_dump_to=/tmp/xla_hlo " "--xla_dump_hlo_as_text " "--xla_dump_hlo_module_re=.*train.*"

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
        "XLA_FLAGS": _XLA_FLAGS,
        "MARIN_DEBUG_HLO_UPLOAD_DIR": _DBG_UPLOAD_DIR,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_z4_r64_{TPU_TYPE.replace('-', '')}_pd{PER_DEVICE}_hlo_s2_{_DBG_RUN_TAG}",
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
        "experiment-z4",
        TPU_TYPE,
        "pd4",
        "r64-alpha64",
        "hlo-dump",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
