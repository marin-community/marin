# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment O: within-family full-FT pd ablation
"""Experiment O: per-statement full-FT DPO with pd=2 for geometry ablation.

Sibling to `debug_full_ft_s10.py`. This keeps the same singleton
`support_mental_health` DPO setup and full fine-tune policy/reference path, but
reduces per-device parallelism so a v6e-16 run uses:

  chips=16, pd=2 -> local microbatch 32, grad_accum=2 for train_batch_size=64

The goal is to test whether full fine-tuning is robust to the same execution
geometry knob that changed the LoRA behavior, without changing TPU family.

Required env vars (injected into the child TPU job via config.env_vars):
  MARIN_DEBUG_LOG_BATCH_INDICES=1   (loader instrumentation: confirm same data)
  MARIN_DEBUG_LOG_STEP_TRACE=1      (trainer instrumentation: grad/param checksums)
"""
# DEBUGEND

import os

from levanter.data.text import PreferenceChatLmDatasetFormat
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import SeparateReferenceConfig

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, mirrored
from marin.processing.tokenize import lm_data_config

STMT_TRAIN = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/train/shard-00000.jsonl.gz",
    budget_gb=1,
)
STMT_VAL = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/val/shard-00000.jsonl.gz",
    budget_gb=1,
)
FULL_VAL = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)

tokenized_train = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_train_marin_tokenizer",
    dataset=STMT_TRAIN,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)
tokenized_stmt_val = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_val_marin_tokenizer",
    dataset=STMT_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)
tokenized_full_val = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=FULL_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={
        "stmt_val": tokenized_stmt_val,
        "full_val": tokenized_full_val,
    },
)

tpu = os.environ.get("TPU_TYPE", "v6e-16")
DEFAULT_REGIONS = {
    "v5p-16": ["us-central1", "us-east5"],
    "v6e-16": ["europe-west4", "us-east5", "us-east1"],
    # Fallbacks if 16-chip slices are unavailable.
    "v6e-32": ["europe-west4", "us-east5", "us-east1"],
    "v5p-32": ["us-central1", "us-east5"],
}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]

# Experiment O: hold global batch fixed at 64 while reducing local execution
# geometry. On 16-chip slices pd=2 => microbatch 32 and grad_accum=2.
PER_DEVICE = {
    "v5p-16": 2,
    "v6e-16": 2,
    "v6e-32": 1,
    "v5p-32": 1,
}
PER_DEVICE_EVAL = {
    "v5p-16": 2,
    "v6e-16": 2,
    "v6e-32": 1,
    "v5p-32": 1,
}

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "o1")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="250g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    reference=SeparateReferenceConfig(),
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
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
    },
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/full_ft_pd2_s10_{tpu_short}_{_DBG_RUN_TAG}",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    include_lm_validation=False,
    tags=[
        "dpo",
        "full-dpo",
        "bloom",
        "per-stmt",
        "support-mental-health",
        "debug-accum",
        "experiment-o",
        "per-stmt-full-ft",
        "pd2",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
