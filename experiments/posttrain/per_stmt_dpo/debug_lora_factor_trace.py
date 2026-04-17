# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment J2: 1-step LoRA factor trace
"""Experiment J2: 1-step trace that logs the LoRA forward factor z and
upstream cotangent for every LoRA module on both v6e-8 and v5p-8.

Requires these env vars:
  MARIN_DEBUG_LORA_FACTOR_TRACE=1    (lora.py instrumentation)
  MARIN_DEBUG_LOG_BATCH_INDICES=1    (confirm same batches)
  MARIN_DEBUG_LOG_STEP_TRACE=1       (keep grad/param checksums)

Expected log lines per LoRA module:
  DEBUGJ LORA_FWD stage=z_after_lora_A tag=... l2=... sum=...
  DEBUGJ LORA_BWD stage=lora_B_output tag=... l2=... sum=...

The `tag` identifies module shape, e.g.:
  embed4096_... → q_proj/k_proj/v_proj/gate_proj/up_proj input
  mlp14336      → gate_proj/up_proj output (intermediate)
  embed4096     → o_proj/down_proj output
"""
# DEBUGEND

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.data.text import PreferenceChatLmDatasetFormat
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

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

tpu = os.environ.get("TPU_TYPE", "v6e-8")
DEFAULT_REGIONS = {"v5p-8": ["us-central1", "us-east5"], "v6e-8": ["europe-west4", "us-east5", "us-east1"]}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]
PER_DEVICE = {"v5p-8": -1, "v6e-8": 4}
PER_DEVICE_EVAL = {"v5p-8": 16, "v6e-8": 4}

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="150g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
    train_batch_size=64,
    num_train_steps=1,  # ONLY 1 STEP (we just need the forward/backward trace)
    steps_per_eval=9999,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=16, alpha=32, dropout=0.0, zero_init_b=True, target_modules=None),
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
    # DEBUGSTART — minimize eval to avoid print-induced slowdown (can't be 0 due to DataLoader bug)
    max_eval_batches=1,
    # Explicitly inject debug env vars into the child TPU training job (per Codex feedback).
    # Ambient shell env vars don't reliably propagate — must set here on the config.
    env_vars={
        "MARIN_DEBUG_LORA_FACTOR_TRACE": "1",
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
    },
    # DEBUGEND
)

tpu_short = tpu.replace("-", "")
# DEBUGSTART — version suffix forces fresh output so executor doesn't skip cached runs
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "j3")
# DEBUGEND
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/lora_factor_trace_{tpu_short}_{_DBG_RUN_TAG}",
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
        "experiment-j2",
        "lora-factor-trace",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
