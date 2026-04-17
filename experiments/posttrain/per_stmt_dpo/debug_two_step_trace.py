# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment J: 2-step deterministic trace
"""Experiment J: 2-step trace that logs actual batch indices + trainable-only
gradient/update/param checksums on both v6e-8 and v5p-8.

Requires these env vars to activate the debug logging:
  MARIN_DEBUG_LOG_BATCH_INDICES=1   (loader.py instrumentation)
  MARIN_DEBUG_LOG_STEP_TRACE=1      (trainer.py instrumentation)

Answers:
  - Do both TPU types see the SAME training batch indices? (data path)
  - On the same batch, do trainable grad checksums match? (compute path)
  - Do optimizer updates match given matching grads? (optimizer state)
  - Do param checksums match after step 1? (round-trip)

Config: exactly matches smh_lr1em06_s35 but with num_train_steps=2 and NO eval.
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
# Allow override via env var (comma-separated)
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]
PER_DEVICE = {"v5p-8": -1, "v6e-8": 4}
PER_DEVICE_EVAL = {"v5p-8": 16, "v6e-8": 4}

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="250g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
    train_batch_size=64,
    num_train_steps=2,  # ONLY 2 STEPS
    steps_per_eval=9999,  # disable eval during training (eval only runs when step % steps_per_eval == 0)
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
    reference_eval_cache=ReferenceEvalCacheConfig(mode="disabled"),  # skip building ref cache
    steps_per_checkpoint=9999,  # no intermediate checkpoints
    steps_per_hf_export=9999,  # no HF exports
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/two_step_trace_{tpu_short}",
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
        "experiment-j",
        "two-step-trace",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
