# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment K: LoRA-rank ablation on per-stmt DPO
"""Experiment K: swap per-stmt LoRA DPO to the actual Marin r=64/alpha=64 recipe.

Keeps the pathological per-stmt setup (same data, same lr, same beta, same
seed, same seq len, same `zero_init_b=True`, same target_modules=None) and
swaps ONLY the adapter config to the checked-in tune_lora convention:

    original (pathological): r=16, alpha=32   (alpha/r = 2, rank 16)
    this probe:              r=64, alpha=64   (alpha/r = 1, rank 64 — matches
                                               experiments/tune_lora/README.md)

Note: this changes BOTH rank and adapter scale vs. the pathological run.
That's intentional — the question is "does swapping to the real Marin LoRA
recipe fix v5p/v6e divergence on per-stmt data?", not "is rank alone enough?".
If this still splits, a follow-up r=64/alpha=128 isolates rank from scale.

10 training steps is enough — the v5p↔v6e split was visible by step 2 in
Experiment J. Paired launch on v5p-8 and v6e-8 via `TPU_TYPE` env.

Required env vars (set on the iris parent via -e, or inject via config.env_vars):
  MARIN_DEBUG_LOG_BATCH_INDICES=1   (loader instrumentation: confirm same data)
  MARIN_DEBUG_LOG_STEP_TRACE=1      (trainer instrumentation: grad/param checksums)

Interpretation:
  * v5p ≈ v6e at r=64 → rank 16 bottleneck was amplifying bf16 noise → LoRA
    capacity / conditioning is a real factor in the pathology.
  * v5p ≠ v6e at r=64 → rank is not the main culprit; LoRA-DPO regime itself
    is the fragile piece. Next probe: full bloom_v2 LoRA DPO on both TPUs.
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

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "k1")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="150g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
    train_batch_size=64,
    num_train_steps=10,  # 10 steps per Codex (split is visible by step 2)
    steps_per_eval=10,  # eval exactly once at step 10
    learning_rate=1e-6,  # same as pathological smh_lr1em06_s70
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    # Swap to the actual Marin LoRA recipe (r=64, alpha=64) — matches
    # experiments/tune_lora/README.md and experiments/tune_lora/beta0p1_*.
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
    # Inject debug env vars into the child TPU training job. Ambient shell env
    # vars don't propagate through iris job-spawn; must set here.
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
    },
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/r64_alpha64_s10_{tpu_short}_{_DBG_RUN_TAG}",
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
        "experiment-k",
        "rank-ablation",
        "r64-alpha64",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
