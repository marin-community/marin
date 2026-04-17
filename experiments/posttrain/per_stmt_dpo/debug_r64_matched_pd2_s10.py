# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment N: per-stmt LoRA DPO on matched-family pods, pd=2
"""Experiment N: per-stmt LoRA DPO paired on v5p-16 / v6e-16 with matched pd=2.

Isolates the biggest remaining confound from the original pathological pair:
the old per-stmt LoRA runs compared v5p-8 (pd=-1→8, no accum) against v6e-8
(pd=4, accum=2). Both geometry and execution geometry were mismatched on
top of LoRA. This run holds all of {data, seed, recipe, reference config}
constant and makes TPU family the only variable, with pd=2 matched on both
sides.

All non-hardware knobs mirror the pathological smh_lr1em06_s70 baseline:
  - data: bloom_v2_singleton/support_mental_health
  - adapter: LoraAdaptationConfig(r=64, alpha=64, zero_init_b=True, target_modules=None)
  - reference: AdapterBaseReferenceConfig  (canonical LoRA DPO path)
  - lr=1e-6, beta=0.1, seed=0, train_seq_len=4096

Geometry changes (both sides):
  - TPU family upgraded to 16-chip pods to fit full-FT comparisons earlier
  - pd=2 so train_batch_size=64 yields microbatch=32, grad_accum=2 on BOTH sides
    (symmetric accum; no per-side geometry mismatch)

Debug env (injected via config.env_vars):
  MARIN_DEBUG_LOG_BATCH_INDICES=1   (confirm identical batches across TPUs)
  MARIN_DEBUG_LOG_STEP_TRACE=1      (grad/param checksums per step)

Interpretation (per Codex):
  - still splits by step 2-3 → LoRA (+ per-stmt) is the amplifier, not geometry
  - matches closely → original catastrophe was mostly geometry / execution path
  - in between → both contribute; follow-up probes needed
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

tpu = os.environ.get("TPU_TYPE", "v6e-16")
DEFAULT_REGIONS = {
    "v5p-16": ["us-central1", "us-east5"],
    "v6e-16": ["europe-west4", "us-east5", "us-east1"],
    "v6e-32": ["europe-west4", "us-east5", "us-east1"],
}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "n1")

# per_device chosen per TPU to keep microbatch=32 (symmetric grad_accum=2 with bs=64).
# v5p-16 and v6e-16 have 16 chips → pd=2 gives microbatch=32.
# v6e-32 has 32 chips → pd=1 gives microbatch=32 (fallback when v6e-16 quota is exhausted).
PER_DEVICE = {"v5p-16": 2, "v6e-16": 2, "v6e-32": 1}

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="250g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE[tpu],
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,  # one eval at step 10 on stmt_val + full_val
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    # Pathological recipe: r=64/a=64 zero_init_b=True, AdapterBaseReferenceConfig (canonical LoRA DPO path).
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
    },
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/r64_matched_pd2_s10_{tpu_short}_{_DBG_RUN_TAG}",
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
        "experiment-n",
        "matched-pd2",
        "r64-alpha64",
        tpu,
    ],
    # Short debug probe — skip paloma + uncheatable_eval LM validation sets so the
    # parent doesn't spend hours re-tokenizing ~20 datasets in each new region.
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
