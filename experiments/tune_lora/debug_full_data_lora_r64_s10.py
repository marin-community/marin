# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# DEBUGSTART — debug_accum_tpu_type experiment M: full-data LoRA DPO, matched geometry
"""Experiment M (re-spec): LoRA DPO on the full bloom_speceval_v2 preference
distribution, paired on v5p-16 and v6e-16 with matched pd=4 on both sides.

Fills the (full-data, LoRA) quadrant of Codex's isolation matrix. The previous
per-stmt LoRA comparisons on v5p-8 (pd=-1→8) vs v6e-8 (pd=4) had two confounds
stacked: (1) tiny singleton dataset, (2) mismatched execution geometry
(different pd → different CE num_b_blocks). After Experiment L showed full-FT
matches cleanly under matched pd=4, Experiment M now controls for both.

Uses `mirrored()` for the source data so the parent can run from any region
(e.g. europe-west4) without tripping the executor's cross-region assertion.
The canonical `dpo_bloom_speceval_v2` step hard-pins source to us-central1
and cache to us-central2 — that fails the `len(gcs_regions) > 1` check from
any parent that can look up both bucket locations (ew4 especially).

Tradeoff: the cache hash differs from the canonical `tune_lora` sweep, so
this script will re-tokenize the bloom v2 training set once per region on
first use. For a 10-step probe that cost is acceptable and lets us spray
regions for scheduling speed.

Required env vars (injected into child TPU job via config.env_vars):
  MARIN_DEBUG_LOG_BATCH_INDICES=1   (loader: confirm same data across TPUs)
  MARIN_DEBUG_LOG_STEP_TRACE=1      (trainer: grad/param checksums per step)

Deliberately NOT enabled in this first pass: MARIN_DEBUG_LORA_FACTOR_TRACE.
First goal is a scalar-level v5p/v6e compare on loss + grad aggregates. Only
turn on the forward-factor trace in a follow-up if the scalar split is
ambiguous or we need to localize the divergence.

Resources: paired v5p-16 pd=4 and v6e-16 pd=4 → microbatch=64, no grad accum
on either TPU (matches Experiment L geometry).
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

# Full bloom_speceval_v2 preference distribution (all 46 statements, ~109k train ex).
# Wrapped with mirrored() so any parent region can tokenize/read without cross-region block.
TRAIN_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/*.jsonl.gz",
    budget_gb=5,
)
VAL_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)

tokenized_train = default_tokenize(
    name="bloom_speceval_v2_train_prefs_marin_tokenizer",
    dataset=TRAIN_DATA,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)
tokenized_eval = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=VAL_DATA,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"bloom_speceval_v2_val": tokenized_eval},
)

tpu = os.environ.get("TPU_TYPE", "v6e-16")
DEFAULT_REGIONS = {
    "v5p-16": ["us-central1", "us-east5"],
    "v6e-16": ["europe-west4", "us-east5", "us-east1"],
}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "m1")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="250g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    # Matched pd=4 on both TPUs → microbatch = 16 chips x 4 = 64 = train_batch_size (no grad accum).
    per_device_parallelism=4,
    per_device_eval_parallelism=4,
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,  # eval exactly once at step 10
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    # Checked-in Marin LoRA recipe (matches experiments/tune_lora/common.py).
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
    name=f"dpo/tune_lora/debug/full_data_lora_r64_s10_{tpu_short}_{_DBG_RUN_TAG}",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    tags=[
        "dpo",
        "lora-dpo",
        "bloom",
        "speceval-v2",
        "llama3",
        "marin-instruct",
        "debug-accum",
        "experiment-m",
        "full-data-lora-r64",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, training_step])
