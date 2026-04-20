# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment N-closure: the original v5p-16 vs v6e-16 matched-pod pair,
rerun with ``a_init_mode="zero"`` to confirm the fix doesn't *degrade*
the "already working" configuration.

The original Experiment N
(``experiments/posttrain/per_stmt_dpo/debug_r64_matched_pd2_s10.py``,
2026-04-15) ran this exact recipe on v5p-16 pd=2 and v6e-16 pd=2
(both ``|data|=8``) and showed the two loss curves tracking each
other closely — evidence that TPU family alone doesn't explain the
v5p/v6e split. Step-9 losses at r=64/α=64 were ~0.317 (v5p-16) and
~0.319 (v6e-16), a gap of ~0.002.

Under ``a_init_mode="zero"`` we expect:
- v5p-16 and v6e-16 remain bit-identical-to-machine-precision
  (cross-family closure on ``|data|=8``).
- Absolute losses will change (different init, different trajectory)
  but the *gap* should be ≤0.005, matching or improving on Exp N's
  already-small cross-pod gap.

This is the counterpart closure check to Experiment K-closure
(v5p-8 vs v6e-8): K-closure proved the fix *eliminates* a large
divergence; N-closure proves the fix *preserves* an existing
already-close pair. Together they establish that the init flip is a
strict improvement at both pod sizes.

Usage:

    TPU_TYPE=v5p-16 REGIONS_OVERRIDE=us-central1  python experiment_n_closure_azero_matched_s10.py
    TPU_TYPE=v5p-16 REGIONS_OVERRIDE=us-east5     python experiment_n_closure_azero_matched_s10.py
    TPU_TYPE=v6e-16 REGIONS_OVERRIDE=europe-west4 python experiment_n_closure_azero_matched_s10.py
    TPU_TYPE=v6e-16 REGIONS_OVERRIDE=us-east5     python experiment_n_closure_azero_matched_s10.py
    TPU_TYPE=v6e-16 REGIONS_OVERRIDE=us-east1     python experiment_n_closure_azero_matched_s10.py
"""

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
}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]
PER_DEVICE = {"v5p-16": 2, "v6e-16": 2}

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "nclosure")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="250g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE[tpu],
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(
        r=64,
        alpha=64,
        dropout=0.0,
        zero_init_b=False,
        a_init_mode="zero",
        target_modules=None,
    ),
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
        "MARIN_DEBUG_LORA_DEBUG": "1",
        "MARIN_DEBUG_SKIP_HF_EXPORT": "1",
    },
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/n_closure_azero_matched_s10_{tpu_short}_{_DBG_RUN_TAG}",
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
        "experiment-n-closure",
        "a-init-zero",
        "bug-1-closure",
        "r64-alpha64",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
