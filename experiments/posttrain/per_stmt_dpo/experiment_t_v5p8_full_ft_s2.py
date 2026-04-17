# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment T: v5p-8 full-FT DPO feasibility / behavior probe.

This is the first direct test of whether the remaining `v5p-8` pathology is
specific to LoRA / AdapterBaseReferenceConfig or broader to the `v5p-8`
distributed regime.

It intentionally starts as a short compile-and-behavior probe:
- full fine-tuning (no adapter)
- SeparateReferenceConfig
- v5p-8
- batch size 32
- pd=4
- 2 train steps

Checkpointing policy is configurable via env so we can move through the
pre-registered fallback ladder without cloning more scripts:
- EXPERIMENT_T_CHECKPOINTING=default   -> use llama_8b as-is
- EXPERIMENT_T_CHECKPOINTING=offload   -> gradient_checkpointing="offload"
- EXPERIMENT_T_CHECKPOINTING=recompute -> gradient_checkpointing="recompute"
"""

import dataclasses
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

TPU_TYPE = "v5p-8"
DEFAULT_REGIONS = ["us-east5"]
DEFAULT_BS = 32
DEFAULT_PD = 4
DEFAULT_STEPS = 2
DEFAULT_CHECKPOINTING = "offload"
ALLOWED_PD = {2, 4}
ALLOWED_CHECKPOINTING = {"default", "offload", "recompute"}

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_T_BS", str(DEFAULT_BS)))
PER_DEVICE = int(os.environ.get("EXPERIMENT_T_PD", str(DEFAULT_PD)))
if PER_DEVICE not in ALLOWED_PD:
    raise ValueError(f"Experiment T only supports per_device_parallelism in {sorted(ALLOWED_PD)}, got {PER_DEVICE}")

NUM_TRAIN_STEPS = int(os.environ.get("EXPERIMENT_T_STEPS", str(DEFAULT_STEPS)))
CHECKPOINTING_POLICY = os.environ.get("EXPERIMENT_T_CHECKPOINTING", DEFAULT_CHECKPOINTING)
if CHECKPOINTING_POLICY not in ALLOWED_CHECKPOINTING:
    raise ValueError(
        f"Experiment T only supports checkpointing in {sorted(ALLOWED_CHECKPOINTING)}, got {CHECKPOINTING_POLICY}"
    )

if CHECKPOINTING_POLICY == "default":
    model_config = llama_8b
else:
    model_config = dataclasses.replace(llama_8b, gradient_checkpointing=CHECKPOINTING_POLICY)

_DBG_RUN_TAG = os.environ.get(
    "MARIN_DEBUG_RUN_TAG",
    f"tbs{TRAIN_BATCH_SIZE}pd{PER_DEVICE}{CHECKPOINTING_POLICY}",
)

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    steps_per_eval=NUM_TRAIN_STEPS,
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

training_step = default_dpo(
    name=f"exp_t_v5p8_fullft_bs{TRAIN_BATCH_SIZE}_pd{PER_DEVICE}_{CHECKPOINTING_POLICY}_s{NUM_TRAIN_STEPS}_{_DBG_RUN_TAG}",
    tokenized=tokenized_preferences,
    model_config=model_config,
    dpo_config=config,
    tags=[
        "dpo",
        "full-dpo",
        "bloom",
        "per-stmt",
        "support-mental-health",
        "debug-accum",
        "experiment-t",
        "v5p-8",
        "full-ft",
        f"bs{TRAIN_BATCH_SIZE}",
        f"pd{PER_DEVICE}",
        f"ckpt-{CHECKPOINTING_POLICY}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
