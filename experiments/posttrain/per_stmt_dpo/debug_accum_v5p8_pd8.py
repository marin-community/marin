# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: v5p-8 with per_device_parallelism=8 (forces 2x grad accum).

Tests the hypothesis that the v6e-8 vs v5p-8 training discrepancy is caused by
the microbatch/grad-accum code path. This run forces v5p-8 to use the same
microbatch regime as v6e-8:

  microbatch_size = pd(8) x devices(4) = 32
  num_micro_steps = batch(64) / 32 = 2

Compare against:
  - smh_lr1em06_s70_v5p8-964129  (pd=16, NO grad accum — slow learner)
  - smh_lr1em06_s70_v6e8-fbac2a  (pd=4, 2x grad accum — fast learner)

See .agents/logbooks/debug_accum_tpu_type.md for full context.
"""

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

# --- Same data as exp 1a (reuse tokenized caches) ---

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

# --- v5p-8 config with forced pd=8 (2x grad accum, microbatch=32) ---

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-8", ram="400g", regions=["us-central1", "us-east5"]),
    per_device_parallelism=8,  # KEY CHANGE: forces microbatching (normally -1 → 16)
    per_device_eval_parallelism=8,  # match training pd for consistency
    train_batch_size=64,
    num_train_steps=70,  # match the clean s70 comparison pair
    steps_per_eval=23,  # ~3 evals
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(
        r=16,
        alpha=32,
        dropout=0.0,
        zero_init_b=True,
        target_modules=None,
    ),
    reference=AdapterBaseReferenceConfig(),
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
    steps_per_checkpoint=70,
    steps_per_hf_export=70,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
)

training_step = default_dpo(
    name="dpo/stmt_dpo/debug/smh_lr1em06_s70_v5p8_pd8",
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
        "v5p-8",
        "pd8",
        "forced-grad-accum",
    ],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_stmt_val,
            tokenized_full_val,
            training_step,
        ]
    )
