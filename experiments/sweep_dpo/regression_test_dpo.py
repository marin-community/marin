# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off full-DPO regression run: old new_dpo_v2 training shape with deduped validation."""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

GCS_PREFIX = "gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite"
PREFERENCE_FORMAT = PreferenceChatLmDatasetFormat()

tokenized_train = default_tokenize(
    name="bloom_speceval_v2_train_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/train/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PREFERENCE_FORMAT,
)

# Keep the validation path explicit in this one-off repro so the comparison stays pinned even if the canonical
# Bloom experiment changes again.
tokenized_eval = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/val_deduped/shard-00000.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PREFERENCE_FORMAT,
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"bloom_speceval_v2_val": tokenized_eval},
)

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-32", ram="256g"),
    per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-32"],
    train_batch_size=128,
    num_train_steps=850,
    learning_rate=7.5e-7,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=200,
    seed=2,
)

training_step = default_dpo(
    name="dpo/regression_test_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2_deduped_val",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=[
        "dpo",
        "bloom",
        "speceval-v2",
        "llama3",
        "marin-instruct",
        "beta0.1",
        "lr7.5e-7",
        "regression-test",
        "deduped-val",
        "current-validation-stack",
    ],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step,
        ]
    )
