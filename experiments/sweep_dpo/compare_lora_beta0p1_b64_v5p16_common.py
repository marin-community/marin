# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared full-DPO config for beta=0.1 batch-64 one-epoch comparison runs."""

from experiments.dpo_bloom_speceval_v2 import tokenized_eval, tokenized_preferences, tokenized_train
from experiments.defaults import default_dpo
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main


def dpo_config_for_seed(seed: int) -> SimpleDPOConfig:
    return SimpleDPOConfig(
        resources=ResourceConfig.with_tpu("v5p-16", ram="256g"),
        per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-16"],
        train_batch_size=64,
        num_epochs=1.0,
        learning_rate=5e-7,
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
        steps_per_checkpoint=1000,
        steps_per_hf_export=200,
        seed=seed,
    )


def training_step_for_seed(seed: int):
    return default_dpo(
        name=f"dpo/compare_lora/bloom_speceval_v2_marin_instruct_beta0.1_seed{seed}_b64_v5p16",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=dpo_config_for_seed(seed),
        tags=[
            "dpo",
            "full-dpo",
            "bloom",
            "speceval-v2",
            "llama3",
            "marin-instruct",
            "beta0.1",
            "compare-lora",
            "b64",
            "v5p16",
            "one-epoch",
        ],
    )


def run_executor(seed: int) -> None:
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step_for_seed(seed),
        ]
    )
