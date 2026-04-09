# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off LoRA DPO smoke run to validate short training and merged HF export."""

from dataclasses import replace

from experiments.defaults import default_dpo
from experiments.dpo_bloom_speceval_v2 import tokenized_eval, tokenized_preferences, tokenized_train
from experiments.llama import llama_8b
from experiments.tune_lora.common import (
    LORA_RUN_PREFIX,
    LoraTuneSpec,
    bloom_speceval_v2_lora_config,
)
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

SMOKE_SPEC = LoraTuneSpec(
    slug="smoke_export_marin_8b_instruct_v5p8_5step",
    learning_rate=5e-6,
    seed=0,
)


def smoke_lora_dpo_config():
    base_config = bloom_speceval_v2_lora_config(SMOKE_SPEC)
    return replace(
        base_config,
        resources=ResourceConfig.with_tpu("v5p-8", ram="256g"),
        num_train_steps=5,
        steps_per_eval=5,
        steps_per_checkpoint=5,
        steps_per_hf_export=5,
    )


def smoke_lora_dpo_step():
    tags = [
        "dpo",
        "lora-dpo",
        "bloom",
        "speceval-v2",
        "llama3",
        "marin-instruct",
        "executor",
        "tune-lora",
        "smoke-test",
        "hf-export",
        "v5p-8",
        "b64",
        "steps5",
        "lr5e-6",
        "seed0",
        "r64",
    ]
    return default_dpo(
        name=f"dpo/tune_lora/{LORA_RUN_PREFIX}{SMOKE_SPEC.slug}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=smoke_lora_dpo_config(),
        tags=tags,
    )


if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            smoke_lora_dpo_step(),
        ]
    )
