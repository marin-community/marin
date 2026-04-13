# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo
from experiments.dpo_bloom_speceval_v2 import tokenized_eval, tokenized_preferences, tokenized_train
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

LORA_RUN_PREFIX = "lora_"


@dataclass(frozen=True)
class LoraTuneSpec:
    slug: str
    learning_rate: float
    seed: int
    beta: float = 0.1
    train_batch_size: int = 64
    num_epochs: float = 1.0
    rank: int = 64


def bloom_speceval_v2_lora_config(spec: LoraTuneSpec) -> SimpleDPOConfig:
    return SimpleDPOConfig(
        resources=ResourceConfig.with_tpu("v5p-8", ram="400g"),
        per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-8"],
        train_batch_size=spec.train_batch_size,
        num_epochs=spec.num_epochs,
        learning_rate=spec.learning_rate,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path="marin-community/marin-8b-instruct",
        adapter=LoraAdaptationConfig(
            r=spec.rank,
            alpha=spec.rank,
            dropout=0.0,
            zero_init_b=True,
            target_modules=None,
        ),
        reference=AdapterBaseReferenceConfig(),
        train_seq_len=4096,
        max_seq_len=4096,
        beta=spec.beta,
        validation_split_fraction=None,
        reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
        steps_per_checkpoint=200,
        steps_per_hf_export=200,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=spec.seed,
    )


def bloom_speceval_v2_lora_step(spec: LoraTuneSpec):
    beta_tag = f"beta{spec.beta:g}"
    lr_tag = f"lr{spec.learning_rate:g}"
    seed_tag = f"seed{spec.seed}"
    batch_tag = f"b{spec.train_batch_size}"
    rank_tag = f"r{spec.rank}"
    tags = [
        "dpo",
        "lora-dpo",
        "bloom",
        "speceval-v2",
        "llama3",
        "marin-instruct",
        "executor",
        "tune-lora",
        "reference-logprob-cache",
        beta_tag,
        lr_tag,
        seed_tag,
        batch_tag,
        rank_tag,
    ]

    return default_dpo(
        name=f"dpo/tune_lora/{LORA_RUN_PREFIX}{spec.slug}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=bloom_speceval_v2_lora_config(spec),
        tags=tags,
    )


def run_executor(spec: LoraTuneSpec) -> None:
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            bloom_speceval_v2_lora_step(spec),
        ]
    )
