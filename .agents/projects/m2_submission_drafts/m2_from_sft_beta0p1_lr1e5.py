# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# STAGING DRAFT — target path on dpo-lora-clean:
# experiments/tune_lora/m2_from_sft_beta0p1_lr1e5_seed0_b64.py

"""M2 pilot run: LoRA DPO on bloomv2_m2, policy starts from SFT (marin-8b-instruct).

Identical training setup to M1 (bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8)
except for the dataset, which is bloomv2 + 40-60 pilot tension preference pairs
from the 10-point pilot.

This is the "is the tension data even detectable?" configuration — same base,
same LR, same batch, just different dataset.
"""

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo
from experiments.dpo_bloomv2_m2 import (
    tokenized_eval,
    tokenized_preferences,
    tokenized_train,
)
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

SLUG = "m2_from_sft_bloomv2_m2_beta0p1_lr1e5_seed0_b64_v5p8"
LORA_RUN_PREFIX = "lora_"


def _config() -> SimpleDPOConfig:
    return SimpleDPOConfig(
        resources=ResourceConfig.with_tpu("v5p-8", ram="400g"),
        per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-8"],
        train_batch_size=64,
        num_epochs=1.0,
        learning_rate=1e-5,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        # Start from SFT — same base that M1 used.
        model_name_or_path="marin-community/marin-8b-instruct",
        adapter=LoraAdaptationConfig(
            r=64,
            alpha=64,
            dropout=0.0,
            zero_init_b=True,
            target_modules=None,
        ),
        reference=AdapterBaseReferenceConfig(),  # reference = adapter base = SFT
        train_seq_len=4096,
        max_seq_len=4096,
        beta=0.1,
        validation_split_fraction=None,
        reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
        steps_per_checkpoint=200,
        steps_per_hf_export=200,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
    )


def _step():
    tags = [
        "dpo",
        "lora-dpo",
        "bloomv2",
        "bloomv2-m2",
        "pilot-tension-10pt",
        "llama3",
        "marin-instruct",
        "from-sft",
        "beta0.1",
        "lr1e-5",
        "seed0",
        "b64",
        "r64",
        "v5p-8",
    ]
    return default_dpo(
        name=f"dpo/tune_lora/{LORA_RUN_PREFIX}{SLUG}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=_config(),
        tags=tags,
    )


if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, _step()])
