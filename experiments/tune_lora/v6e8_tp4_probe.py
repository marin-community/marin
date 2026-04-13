# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-8 LoRA DPO with TP=4+FSDP=2 — pd=8, no gradient accumulation.

Uses tensor parallelism to shard within-layer intermediates (attention scores,
MLP, logits) across 4 chips, with FSDP across 2 groups. This reduces per-chip
HBM from 44.23 GB to estimated ~28 GB, enabling per_device=8 without grad accum
and without carry offloading (which crashes on v6e due to XLA codegen bug).
"""

from levanter.callbacks.profiler import ProfilerConfig

from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.tune_lora.v6e8_probe_multiregion import (
    default_dpo,
    tokenized_preferences,
)
from experiments.simple_dpo_config import SimpleDPOConfig
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig


def make_tp4_probe(regions: list[str], name_suffix: str):
    config = SimpleDPOConfig(
        resources=ResourceConfig.with_tpu("v6e-8", regions=regions),
        per_device_parallelism=-1,  # auto → pd=8 with TP=4+FSDP=2
        per_device_eval_parallelism=4,
        train_batch_size=64,
        num_train_steps=20,
        steps_per_eval=20,
        learning_rate=5e-6,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path="marin-community/marin-8b-instruct",
        adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None),
        reference=AdapterBaseReferenceConfig(),
        train_seq_len=4096,
        max_seq_len=4096,
        beta=0.1,
        validation_split_fraction=None,
        reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
        steps_per_checkpoint=200,
        steps_per_hf_export=200,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
        profiler=ProfilerConfig(enabled=True, start_step=5, num_steps=10),
    )

    return default_dpo(
        name=f"dpo/tune_lora/v6e8_tp4{name_suffix}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=config,
        tags=["dpo", "lora-dpo", "bloom", "speceval-v2", "llama3", "marin-instruct", "v6e-tp4"],
    )
