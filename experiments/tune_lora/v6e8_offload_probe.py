# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-8 LoRA DPO with carry offloading — pd=8, no gradient accumulation.

Offloads checkpoint saves (carries) from HBM to pinned host memory,
freeing ~17 GB HBM. This should allow per_device=8 without grad accum,
which halves weight reads and eliminates the 9% buffer allocation stalls
observed in xprof.

Expected: ~27 GB HBM (down from 44.23 at pd=8 without offloading).
Host RAM: ~65 GB (well within the 128 GB request).
"""
import dataclasses

from levanter.callbacks.profiler import ProfilerConfig

from experiments.llama import llama_8b
from experiments.tune_lora.v6e8_probe_multiregion import (
    default_dpo,
    tokenized_eval,
    tokenized_preferences,
    tokenized_train,
)
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.simple_dpo_config import SimpleDPOConfig
from experiments.marin_models import marin_tokenizer
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS
from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

# Override llama_8b to use carry offloading
llama_8b_offload = dataclasses.replace(llama_8b, gradient_checkpointing="offload")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v6e-8", regions=["europe-west4", "us-east5"]),
    per_device_parallelism=-1,  # auto → should be pd=8 with offloaded carries
    per_device_eval_parallelism=4,  # eval still needs to fit without offloading
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

step = default_dpo(
    name="dpo/tune_lora/v6e8_offload",
    tokenized=tokenized_preferences,
    model_config=llama_8b_offload,
    dpo_config=config,
    tags=["dpo", "lora-dpo", "bloom", "speceval-v2", "llama3", "marin-instruct", "v6e-offload"],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, step])
