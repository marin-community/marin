# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment C8: continuous relative noise on LoRA grads under c=f32.

Purpose
=======

BD v1 (std=1e-5 one-shot step 1) did nothing. BD v2 (std=1e-2 one-shot
step 1) hurt. These tested ONE-SHOT noise at a specific step. What if
the c=bf16 benefit comes from CONTINUOUS per-step noise akin to bf16's
rounding error at every op?

C8 injects Gaussian noise RELATIVE to per-leaf grad norm, every step,
on both lora_A and lora_B. Scale 0.03 (~3% relative) matches the
effective bf16 rounding noise accumulated over ~32 transformer layers
(each contributing ~2^-7 = 0.8% relative noise per matmul).

Env:
  MARIN_DEBUG_LORA_GRAD_NOISE_RELATIVE=0.03
  MARIN_DEBUG_LORA_GRAD_NOISE_TARGET=both

Decision rule
=============

- Step 9 ≤ 0.40 (approaches Exp N 0.32) → continuous relative noise
  rescues. bf16's rounding IS acting as stochastic regularization, and
  it's the per-step continuous aspect that mattered.
- Step 9 ≥ 0.65 (tracks AC) → continuous relative noise does not
  rescue. The c=bf16 benefit is NOT reducible to random Gaussian noise
  on grads.

Note
----

New runs in this campaign should use the same-region GCS copy of
``Llama-3.1-8B-Instruct`` as the base model (see
``experiments.posttrain.per_stmt_dpo.base_model.LLAMA_3_1_8B_INSTRUCT_GCS_PATH``).
Loading is ~3-5x faster than the HuggingFace Hub CDN and Bug-1 has been
verified model-agnostic. Do NOT retroactively edit the ``model_name_or_path``
here - its historical value is tied to the run artifacts. This note is
for future forks of this experiment.
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

import jmp

from experiments.defaults import default_dpo
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.per_stmt_dpo.common import (
    tokenized_full_val,
    tokenized_preferences,
    tokenized_stmt_val,
    tokenized_train,
)
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

TPU_TYPE = "v5p-16"
REGIONS = ["us-central1", "us-east5"]
PER_DEVICE = 2

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_NOISE_REL = os.environ.get("MARIN_DEBUG_LORA_GRAD_NOISE_RELATIVE", "0.03")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"c8-v5p16-fp32-relnoise{_NOISE_REL}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None),
    reference=AdapterBaseReferenceConfig(),
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="disabled"),
    steps_per_checkpoint=9999,
    steps_per_hf_export=9999,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    mp=jmp.get_policy("p=f32,c=f32"),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    lr_schedule="cosine",
    warmup=0.1,
    seed=0,
    max_eval_batches=1,
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
        "MARIN_DEBUG_LORA_GRAD_NOISE_RELATIVE": _NOISE_REL,
        "MARIN_DEBUG_LORA_GRAD_NOISE_TARGET": "both",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_c8_r64_v5p16_pd{PER_DEVICE}_fp32_relnoise_s10_{_DBG_RUN_TAG}",
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
        "experiment-c8",
        "class-c",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "cont-rel-noise",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
