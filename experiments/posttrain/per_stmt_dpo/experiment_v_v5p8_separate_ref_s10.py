# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment V: v5p-8 LoRA DPO with SeparateReferenceConfig.

Identical to Experiment Q pd=4 except that the reference path is
`SeparateReferenceConfig` instead of `AdapterBaseReferenceConfig` (ABRC).

Why this is the next experiment:

- Every bad v5p-8 LoRA run uses ABRC, which computes the reference forward by
  reusing the same model with the LoRA adapter zeroed.
- The working v5p-8 full-FT run (Experiment T) used `SeparateReferenceConfig`
  (a physically separate reference model copy).
- Experiment U ruled out fp32-vs-bf16 numerical precision as the cause.
- The remaining single-variable structural confound is the reference path.

Outcomes:

- If V escapes ln(2) at step 2 and reaches the good ~0.32 regime, the
  ABRC code path is the load-bearing variable on v5p-8 LoRA.
- If V stays stuck near ln(2), the v5p-8 LoRA pathology is independent of
  the reference path — narrows the next probe to LoRA sharding / mesh
  topology.
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import SeparateReferenceConfig

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

TPU_TYPE = "v5p-8"
DEFAULT_REGIONS = ["us-east5", "us-central1"]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "v-pd4")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None),
    reference=SeparateReferenceConfig(),  # only diff vs Exp Q pd=4
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
    name=f"dpo/stmt_dpo/debug/experiment_v_r64_v5p8_pd{PER_DEVICE}_separate_ref_s10_{_DBG_RUN_TAG}",
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
        "experiment-v",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        "separate-reference",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
