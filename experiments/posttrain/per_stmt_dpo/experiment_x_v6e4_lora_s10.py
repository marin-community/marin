# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment X: v6e-4 LoRA DPO (4-chip cross-family probe).

Identical to Experiment Q pd=4 except that the TPU type is `v6e-4` instead of
`v5p-8`. This is the architectural discriminator between two leading
hypotheses:

- "4-chip mesh causes the LoRA DPO pathology, regardless of TPU family."
  Both v5p-8 and v6e-4 have 4 chips on a single host. If v6e-4 also fails,
  the cause is the small-mesh sharding regime, independent of v5p hardware.
- "v5p-8 specifically has a hardware/topology bug in the LoRA path."
  If v6e-4 succeeds while v5p-8 fails on the same recipe, the failure is
  isolated to v5p-8 collectives / HLO / kernel choices.

Reference path is left as `AdapterBaseReferenceConfig` to match Experiment Q
exactly, so the cross-family contrast is a single-variable change. Experiment
V independently probes the reference-path question on v5p-8.

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

TPU_TYPE = "v6e-4"
DEFAULT_REGIONS = ["us-east5", "us-east1", "europe-west4"]
# v6e-4: 4 chips * 32GB HBM = 128GB total. Llama-8B LoRA at bs=64 pd=4 OOMs
# during XLA compile; bs=64 pd=2/1 also fails on program-load budget.
# Match Experiment R (bs=32 pd=4 on v5p-8) instead: same seq_len=4096,
# same pd, half the batch. Per-chip microbatch load equals v6e-8 pd=4,
# which has been validated working. This keeps the cross-family 4-chip
# test scientifically tight vs Exp R's known-bad v5p-8 baseline.
PER_DEVICE = int(os.environ.get("EXPERIMENT_X_PD", "4"))
TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_X_BS", "32"))

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"x-bs{TRAIN_BATCH_SIZE}-pd{PER_DEVICE}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
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
    name=f"dpo/stmt_dpo/debug/experiment_x_r64_v6e4_bs{TRAIN_BATCH_SIZE}_pd{PER_DEVICE}_s10_{_DBG_RUN_TAG}",
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
        "experiment-x",
        "v6e-4",
        "pd4",
        "r64-alpha64",
        "cross-family-4chip",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
