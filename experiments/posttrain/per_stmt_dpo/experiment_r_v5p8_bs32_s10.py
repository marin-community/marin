# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment R: v5p-8 local-shape matching probe.

Follow-up to Exp Q. Exp Q showed that changing `per_device_parallelism` from
8 to 4 on v5p-8 does not move the local CE kernel shape (both runs logged
`DEBUGCE x.shape=(65536, 4096) b_block_size=1024 num_b_blocks=64`), and that
both remained stuck near ln(2). Meanwhile the good v5p-16 pd=2 Exp N run
logged `DEBUGCE x.shape=(32768, 4096) b_block_size=32768 num_b_blocks=1`.

The highest-information next move is therefore to lower `train_batch_size`
(not `pd`), since `train_batch_size` is the only knob left that should
actually change the per-chip local CE workload on v5p-8.

Exact change vs Exp Q:
- `train_batch_size: 64 -> 32`
- `per_device_parallelism = 4` (same as the known-working Exp Q pd=4 path)

Everything else — LoRA recipe, data, seed, beta, lr, reference config,
reference_eval_cache, max_eval_batches, num_train_steps — matches Exp Q.

Override knobs via env:
- `EXPERIMENT_R_BS` (default 32)
- `EXPERIMENT_R_PD` (default 4)
- `REGIONS_OVERRIDE`
- `MARIN_DEBUG_RUN_TAG`
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

TPU_TYPE = "v5p-8"
DEFAULT_REGIONS = ["us-central1", "us-east5"]
DEFAULT_BS = 32
DEFAULT_PD = 4
ALLOWED_PD = {2, 4, 8}

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_R_BS", str(DEFAULT_BS)))

_pd_raw = os.environ.get("EXPERIMENT_R_PD", str(DEFAULT_PD))
PER_DEVICE = int(_pd_raw)
if PER_DEVICE not in ALLOWED_PD:
    raise ValueError(f"Experiment R only supports per_device_parallelism in {sorted(ALLOWED_PD)}, got {PER_DEVICE}")

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"rbs{TRAIN_BATCH_SIZE}pd{PER_DEVICE}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g" if PER_DEVICE <= 4 else "400g", regions=REGIONS_FOR_TPU),
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
    name=f"dpo/stmt_dpo/debug/experiment_r_r64_v5p8_bs{TRAIN_BATCH_SIZE}_pd{PER_DEVICE}_s10_{_DBG_RUN_TAG}",
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
        "experiment-r",
        "v5p-8",
        f"bs{TRAIN_BATCH_SIZE}",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
