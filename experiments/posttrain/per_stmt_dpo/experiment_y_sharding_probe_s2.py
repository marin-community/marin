# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment Y: dump FSDP sharding specs for LoRA params on v5p-8 vs v6e-8.

Identical to Experiment Q pd=4 except:

- ``num_train_steps=2`` — we only need initial-state sharding info plus
  enough to confirm the run reaches the training loop.
- Sets ``MARIN_DEBUG_DUMP_SHARDING=1`` so `train_dpo.main` prints a
  ``DEBUGJ SHARDING ...`` line for every LoRA A/B parameter and its
  optimizer state, recording partition spec, named axes, mesh shape,
  and physical mesh mapping at init.
- TPU type is selected via ``EXPERIMENT_Y_TPU`` (default ``v5p-8``,
  allowed: ``v5p-8``, ``v6e-8``). Regions default per TPU family.

Why: Exp W showed that pure-TP on v5p-8 recovers LoRA DPO training
while the canonical FSDP mesh does not. Exp W did not establish
*why* the FSDP-based v6e-8 run (which has the same ``pd=4`` but 8
chips and 1 host) trains correctly. This probe collects the raw
partition-spec layout from both TPU types so we can compare element
by element — specifically:

- is the FSDP shard of LoRA A / LoRA B the same logical layout?
- do the mesh shapes map to the same physical devices?
- is some axis silently mapped to an axis that is size 1 on one
  TPU type but size 4 on the other?

Expected outcome: the two dumps differ in a concrete,
inspectable way that explains the training-time divergence.

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

_ALLOWED_TPUS = {
    "v5p-8": ["us-east5", "us-central1"],
    "v6e-8": ["us-east5", "us-east1", "europe-west4"],
}

TPU_TYPE = os.environ.get("EXPERIMENT_Y_TPU", "v5p-8")
if TPU_TYPE not in _ALLOWED_TPUS:
    raise ValueError(f"EXPERIMENT_Y_TPU must be one of {sorted(_ALLOWED_TPUS)}, got {TPU_TYPE}")

DEFAULT_REGIONS = _ALLOWED_TPUS[TPU_TYPE]
PER_DEVICE = 4

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"y-{TPU_TYPE.replace('-', '')}")

# v5p-8 has 95 GB/chip HBM, v6e-8 has 32 GB/chip. Keep ram request sized for
# the smaller platform when on v6e; v5p gets the usual 250g.
_RAM = "250g" if TPU_TYPE == "v5p-8" else "400g"

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram=_RAM, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=64,
    num_train_steps=2,
    steps_per_eval=2,
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
        "MARIN_DEBUG_DUMP_SHARDING": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_y_r64_{TPU_TYPE.replace('-', '')}_pd{PER_DEVICE}_sharding_s2_{_DBG_RUN_TAG}",
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
        "experiment-y",
        TPU_TYPE,
        "pd4",
        "r64-alpha64",
        "sharding-probe",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
