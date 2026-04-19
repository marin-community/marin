# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment AD: Exp N recipe + c=f32 + zero_init_b=False.

Purpose
=======

Narrow test of hypothesis H_B: does disabling LoRA's zero-init-B
recover training at c=f32 on the otherwise-good v5p-16 pd=2 mesh?

AC showed that taking the Exp N recipe and switching c=bf16 to
c=f32 breaks training on a known-good mesh. AD changes only
zero_init_b (True → False) on top of AC's config. Strict
single-knob deviation from AC. Same reference type as Exp N
(AdapterBase), same mesh, same TPU, same dtype, same data.

Prereq for this to run:
  MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1

That env var relaxes the validation in
`lib/levanter/src/levanter/main/train_dpo.py:372-378` which by
default forbids `AdapterBaseReferenceConfig` + LoRA +
`zero_init_b=False`. The validation encodes the canonical-DPO
assumption that `policy = reference` at init (required for DPO
loss to equal log(2) at step 0), which zero_init_b=True enforces
with AdapterBase. We relax it here because H_B is specifically
about whether breaking that symmetry recovers training at c=f32.
Revert the code change after this experiment finishes.

What this experiment does NOT test:
- Whether a Separate reference config behaves differently at c=f32.
  That would require an additional run; skipping to keep the test
  single-knob.

Decision rule (narrow):
- AD recovers (step-2 loss <= 0.5) -> conclude "c=f32 + LoRA
  AdapterBase + zero_init_b=True is broken on this stack;
  zero_init_b=False is a workaround." Do not claim to have
  identified the mechanism.
- AD stays stuck -> H_B ruled out. Move to kernel / optimizer
  probes.

Config
======

Base: debug_r64_matched_pd2_s10.py with TPU_TYPE=v5p-16 (Exp N).
Deltas from Exp N:
- mp=jmp.get_policy("p=f32,c=f32") (was default c=bf16 in Exp N)
- adapter.zero_init_b=False (was True)
- HLO dump enabled, mirroring AB/AC setup.
Reference type is unchanged: AdapterBaseReferenceConfig (as in Exp N).

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

import jmp
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

TPU_TYPE = "v5p-16"
REGIONS = ["us-central1", "us-east5"]
PER_DEVICE = 2

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or REGIONS

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "ad-v5p16-pd2-fp32-zibF")
_DBG_UPLOAD_PREFIX = os.environ.get(
    "EXPERIMENT_AD_HLO_PREFIX",
    "gs://marin-us-central1/debug/xla_hlo",
)
_DBG_UPLOAD_DIR = f"{_DBG_UPLOAD_PREFIX}/{_DBG_RUN_TAG}/"

_XLA_FLAGS = "--xla_dump_to=/tmp/xla_hlo " "--xla_dump_hlo_as_text " "--xla_dump_hlo_module_re=.*train.*"

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="400g", regions=REGIONS_FOR_TPU),
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=False, target_modules=None),
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
        "XLA_FLAGS": _XLA_FLAGS,
        "MARIN_DEBUG_HLO_UPLOAD_DIR": _DBG_UPLOAD_DIR,
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_ad_r64_v5p16_pd{PER_DEVICE}_fp32_zibF_s10_{_DBG_RUN_TAG}",
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
        "experiment-ad",
        "v5p-16",
        f"pd{PER_DEVICE}",
        "r64-alpha64",
        "p-f32-c-f32",
        "zero-init-b-false",
        "hlo-dump",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
