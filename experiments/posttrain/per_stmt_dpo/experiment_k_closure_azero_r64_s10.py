# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment K-closure: the original v5p-8 vs v6e-8 per-stmt DPO split,
rerun with ``a_init_mode="zero"`` (the L4b-AZ fix) to confirm Bug-1 is
eliminated at its origin.

The original Experiment K
(``experiments/posttrain/per_stmt_dpo/debug_r64_alpha64_s10.py``,
2026-04-14) ran this exact recipe under the standard LoRA init
(``zero_init_b=True``, ``a_init_mode="random"``) and reported:

- v5p-8 step-3 loss: 0.6796
- v6e-8 step-3 loss: 0.3254
- gap: 0.354

That was the first formal confirmation of the v5p/v6e LoRA-DPO
divergence — the experiment that kicked off the whole investigation.

This script is a byte-for-byte mirror of the original Exp K except for
the LoRA init flip:

- **Exp K** (original):      ``zero_init_b=True,  a_init_mode="random"``
- **Exp K-closure** (this):  ``zero_init_b=False, a_init_mode="zero"``

Both configurations produce ``B @ A = 0`` at initialisation (DPO
identity preserved), so the training signal is mathematically
equivalent. They differ only in which adapter matrix receives the
first non-trivial update, and through which SPMD collective class
that update flows.

Prediction under the shard-class mechanism established in
``.agents/logbooks/bug_1_dpo_lora_physical_topology.md``:

- With ``a_init_mode="zero"``, the step-0 gradient to ``B`` is zero
  (backprop through ``A=0`` nulls the chain). The first non-trivial
  gradient goes to ``A``, whose gradient path contracts on the
  FSDP-sharded axis as an *input-axis* reduction — the collective
  class empirically proved permutation-invariant across 5 independent
  ``v5p-8`` probes (L3a, L3c, L4a, and the single-module tests).
- Result: v5p-8 should no longer hit the pathological step-0
  bit-fork, and the paired v5p-8 / v6e-8 losses should track each
  other as they do in Exp N (which worked by matching ``|data|=8``
  on both pods).

Success criterion: step-9 loss gap ``|v5p − v6e| ≤ 0.05``. The
original Exp K gap was 0.354, so a 7× reduction closes the thread.

Usage:

    TPU_TYPE=v5p-8 REGIONS_OVERRIDE=us-central1 python experiment_k_closure_azero_r64_s10.py
    TPU_TYPE=v5p-8 REGIONS_OVERRIDE=us-east5    python experiment_k_closure_azero_r64_s10.py
    TPU_TYPE=v6e-8 REGIONS_OVERRIDE=europe-west4 python experiment_k_closure_azero_r64_s10.py
    TPU_TYPE=v6e-8 REGIONS_OVERRIDE=us-east5     python experiment_k_closure_azero_r64_s10.py
    TPU_TYPE=v6e-8 REGIONS_OVERRIDE=us-east1     python experiment_k_closure_azero_r64_s10.py
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.data.text import PreferenceChatLmDatasetFormat
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, mirrored
from marin.processing.tokenize import lm_data_config

STMT_TRAIN = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/train/shard-00000.jsonl.gz",
    budget_gb=1,
)
STMT_VAL = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/val/shard-00000.jsonl.gz",
    budget_gb=1,
)
FULL_VAL = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)

tokenized_train = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_train_marin_tokenizer",
    dataset=STMT_TRAIN,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)
tokenized_stmt_val = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_val_marin_tokenizer",
    dataset=STMT_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)
tokenized_full_val = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=FULL_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={
        "stmt_val": tokenized_stmt_val,
        "full_val": tokenized_full_val,
    },
)

tpu = os.environ.get("TPU_TYPE", "v6e-8")
DEFAULT_REGIONS = {
    "v5p-8": ["us-central1", "us-east5"],
    "v6e-8": ["europe-west4", "us-east5", "us-east1"],
}
_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS[tpu]
PER_DEVICE = {"v5p-8": -1, "v6e-8": 4}
PER_DEVICE_EVAL = {"v5p-8": 16, "v6e-8": 4}

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", "kclosure")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(tpu, ram="150g" if tpu.startswith("v5p") else None, regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE[tpu],
    per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
    train_batch_size=64,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(
        r=64,
        alpha=64,
        dropout=0.0,
        zero_init_b=False,
        a_init_mode="zero",
        target_modules=None,
    ),
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
        "MARIN_DEBUG_LORA_DEBUG": "1",
        "MARIN_DEBUG_SKIP_HF_EXPORT": "1",
    },
)

tpu_short = tpu.replace("-", "")
training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/k_closure_azero_r64_s10_{tpu_short}_{_DBG_RUN_TAG}",
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
        "experiment-k-closure",
        "a-init-zero",
        "bug-1-closure",
        "r64-alpha64",
        tpu,
    ],
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
