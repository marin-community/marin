# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment L2 (Llama-3.1 base): MLP-only LoRA on the canonical Bug-1 recipe.

Phase A probe complementary to L1. Tests whether MLP LoRA modules
(``gate_proj``, ``up_proj``, ``down_proj``) reproduce the
canonical-vs-reverse loss split on their own. Pairs with the L1
attn-only result (gap 0.209 at step 9, 34% shrink from BL) to attribute
the residual gap.

Decision rule (vs BL-Llama3.1 reference gap 0.316):

- **MLP gap near BL gap** → MLP is also a Bug-1 driver; the bug is
  module-agnostic inside LoRA. Next: investigate the common denominator
  (sharding class, update-path, or optimizer state).
- **MLP gap << BL gap** → attention is the dominant driver; L1 residual
  must come from Q/K/V/O internals. Next: L3a Q/V narrowing.
- **partial split** → both families contribute roughly additively.

Usage:

    EXPERIMENT_L2_LLAMA31_ORDER=canonical   # (0,1,2,3) — bad under BL
    EXPERIMENT_L2_LLAMA31_ORDER=reverse     # (3,2,1,0) — good under BL
    EXPERIMENT_L2_LLAMA31_ORDER=swap12      # (0,2,1,3)
    EXPERIMENT_L2_LLAMA31_ORDER=rotate      # (1,2,3,0)
    EXPERIMENT_L2_LLAMA31_ORDER=3,1,2,0     # explicit permutation
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_dpo
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.per_stmt_dpo.base_model import llama_3_1_8b_instruct_gcs_path_for
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
DEFAULT_BS = 64
DEFAULT_STEPS = 10
PER_DEVICE = 4
DEVICE_COUNT = 4

MLP_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

ORDER_ALIASES = {
    "canonical": (0, 1, 2, 3),
    "reverse": (3, 2, 1, 0),
    "swap12": (0, 2, 1, 3),
    "rotate": (1, 2, 3, 0),
}


def _parse_device_permutation(raw: str) -> tuple[int, ...]:
    alias = ORDER_ALIASES.get(raw)
    if alias is not None:
        return alias

    pieces = [part.strip() for part in raw.split(",") if part.strip()]
    if len(pieces) != DEVICE_COUNT:
        raise ValueError(
            f"EXPERIMENT_L2_LLAMA31_ORDER must name one of {sorted(ORDER_ALIASES)} or provide "
            f"{DEVICE_COUNT} comma-separated device indices, got {raw!r}"
        )

    permutation = tuple(int(piece) for piece in pieces)
    if sorted(permutation) != list(range(DEVICE_COUNT)):
        raise ValueError(f"Invalid permutation {permutation}; expected each of 0..{DEVICE_COUNT - 1} exactly once.")

    return permutation


_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_order_raw = os.environ.get("EXPERIMENT_L2_LLAMA31_ORDER", "canonical").strip().lower()
DEVICE_PERMUTATION = _parse_device_permutation(_order_raw)
ORDER_LABEL = _order_raw.replace(",", "-")
TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_L2_LLAMA31_BS", str(DEFAULT_BS)))
NUM_TRAIN_STEPS = int(os.environ.get("EXPERIMENT_L2_LLAMA31_STEPS", str(DEFAULT_STEPS)))
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"l2l31-{ORDER_LABEL}")

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, ram="250g", regions=REGIONS_FOR_TPU),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    steps_per_eval=NUM_TRAIN_STEPS,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path=llama_3_1_8b_instruct_gcs_path_for(REGIONS_FOR_TPU),
    adapter=LoraAdaptationConfig(
        r=64,
        alpha=64,
        dropout=0.0,
        zero_init_b=True,
        target_modules=MLP_TARGET_MODULES,
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
    mesh=MeshConfig(
        axes={"replica": 1, "data": 4, "model": 1},
        param_mapping={"embed": "data"},
        shared_mapping={},
        device_permutation=DEVICE_PERMUTATION,
        preserve_device_order=True,
    ),
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
        "MARIN_DEBUG_LORA_DEBUG": "1",
        "MARIN_DEBUG_SKIP_HF_EXPORT": "1",
    },
)

training_step = default_dpo(
    name=(
        "dpo/stmt_dpo/debug/"
        f"experiment_l2l31_mlp_r64_v5p8_bs{TRAIN_BATCH_SIZE}_pd{PER_DEVICE}"
        f"_perm_{ORDER_LABEL}_s{NUM_TRAIN_STEPS}_{_DBG_RUN_TAG}"
    ),
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
        "experiment-l2-llama31",
        "mlp-only-lora",
        "llama31-8b-instruct",
        "bug-1",
        "v5p-8",
        f"bs{TRAIN_BATCH_SIZE}",
        "pd4",
        "r64-alpha64",
        f"perm-{ORDER_LABEL}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_stmt_val,
            tokenized_full_val,
            training_step,
        ]
    )
