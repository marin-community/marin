# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment L4b-AZ (A=zero variant): Down-only LoRA with A=0, B=Gaussian.

Direct test of the step-0-damping hypothesis from the
``lora_debug/*`` direction audit. Keeps everything identical to L4b
(the most pathological Bug-1 probe, gap 0.373 at step 9) except the
LoRA init mode:

- **L4b**   (reference): ``zero_init_b=True``,  ``a_init_mode="random"``
- **L4b-AZ** (this run):  ``zero_init_b=False``, ``a_init_mode="zero"``

Both configurations produce ``B @ A = 0`` at initialization, preserving
DPO's policy=reference identity. They differ in *which* adapter matrix
carries the random init and *which* accumulates the first non-trivial
update:

- L4b step 0: ``B = 0`` (zero-init) → first update is ``LR * grad_B``
  applied to zero. Grad_B flows through the pathological output-axis
  collective on ``embed``. Step-0 bit-fork between canonical/reverse
  becomes the entire ``B`` going forward → chaotic amplification.
- L4b-AZ step 0: ``A = 0`` (zero-init) → ``grad_B = 0`` (backprop
  through ``A=0`` nulls the B gradient). First non-zero gradient goes
  to ``A``, whose gradient path is an *input-axis* reduction on
  embed — the class we empirically proved permutation-invariant
  (L3a/L3c/L4a all showed gap ≤0.006). At step 1+, grad_B is non-zero
  and lands as a small perturbation on an already-large random ``B``.

Prediction: if the step-0 B-zero damping is the sole load-bearing
variable, L4b-AZ should have gap ~0 (matching the input-sharded L*
probes). If the gap remains near L4b's 0.373, the pathology is
deeper than step-0 damping and the output-axis collective fork is
tripping Adam even with existing B magnitude.

Usage:

    EXPERIMENT_L4BAZ_LLAMA31_ORDER=canonical   # (0,1,2,3)
    EXPERIMENT_L4BAZ_LLAMA31_ORDER=reverse     # (3,2,1,0)
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

DOWN_TARGET_MODULES = ["down_proj"]

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
            f"EXPERIMENT_L4BAZ_LLAMA31_ORDER must name one of {sorted(ORDER_ALIASES)} or provide "
            f"{DEVICE_COUNT} comma-separated device indices, got {raw!r}"
        )
    permutation = tuple(int(piece) for piece in pieces)
    if sorted(permutation) != list(range(DEVICE_COUNT)):
        raise ValueError(f"Invalid permutation {permutation}; expected each of 0..{DEVICE_COUNT - 1} exactly once.")
    return permutation


_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_order_raw = os.environ.get("EXPERIMENT_L4BAZ_LLAMA31_ORDER", "canonical").strip().lower()
DEVICE_PERMUTATION = _parse_device_permutation(_order_raw)
ORDER_LABEL = _order_raw.replace(",", "-")
TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_L4BAZ_LLAMA31_BS", str(DEFAULT_BS)))
NUM_TRAIN_STEPS = int(os.environ.get("EXPERIMENT_L4BAZ_LLAMA31_STEPS", str(DEFAULT_STEPS)))
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"l4bazl31-{ORDER_LABEL}")

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
        r=64, alpha=64, dropout=0.0,
        zero_init_b=False,
        a_init_mode="zero",
        target_modules=DOWN_TARGET_MODULES,
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
        f"experiment_l4bazl31_dn_az_r64_v5p8_bs{TRAIN_BATCH_SIZE}_pd{PER_DEVICE}"
        f"_perm_{ORDER_LABEL}_s{NUM_TRAIN_STEPS}_{_DBG_RUN_TAG}"
    ),
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    tags=[
        "dpo", "lora-dpo", "bloom", "per-stmt", "support-mental-health",
        "debug-accum", "experiment-l4b-az", "down-only-lora", "a-init-zero",
        "llama31-8b-instruct", "bug-1", "v5p-8",
        f"bs{TRAIN_BATCH_SIZE}", "pd4", "r64-alpha64", f"perm-{ORDER_LABEL}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
