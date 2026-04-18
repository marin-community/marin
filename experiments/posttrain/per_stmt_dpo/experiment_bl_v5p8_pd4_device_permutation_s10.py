# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment BL: v5p-8 Bug-1 probe via explicit physical device permutation.

This is the canonical Bug-1 LoRA DPO recipe on `v5p-8`, but instead of letting
JAX choose a topology-aware device mesh from `jax.devices()`, we build the
canonical `{replica:1, data:4, model:1}` mesh from an explicitly permuted device
list. The logical mesh shape stays fixed; only the mapping from logical
`data[0..3]` to physical TPU chips changes.

Why this probe:

- Exp W and Exp Z3 showed that avoiding `data=4` on v5p-8 rescues Bug 1.
- Exp U / AB, Exp V, Exp R / R2a, CM, and CN ruled out the obvious non-topology
  explanations.
- The remaining high-value discriminator is whether Bug 1 depends on the
  *physical assignment* of the 4 devices, or only on the abstract logical
  `data=4` regime.

Expected outcomes:

- If different permutations materially change step-2 / step-9 loss, physical
  device placement is load-bearing and the bug is truly topology-sensitive.
- If all permutations are equivalently bad, the failure is more likely intrinsic
  to the logical `data=4` FSDP regime on this recipe.

Usage:

- `EXPERIMENT_BL_ORDER=canonical`   -> `(0,1,2,3)`
- `EXPERIMENT_BL_ORDER=reverse`     -> `(3,2,1,0)`
- `EXPERIMENT_BL_ORDER=swap12`      -> `(0,2,1,3)`
- `EXPERIMENT_BL_ORDER=rotate`      -> `(1,2,3,0)`
- `EXPERIMENT_BL_ORDER=3,1,2,0`     -> explicit permutation

Important:

- This experiment uses `preserve_device_order=True`, so its `canonical` variant
  is a new explicit-order control, not a promise of bit-identical equivalence to
  Exp Q's helper-built mesh.
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig
from levanter.utils.mesh import MeshConfig

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
DEVICE_COUNT = 4

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
            f"EXPERIMENT_BL_ORDER must name one of {sorted(ORDER_ALIASES)} or provide "
            f"{DEVICE_COUNT} comma-separated device indices, got {raw!r}"
        )

    permutation = tuple(int(piece) for piece in pieces)
    if sorted(permutation) != list(range(DEVICE_COUNT)):
        raise ValueError(f"Invalid permutation {permutation}; expected each of 0..{DEVICE_COUNT - 1} exactly once.")

    return permutation


_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

_order_raw = os.environ.get("EXPERIMENT_BL_ORDER", "canonical").strip().lower()
DEVICE_PERMUTATION = _parse_device_permutation(_order_raw)
ORDER_LABEL = _order_raw.replace(",", "-")
_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"bl-{ORDER_LABEL}")

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
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_bl_r64_v5p8_pd{PER_DEVICE}_perm_{ORDER_LABEL}_s10_{_DBG_RUN_TAG}",
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
        "experiment-bl",
        "bug-1",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        f"perm-{ORDER_LABEL}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
