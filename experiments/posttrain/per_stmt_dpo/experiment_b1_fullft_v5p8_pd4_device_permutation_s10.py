# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bug-1 full-FT permutation discriminator on v5p-8.

This is the full-fine-tuning analog of experiment BL. It keeps the logical mesh
fixed at `{replica:1, data:4, model:1}` on `v5p-8`, but explicitly permutes the
physical device assignment used to build that mesh.

Goal:

- determine whether the strong permutation sensitivity we saw for LoRA DPO is
  also present for full fine-tuning
- collect comparable HLO dumps for the dense path

If full FT is largely permutation-invariant while LoRA is not, that sharply
strengthens the case that LoRA's update geometry is what makes Bug 1
pathological. If full FT also splits, the issue is broader than LoRA.

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

import dataclasses
import os

from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import SeparateReferenceConfig
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
DEFAULT_REGIONS = ["us-central1"]
DEFAULT_BS = 32
DEFAULT_PD = 4
DEFAULT_STEPS = 10
DEFAULT_CHECKPOINTING = "default"
ALLOWED_PD = {2, 4}
ALLOWED_CHECKPOINTING = {"default", "offload", "recompute"}
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
            f"EXPERIMENT_B1_ORDER must name one of {sorted(ORDER_ALIASES)} or provide "
            f"{DEVICE_COUNT} comma-separated device indices, got {raw!r}"
        )

    permutation = tuple(int(piece) for piece in pieces)
    if sorted(permutation) != list(range(DEVICE_COUNT)):
        raise ValueError(f"Invalid permutation {permutation}; expected each of 0..{DEVICE_COUNT - 1} exactly once.")

    return permutation


_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_B1_BS", str(DEFAULT_BS)))
PER_DEVICE = int(os.environ.get("EXPERIMENT_B1_PD", str(DEFAULT_PD)))
if PER_DEVICE not in ALLOWED_PD:
    raise ValueError(
        f"Experiment B1 full FT only supports per_device_parallelism in {sorted(ALLOWED_PD)}, got {PER_DEVICE}"
    )

NUM_TRAIN_STEPS = int(os.environ.get("EXPERIMENT_B1_STEPS", str(DEFAULT_STEPS)))
CHECKPOINTING_POLICY = os.environ.get("EXPERIMENT_B1_CHECKPOINTING", DEFAULT_CHECKPOINTING)
if CHECKPOINTING_POLICY not in ALLOWED_CHECKPOINTING:
    raise ValueError(
        "Experiment B1 full FT only supports checkpointing in "
        f"{sorted(ALLOWED_CHECKPOINTING)}, got {CHECKPOINTING_POLICY}"
    )

_order_raw = os.environ.get("EXPERIMENT_B1_ORDER", "canonical").strip().lower()
DEVICE_PERMUTATION = _parse_device_permutation(_order_raw)
ORDER_LABEL = _order_raw.replace(",", "-")

if CHECKPOINTING_POLICY == "default":
    model_config = llama_8b
else:
    model_config = dataclasses.replace(llama_8b, gradient_checkpointing=CHECKPOINTING_POLICY)

_DBG_RUN_TAG = os.environ.get(
    "MARIN_DEBUG_RUN_TAG",
    f"b1-fullft-{ORDER_LABEL}-bs{TRAIN_BATCH_SIZE}-pd{PER_DEVICE}-{CHECKPOINTING_POLICY}",
)

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
    model_name_or_path="marin-community/marin-8b-instruct",
    reference=SeparateReferenceConfig(),
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
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
        "MARIN_DEBUG_DUMP_SHARDING": "1",
        "MARIN_DEBUG_DUMP_GRAD_VALUES": "1",
    },
)

training_step = default_dpo(
    name=(
        "dpo/stmt_dpo/debug/"
        f"experiment_b1_fullft_v5p8_pd{PER_DEVICE}_perm_{ORDER_LABEL}_s{NUM_TRAIN_STEPS}_{_DBG_RUN_TAG}"
    ),
    tokenized=tokenized_preferences,
    model_config=model_config,
    dpo_config=config,
    tags=[
        "dpo",
        "full-dpo",
        "bloom",
        "per-stmt",
        "support-mental-health",
        "debug-accum",
        "bug-1",
        "experiment-b1-fullft",
        "v5p-8",
        "full-ft",
        f"bs{TRAIN_BATCH_SIZE}",
        f"pd{PER_DEVICE}",
        f"perm-{ORDER_LABEL}",
        f"ckpt-{CHECKPOINTING_POLICY}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
