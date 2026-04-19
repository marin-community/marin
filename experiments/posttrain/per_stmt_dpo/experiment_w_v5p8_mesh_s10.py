# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment W: v5p-8 LoRA DPO with alternative mesh axes.

Identical to Experiment Q pd=4 except the mesh axes are overridden to test
the "LoRA FSDP sharding on 4-device v5p-8 mesh is the load-bearing bug"
hypothesis. Three variants selectable via ``EXPERIMENT_W_MESH``:

- ``EXPERIMENT_W_MESH=tp`` (default): pure tensor parallel on the 4 chips
  — ``axes = {replica:1, data:1, model:4}``, TP-shard MLP/heads along
  ``model``. Zero FSDP on the base weights and zero FSDP on LoRA params.
- ``EXPERIMENT_W_MESH=mix``: 2x2 mix — ``axes = {replica:1, data:2, model:2}``,
  TP-shard MLP/heads along ``model`` and FSDP-shard the data axis across
  2 replicas.
- ``EXPERIMENT_W_MESH=fsdp``: reference run with the canonical bad config
  — ``axes = {replica:1, data:4, model:1}``. Useful as a sanity in-script
  comparison to Exp Q.

Why this experiment:

- Exp V ruled out ``AdapterBaseReferenceConfig`` as the cause of the v5p-8
  LoRA pathology.
- Exp U ruled out all numeric-precision theories.
- The leading remaining hypothesis is that the 4-device FSDP sharding
  layout interacts badly with the LoRA update path on v5p-8.
- Mesh-axis rearrangement directly probes that: pure-TP eliminates FSDP
  entirely, and 2x2 reduces FSDP span from 4 to 2.

Expected outcomes:

- ``tp`` recovers → FSDP-on-4 is the bug. The LoRA update geometry on a
  fully-FSDP 4-chip mesh is the problem.
- ``tp`` still bad AND ``mix`` recovers → something about the
  specifically-4-wide FSDP is the bug (not FSDP per se).
- Both ``tp`` and ``mix`` bad → the issue is deeper than axis
  arrangement; move to per-module gradient probes.

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

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

MESH_VARIANT = os.environ.get("EXPERIMENT_W_MESH", "tp").lower()

# Maps mesh variant -> (ici_axes, param_mapping, shared_mapping)
# For pure-TP and 2x2 we also TP-shard MLP hidden and attention heads along
# the model axis (the Marin-standard TP placement).
_TP_SHARED_MAPPING = {"mlp": "model", "heads": "model"}

if MESH_VARIANT == "tp":
    # Pure-TP: data axis is size 1 so `param_mapping={"embed":"data"}` is a
    # no-op (embed replicated). We must NOT map `embed` to `model` because Q/K/V
    # projections have both `embed` and `heads` named axes, and `shared_mapping`
    # already puts `heads` on `model` — mapping `embed` to `model` too would
    # produce a PartitionSpec with duplicate `model` entries (DuplicateSpecError).
    MESH_AXES = {"replica": 1, "data": 1, "model": 4}
    PARAM_MAPPING = {"embed": "data"}
    SHARED_MAPPING = _TP_SHARED_MAPPING
elif MESH_VARIANT == "mix":
    MESH_AXES = {"replica": 1, "data": 2, "model": 2}
    PARAM_MAPPING = {"embed": "data"}
    SHARED_MAPPING = _TP_SHARED_MAPPING
elif MESH_VARIANT == "fsdp":
    MESH_AXES = {"replica": 1, "data": 4, "model": 1}
    PARAM_MAPPING = {"embed": "data"}
    SHARED_MAPPING = {}
else:
    raise ValueError(f"EXPERIMENT_W_MESH must be one of tp|mix|fsdp, got {MESH_VARIANT}")

_DBG_RUN_TAG = os.environ.get("MARIN_DEBUG_RUN_TAG", f"w-{MESH_VARIANT}")

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
        axes=MESH_AXES,
        param_mapping=PARAM_MAPPING,
        shared_mapping=SHARED_MAPPING,
    ),
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
    },
)

training_step = default_dpo(
    name=f"dpo/stmt_dpo/debug/experiment_w_r64_v5p8_pd{PER_DEVICE}_mesh_{MESH_VARIANT}_s10_{_DBG_RUN_TAG}",
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
        "experiment-w",
        "v5p-8",
        "pd4",
        "r64-alpha64",
        f"mesh-{MESH_VARIANT}",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])
