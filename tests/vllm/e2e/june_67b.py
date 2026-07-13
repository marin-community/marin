# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint locations and loading helpers for the vendored June 67B model."""

import json
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from rigging.filesystem import StoragePath

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer

from .reference import CHECKPOINT_NAME, RUN_NAME

RUN_ROOT = f"s3://marin-us-east-02a/marin/grug/{RUN_NAME}"
EXECUTOR_INFO_PATH = f"{RUN_ROOT}/.executor_info"
CHECKPOINT_PATH = f"{RUN_ROOT}/checkpoints/{CHECKPOINT_NAME}"
GCS_RUN_ROOT = f"gs://marin-us-east5/grug/{RUN_NAME}"
GCS_EXECUTOR_INFO_PATH = f"{GCS_RUN_ROOT}/.executor_info"
GCS_CHECKPOINT_PATH = f"{GCS_RUN_ROOT}/checkpoints/{CHECKPOINT_NAME}"


def read_executor_info(path: str = EXECUTOR_INFO_PATH) -> dict[str, Any]:
    return json.loads(StoragePath(path).read_text())


def decode_vendored_config(executor_info: dict[str, Any]) -> VendoredGrugModelConfig:
    return draccus.decode(VendoredGrugModelConfig, executor_info["config"]["model"])


def load_checkpoint(
    config: VendoredGrugModelConfig,
    mesh: jax.sharding.Mesh,
    checkpoint_path: str = CHECKPOINT_PATH,
) -> tuple[VendoredTransformer, jax.Array]:
    template = eqx.filter_eval_shape(VendoredTransformer.init, config, key=jax.random.PRNGKey(0))
    checkpoint_state = load_levanter_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        checkpoint_path,
        mesh=mesh,
    )
    jax.block_until_ready(checkpoint_state)
    return checkpoint_state["params"], checkpoint_state["pending_qb_betas"]


def apply_pending_qb_betas(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    assert model.stacked_blocks is not None
    # Mirrors train._apply_qb_betas without importing the training entrypoint.
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)
