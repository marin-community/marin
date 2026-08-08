# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the vendored Snowball training model."""

import json
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from levanter.tensorstore_serialization import DtypeSource
from rigging.filesystem import StoragePath

# The vendored experiment path is the immutable training source for Snowball.
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer
from tests.cluster.vllm.snowball import SNOWBALL, ModelIdentity


def read_executor_info(model: ModelIdentity = SNOWBALL) -> dict[str, Any]:
    return json.loads(StoragePath(model.executor_info_path).read_text())


def decode_vendored_config(executor_info: dict[str, Any]) -> VendoredGrugModelConfig:
    return draccus.decode(VendoredGrugModelConfig, executor_info["config"]["model"])


def load_checkpoint(
    config: VendoredGrugModelConfig,
    mesh: jax.sharding.Mesh,
    *,
    model: ModelIdentity = SNOWBALL,
    parameter_dtype: DTypeLike | None = None,
) -> tuple[VendoredTransformer, jax.Array]:
    template = eqx.filter_eval_shape(VendoredTransformer.init, config, key=jax.random.PRNGKey(0))
    if parameter_dtype is not None:

        def cast_dtype(value):
            dtype = getattr(value, "dtype", None)
            if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
                return value
            return jax.ShapeDtypeStruct(
                value.shape,
                parameter_dtype,
                sharding=getattr(value, "sharding", None),
                weak_type=getattr(value, "weak_type", False),
            )

        template = jax.tree.map(cast_dtype, template)
    checkpoint_state = load_levanter_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        model.checkpoint_path,
        mesh=mesh,
        dtype_source=DtypeSource.EXEMPLAR if parameter_dtype is not None else DtypeSource.CHECKPOINT,
    )
    jax.block_until_ready(checkpoint_state)
    return checkpoint_state["params"], checkpoint_state["pending_qb_betas"]


def apply_pending_qb_betas(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    assert model.stacked_blocks is not None
    # Mirrors train._apply_qb_betas without importing the training entrypoint.
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)


def prepare_bf16_parameters(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    """Apply the FP32 pending router update before casting inference parameters."""
    assert pending_qb_betas.dtype == jnp.float32
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        model,
    )
    jax.block_until_ready(model)
    return model
