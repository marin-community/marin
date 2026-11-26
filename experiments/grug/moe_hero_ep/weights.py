# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Hero weights for inference consumers."""

import json
import logging
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorstore as ts
from levanter.checkpoint_manifest import read_manifest
from levanter.tensorstore_serialization import build_kvstore_spec
from rigging.filesystem.storage_path import StoragePath

from experiments.grug import checkpointing
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest

logger = logging.getLogger(__name__)
PENDING_QB_BETAS_KEY = "pending_qb_betas"


def validate_checkpoint_metadata(checkpoint: str, metadata_digest: str) -> None:
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    if digest(metadata) != metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Checkpoint metadata changed or checkpoint is not permanent")


def checkpoint_weight_path(checkpoint: str) -> str:
    """Find authoritative parameters, preferring FP32 masters over compute copies."""
    candidates = (
        checkpointing.MASTER_PARAMS_KEY,
        f"{checkpointing.LEGACY_STATE_KEY}/{checkpointing.MASTER_PARAMS_KEY}",
        "params",
        f"{checkpointing.LEGACY_STATE_KEY}/params",
    )
    manifest = read_manifest(checkpoint)
    if manifest is not None:
        for candidate in candidates:
            if any(path.startswith(candidate + "/") for path in manifest.array_paths):
                return candidate
        raise FileNotFoundError(f"Checkpoint has no model parameters: {checkpoint}")
    root = StoragePath(checkpoint)
    if (root / "manifest.ocdbt").exists():
        kvstore = ts.KvStore.open({"driver": "ocdbt", "base": build_kvstore_spec(checkpoint)}).result()
        for candidate in candidates:
            if kvstore.read(f"{candidate}/token_embed/zarr.json").result().state == "value":
                return candidate
    else:
        for candidate in candidates:
            if (root / candidate).exists():
                return candidate
    raise FileNotFoundError(f"Checkpoint has no model parameters: {checkpoint}")


def restore_weights(
    checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh: jax.sharding.Mesh
) -> Transformer:
    """Restore authoritative weights and pending router bias, with no optimizer or fallback checkpoint."""
    logger.info("Validate checkpoint metadata and weight layout: %s", checkpoint)
    validate_checkpoint_metadata(checkpoint, metadata_digest)
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    weights_key = checkpoint_weight_path(checkpoint).rsplit("/", 1)[-1]
    logger.info("Restore checkpoint arrays: weights=%s", weights_key)
    state_template: dict[str, Transformer | jax.ShapeDtypeStruct] = {
        weights_key: template,
        PENDING_QB_BETAS_KEY: jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
    }
    state = checkpointing.load_grug_checkpoint(
        state=state_template,
        candidate=checkpoint,
        mesh=mesh,
        allow_partial=False,
    )
    jax.block_until_ready(state)
    logger.info("Checkpoint arrays ready; apply pending router bias")
    pending = cast(jax.Array, state[PENDING_QB_BETAS_KEY])
    return apply_qb_betas(cast(Transformer, state[weights_key]), pending)
