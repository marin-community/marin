# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Hero weights for inference consumers."""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorstore as ts
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.tensorstore_serialization import build_kvstore_spec
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.checkpointing import LEGACY_STATE_KEY, MASTER_PARAMS_KEY
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, apply_qb_betas

logger = logging.getLogger(__name__)
PENDING_QB_BETAS_KEY = "pending_qb_betas"


@dataclass(frozen=True)
class RestoredWeights:
    model: Transformer
    weights_key: str


def metadata_hash(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def restore_weights(
    checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh: jax.sharding.Mesh
) -> RestoredWeights:
    """Restore authoritative weights and pending router bias, with no optimizer or fallback checkpoint."""
    logger.info("Validate checkpoint metadata and weight layout: %s", checkpoint)
    checkpoint_path = StoragePath(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
    if metadata_hash(metadata) != metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Checkpoint metadata changed or checkpoint is not permanent")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    manifest = read_manifest(checkpoint)
    if manifest is not None:
        wrapped = any(path.startswith(f"{LEGACY_STATE_KEY}/") for path in manifest.array_paths)
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = any(path.startswith(f"{prefix}{MASTER_PARAMS_KEY}/") for path in manifest.array_paths)
    elif (checkpoint_path / "manifest.ocdbt").exists():
        # Probe known metadata keys inside the database. Filesystem directories cannot reveal this layout.
        kvstore = ts.KvStore.open({"driver": "ocdbt", "base": build_kvstore_spec(checkpoint)}).result()
        wrapped = kvstore.read(f"{LEGACY_STATE_KEY}/{PENDING_QB_BETAS_KEY}/zarr.json").result().state == "value"
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = kvstore.read(f"{prefix}{MASTER_PARAMS_KEY}/token_embed/zarr.json").result().state == "value"
    else:
        # Pre-manifest checkpoints use directory-backed arrays. Probe layout directories only.
        wrapped = (checkpoint_path / LEGACY_STATE_KEY).exists()
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = (checkpoint_path / f"{prefix}{MASTER_PARAMS_KEY}").exists()
    weights_key = MASTER_PARAMS_KEY if master else "params"
    logger.info("Restore checkpoint arrays: weights=%s, wrapped=%s", weights_key, wrapped)
    state_template: dict[str, Transformer | jax.ShapeDtypeStruct] = {
        weights_key: template,
        PENDING_QB_BETAS_KEY: jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
    }
    state = load_checkpoint(
        {LEGACY_STATE_KEY: state_template} if wrapped else state_template,
        checkpoint,
        mesh=mesh,
        allow_partial=False,
    )
    if wrapped:
        state = state[LEGACY_STATE_KEY]
    jax.block_until_ready(state)
    logger.info("Checkpoint arrays ready; apply pending router bias")
    pending = cast(jax.Array, state[PENDING_QB_BETAS_KEY])
    return RestoredWeights(apply_qb_betas(cast(Transformer, state[weights_key]), pending), weights_key)
