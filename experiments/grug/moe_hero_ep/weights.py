# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Hero weights for inference consumers."""

import json
import logging

import equinox as eqx
import jax
import tensorstore as ts
from levanter.tensorstore_serialization import build_kvstore_spec
from rigging.filesystem.storage_path import StoragePath

from experiments.grug import checkpointing
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest

logger = logging.getLogger(__name__)


def restore_weights(
    checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh: jax.sharding.Mesh
) -> Transformer:
    """Restore authoritative weights with their stored router bias, matching training-time evaluation.

    Pending QB betas belong to the next resumed training update, not inference.
    """
    logger.info("Validate checkpoint metadata and weight layout: %s", checkpoint)
    checkpoint_path = StoragePath(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
    if digest(metadata) != metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Checkpoint metadata changed or checkpoint is not permanent")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    try:
        master = checkpointing.checkpoint_stores_master(checkpoint)
    except FileNotFoundError:
        # Inference also accepts checkpoints written before manifests existed.
        markers = (
            checkpointing.MASTER_PARAMS_KEY,
            f"{checkpointing.LEGACY_STATE_KEY}/{checkpointing.MASTER_PARAMS_KEY}",
        )
        if (checkpoint_path / "manifest.ocdbt").exists():
            kvstore = ts.KvStore.open({"driver": "ocdbt", "base": build_kvstore_spec(checkpoint)}).result()
            master = any(kvstore.read(f"{marker}/token_embed/zarr.json").result().state == "value" for marker in markers)
        else:
            master = any((checkpoint_path / marker).exists() for marker in markers)
    weights_key = checkpointing.MASTER_PARAMS_KEY if master else "params"
    logger.info("Restore checkpoint arrays: weights=%s", weights_key)
    state_template = {weights_key: template}
    state = checkpointing.load_grug_checkpoint(
        state=state_template,
        candidate=checkpoint,
        mesh=mesh,
        allow_partial=False,
    )
    jax.block_until_ready(state)
    return state[weights_key]
