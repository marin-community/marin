# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export selected SFT parameters without changing their frozen router biases."""

import json
import logging

import draccus
import equinox as eqx
import jax
from levanter.distributed import DistributedConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug import checkpointing
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig, export_weights
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.snowball_sft.validation_checkpoint import SELECTION_KIND


def restore_selected_weights(
    checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh: jax.sharding.Mesh
) -> Transformer:
    """Restore only committed, validation-selected parameters with pinned metadata."""
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    if (
        digest(metadata) != metadata_digest
        or metadata.get("is_temporary") is not False
        or metadata.get("selection_kind") != SELECTION_KIND
    ):
        raise ValueError("Checkpoint is not the pinned permanent SFT validation selection")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    state = checkpointing.load_grug_checkpoint(
        state={"params": template}, candidate=checkpoint, mesh=mesh, allow_partial=False
    )
    return jax.block_until_ready(state["params"])


def export(config: ExportConfig) -> None:
    """Export the selected SFT weights using the shared resumable HF writer."""
    export_weights(config, restore_selected_weights, weights_kind="snowball_sft_frozen_router")


@draccus.wrap()
def main(config: ExportConfig) -> None:
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    export(config)


if __name__ == "__main__":
    main()
