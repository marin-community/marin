# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one pinned native Hero checkpoint as BF16 split-expert HF weights."""

import gc
import logging
from dataclasses import dataclass

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.compat.hf_checkpoints import DEFAULT_MAX_SHARD_SIZE, _shard_hf_checkpoint
from levanter.compat.hf_export import HFShardResume, run_on_export_writer, save_hf_shards
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import GrugModelConfig, grugmoe_inference_state_dict
from experiments.grug.moe_hero_ep.ops.export_metadata import (
    EXPORT_VERSION,
    artifact_config,
    bf16_hf_config,
    execution_config,
    prepare_export,
    split_expert_names,
    write_export_metadata,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.weights import restore_weights

DEFAULT_EXPORT_HOST_BUDGET_BYTES = 32 * 1024**3


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: str
    metadata_digest: str
    model: GrugModelConfig
    destination: str
    source_revision: str
    expert_axis_size: int = 1
    replica_axis_size: int = 1
    export_host_budget_bytes: int = DEFAULT_EXPORT_HOST_BUDGET_BYTES
    max_concurrent_shards: int = 2


def export(config: ExportConfig) -> None:
    """Write or resume an export; refuse any destination with a completion manifest.

    All JAX processes call this with the same config after distributed initialization.
    A destination belongs to exactly one exporting gang at a time.
    """
    hf_config = bf16_hf_config(config.model)
    encoded = draccus.encode(config)
    request = {
        "export_version": EXPORT_VERSION,
        "implementation": "device",
        "config": artifact_config(encoded),
        "hf_config": hf_config,
    }
    export_id = digest(request)

    run_on_export_writer(lambda: prepare_export(config.destination, request))
    resume = HFShardResume(StoragePath(config.destination), export_id)
    mesh = compact_grug_mesh(expert_axis_size=config.expert_axis_size, replica_axis_size=config.replica_axis_size)
    with jax.set_mesh(mesh):
        model = restore_weights(config.checkpoint, config.metadata_digest, config.model, mesh)
        # restore_weights already applied the pending QB update to authoritative weights.
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model)
        jax.block_until_ready(model)
        gc.collect()
        state_dict = grugmoe_inference_state_dict(model)
        shards, _ = _shard_hf_checkpoint(state_dict, max_shard_size=DEFAULT_MAX_SHARD_SIZE)
        tensor_names = {name: split_expert_names(name, config.model.num_experts) for name in state_dict}
        records = save_hf_shards(
            shards,
            lambda keys: {name: state_dict[name] for name in keys},
            config.destination,
            export_host_budget_bytes=config.export_host_budget_bytes,
            max_concurrent_shards=config.max_concurrent_shards,
            resume=resume,
            tensor_names=tensor_names,
        )
        payload_size = sum(x.size * x.dtype.itemsize for x in state_dict.values())
        run_on_export_writer(
            lambda: write_export_metadata(
                resume,
                request,
                records,
                payload_size,
                {
                    "executor": "device",
                    "execution": execution_config(encoded),
                    "mesh": dict(mesh.shape),
                    "process_count": jax.process_count(),
                },
            )
        )


@draccus.wrap()
def main(config: ExportConfig) -> None:
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    export(config)


if __name__ == "__main__":
    main()
