# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one pinned native Hero checkpoint as BF16 split-expert HF weights."""

import gc
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.compat.hf_export import HFShardProgress, on_export_writer, save_hf_shards, write_export_json
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import GrugModelConfig, grugmoe_inference_state_dict
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.weights import restore_weights

MANIFEST_FILENAME = "export-manifest.json"
REQUEST_FILENAME = "export-request.json"
INDEX_FILENAME = "model.safetensors.index.json"
EXPORT_VERSION = 1
EXPERT_BANK = re.compile(r"^(.*\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$")


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: str
    metadata_digest: str
    model: GrugModelConfig
    destination: str
    source_revision: str
    expert_axis_size: int = 1
    replica_axis_size: int = 1


def _split_names(name: str, num_experts: int) -> list[str]:
    match = EXPERT_BANK.fullmatch(name)
    if match is None:
        return [name]
    return [f"{match[1]}.{i}.{match[2]}.weight" for i in range(num_experts)]


def export(config: ExportConfig) -> None:
    """Write or resume an export; refuse any destination with a completion manifest.

    All JAX processes call this with the same config after distributed initialization.
    A destination belongs to exactly one exporting gang at a time.
    """
    if re.fullmatch(r"[0-9a-f]{40}", config.source_revision) is None:
        raise ValueError("source_revision must be a 40-character lowercase hexadecimal Marin commit")
    root = StoragePath(config.destination)
    hf_config = config.model.to_hf_config(config.model.vocab_size).to_dict()
    request = {"export_version": EXPORT_VERSION, "config": draccus.encode(config), "hf_config": hf_config}
    export_id = digest(request)

    def prepare():
        if (root / MANIFEST_FILENAME).exists():
            raise FileExistsError(f"Export already complete: {root}")
        if not (root / REQUEST_FILENAME).exists() and root.exists() and root.ls():
            raise FileExistsError(f"Export requires a fresh destination: {root}")
        write_export_json(root / REQUEST_FILENAME, request)

    on_export_writer(prepare)
    mesh = compact_grug_mesh(expert_axis_size=config.expert_axis_size, replica_axis_size=config.replica_axis_size)
    with jax.set_mesh(mesh):
        model = restore_weights(config.checkpoint, config.metadata_digest, config.model, mesh)
        # restore_weights already applied the pending QB update to authoritative weights.
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model)
        jax.block_until_ready(model)
        gc.collect()
        state_dict = grugmoe_inference_state_dict(model)
        groups: dict[str, list[str]] = {}
        for name in state_dict:
            match = re.match(r"^model\.layers\.(\d+)\.", name)
            group = f"layer-{int(match[1]):03d}" if match else "global"
            groups.setdefault(group, []).append(name)

        shards = {
            f"model-{group}.safetensors": {name: state_dict[name] for name in names} for group, names in groups.items()
        }
        tensor_names = {name: tuple(_split_names(name, config.model.num_experts)) for name in state_dict}
        records = save_hf_shards(
            shards,
            lambda keys: {name: state_dict[name] for name in keys},
            config.destination,
            # One writer and an oversized-shard budget stage only one layer plus serialization buffers.
            export_host_budget_bytes=1,
            max_concurrent_shards=1,
            progress=HFShardProgress(root, export_id),
            tensor_names=tensor_names,
        )
        payload_size = sum(x.size * x.dtype.itemsize for x in state_dict.values())
        weight_map = {name: record.filename for record in records for name in record.tensor_names}

        def finish():
            write_export_json(root / "config.json", hf_config)
            write_export_json(
                root / INDEX_FILENAME, {"metadata": {"total_size": payload_size}, "weight_map": weight_map}
            )
            write_export_json(
                root / MANIFEST_FILENAME,
                {
                    "export_id": export_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "request": request,
                    "shards": [asdict(record) for record in records],
                    "mesh": dict(mesh.shape),
                    "process_count": jax.process_count(),
                },
            )

        on_export_writer(finish)


@draccus.wrap()
def main(config: ExportConfig) -> None:
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    export(config)


if __name__ == "__main__":
    main()
