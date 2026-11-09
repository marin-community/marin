# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one pinned native Hero checkpoint as BF16 split-expert HF weights."""

import gc
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import uuid4

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.compat.hf_checkpoints import DEFAULT_MAX_SHARD_SIZE, _shard_hf_checkpoint
from levanter.compat.hf_export import HFShardResume, run_on_export_writer, save_hf_shards, write_export_json
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.ops.export_metadata import split_expert_names
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.sft_runtime.model import (
    GrugModelConfig,
    Transformer,
    grugmoe_inference_state_dict,
)
from experiments.post_training.long_loop.timing import durable_event_log

MANIFEST_FILENAME = "export-manifest.json"
REQUEST_FILENAME = "export-request.json"
INDEX_FILENAME = "model.safetensors.index.json"
EXPORT_VERSION = 6


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: str
    metadata_digest: str
    model: GrugModelConfig
    destination: str
    source_revision: str
    max_position_embeddings: int
    expert_axis_size: int = 1
    replica_axis_size: int = 1


def export_weights(
    config: ExportConfig,
    restore: Callable[[str, str, GrugModelConfig, jax.sharding.Mesh], Transformer],
    *,
    weights_kind: str,
) -> None:
    """Write or resume an export; refuse any destination with a completion manifest.

    All JAX processes call this with the same config after distributed initialization.
    A destination belongs to exactly one exporting gang at a time.
    """
    if re.fullmatch(r"[0-9a-f]{40}", config.source_revision) is None:
        raise ValueError("source_revision must be a 40-character lowercase hexadecimal Marin commit")
    root = StoragePath(config.destination)
    hf_config = config.model.to_hf_config(config.model.vocab_size).to_dict()
    if config.max_position_embeddings < config.model.max_seq_len:
        raise ValueError("Export context bound is smaller than the training sequence length")
    hf_config["max_position_embeddings"] = config.max_position_embeddings
    hf_config["max_seq_len"] = config.max_position_embeddings
    request = {
        "export_version": EXPORT_VERSION,
        "weights_kind": weights_kind,
        "config": draccus.encode(config),
        "hf_config": hf_config,
    }
    export_id = digest(request)

    def prepare():
        if (root / MANIFEST_FILENAME).exists():
            raise FileExistsError(f"Export already complete: {root}")
        if not (root / REQUEST_FILENAME).exists() and root.exists() and root.ls():
            raise FileExistsError(f"Export requires a fresh destination: {root}")
        write_export_json(root / REQUEST_FILENAME, request)

    run_on_export_writer(prepare)
    mesh = compact_grug_mesh(expert_axis_size=config.expert_axis_size, replica_axis_size=config.replica_axis_size)
    with (
        jax.set_mesh(mesh),
        durable_event_log(
            str(root / "timing" / f"export-rank-{jax.process_index()}-{uuid4().hex}.jsonl"),
            config.source_revision,
            "sft_export",
            export_id,
        ) as timing,
    ):
        with timing.span("checkpoint_restore"):
            model = restore(config.checkpoint, config.metadata_digest, config.model, mesh)
        with timing.span("bf16_conversion_and_shard_plan"):
            model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model)
            jax.block_until_ready(model)
            gc.collect()
            state_dict = grugmoe_inference_state_dict(model)
            shards, _ = _shard_hf_checkpoint(state_dict, max_shard_size=DEFAULT_MAX_SHARD_SIZE)
        with timing.span("serialization_hash_and_upload"):
            records = save_hf_shards(
                shards,
                lambda keys: {name: state_dict[name] for name in keys},
                config.destination,
                # One writer stages one planned shard plus serialization buffers.
                export_host_budget_bytes=1,
                max_concurrent_shards=1,
                resume=HFShardResume(root, export_id),
                tensor_names={name: split_expert_names(name, config.model.num_experts) for name in state_dict},
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

        run_on_export_writer(finish)
