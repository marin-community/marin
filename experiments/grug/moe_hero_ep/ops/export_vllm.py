# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one pinned native Hero checkpoint as BF16 split-expert HF weights."""

import gc
import hashlib
import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from safetensors.numpy import save_file

from experiments.grug.moe_hero_ep.model import GrugModelConfig, grugmoe_inference_state_dict
from experiments.grug.moe_hero_ep.weights import metadata_hash, restore_weights

logger = logging.getLogger(__name__)
MANIFEST_FILENAME = "export-manifest.json"
REQUEST_FILENAME = "export-request.json"
INDEX_FILENAME = "model.safetensors.index.json"
EXPORT_VERSION = 1
COPY_BLOCK_BYTES = 32 * 1024 * 1024
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


def _sha256(path: StoragePath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(COPY_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: StoragePath, value: dict) -> None:
    contents = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = conditional_object(str(path))
    existing = target.read()
    if existing is not None:
        if existing.data != contents:
            raise FileExistsError(f"Refusing to overwrite {path}")
        return
    target.write(contents, expected_version=None)


def _writer_step[T](action: Callable[[], T]) -> T | None:
    # Only the main thread enters collectives. A writer I/O error reaches every rank
    # before any rank can enter the next gather or skip a completed shard.
    error = None
    result = None
    if jax.process_index() == 0:
        try:
            result = action()
        except Exception as exc:
            error = exc
    success = multihost_utils.broadcast_one_to_all(np.asarray(error is None))
    if not success:
        if error is not None:
            raise error
        raise RuntimeError("Export I/O failed on process zero")
    return result


def _split_experts(name: str, value: np.ndarray) -> dict[str, np.ndarray]:
    match = EXPERT_BANK.fullmatch(name)
    if match is None:
        return {name: value}
    if value.ndim != 3:
        raise ValueError(f"Expected a 3D routed expert bank: {name} {value.shape}")
    return {f"{match[1]}.{i}.{match[2]}.weight": value[i] for i in range(value.shape[0])}


def _completed_shard(root: StoragePath, group: str, export_id: str, names: list[str]) -> dict | None:
    progress = root / f".export-progress-{group}.json"
    if not progress.exists():
        return None
    record = json.loads(progress.read_text())
    filename = f"model-{group}.safetensors"
    if record["export_id"] != export_id or record["filename"] != filename or record["tensor_names"] != names:
        raise ValueError(f"Shard identity changed: {progress}")
    shard = root / filename
    if not shard.exists() or shard.size() != record["bytes"] or _sha256(shard) != record["sha256"]:
        raise ValueError(f"Shard integrity check failed: {shard}")
    return record


def _store_shard(
    root: StoragePath, group: str, export_id: str, names: list[str], tensors: dict[str, np.ndarray], local_root: Path
) -> dict:
    if sorted(tensors) != names:
        raise ValueError(f"Incomplete tensor mapping for {group}")
    filename = f"model-{group}.safetensors"
    local = local_root / filename
    save_file(tensors, local, metadata={"format": "pt"})
    size, checksum = local.stat().st_size, _sha256(StoragePath(str(local)))
    target = root / filename
    # An upload without progress is uncommitted and may be rewritten after interruption.
    # Committed shards were already verified by _completed_shard.
    with local.open("rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=COPY_BLOCK_BYTES)
    record = {"export_id": export_id, "filename": filename, "bytes": size, "sha256": checksum, "tensor_names": names}
    _write_json(root / f".export-progress-{group}.json", record)
    local.unlink()
    return record


def export(config: ExportConfig) -> None:
    """Write or resume an export; refuse any destination with a completion manifest.

    All JAX processes call this with the same config after distributed initialization.
    Host staging holds one layer (or the global tensors) plus serialization buffers.
    A destination belongs to exactly one exporting gang at a time.
    """
    root = StoragePath(config.destination)
    hf_config = config.model.to_hf_config(config.model.vocab_size).to_dict()
    request = {"export_version": EXPORT_VERSION, "config": draccus.encode(config), "hf_config": hf_config}
    export_id = metadata_hash(request)

    def prepare():
        if (root / MANIFEST_FILENAME).exists():
            raise FileExistsError(f"Export already complete: {root}")
        if not (root / REQUEST_FILENAME).exists() and root.exists() and root.ls():
            raise FileExistsError(f"Export requires a fresh destination: {root}")
        _write_json(root / REQUEST_FILENAME, request)

    _writer_step(prepare)
    mesh = compact_grug_mesh(expert_axis_size=config.expert_axis_size, replica_axis_size=config.replica_axis_size)
    with jax.set_mesh(mesh):
        restored = restore_weights(config.checkpoint, config.metadata_digest, config.model, mesh)
        weights_key = restored.weights_key
        authoritative_dtypes = sorted({str(x.dtype) for x in jax.tree.leaves(restored.model) if eqx.is_inexact_array(x)})
        pending = jax.sharding.reshard(restored.pending_qb_betas, P())
        jax.block_until_ready(pending)
        pending_hash = hashlib.sha256(np.asarray(pending).tobytes()).hexdigest() if jax.process_index() == 0 else None
        # restore_weights already applied the pending QB update to authoritative weights.
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, restored.model)
        jax.block_until_ready(model)
        del restored, pending
        gc.collect()
        state_dict = grugmoe_inference_state_dict(model)
        groups: dict[str, list[str]] = {}
        for name in state_dict:
            match = re.match(r"^model\.layers\.(\d+)\.", name)
            group = f"layer-{int(match[1]):03d}" if match else "global"
            groups.setdefault(group, []).append(name)

        weight_map: dict[str, str] = {}
        records: list[dict] = []
        payload_size = sum(x.size * x.dtype.itemsize for x in state_dict.values())
        with TemporaryDirectory(prefix="hero-export-") as directory:
            for group, source_names in groups.items():
                expected_names = []
                for name in source_names:
                    match = EXPERT_BANK.fullmatch(name)
                    expected_names.extend(
                        [f"{match[1]}.{i}.{match[2]}.weight" for i in range(config.model.num_experts)]
                        if match
                        else [name]
                    )
                expected_names.sort()
                completed = _writer_step(partial(_completed_shard, root, group, export_id, expected_names))
                reuse = multihost_utils.broadcast_one_to_all(np.asarray(completed is not None))
                if not reuse:
                    tensors: dict[str, np.ndarray] = {}
                    for name in source_names:
                        replicated = jax.sharding.reshard(state_dict[name], P())
                        jax.block_until_ready(replicated)
                        if jax.process_index() == 0:
                            tensors.update(_split_experts(name, np.ascontiguousarray(np.asarray(replicated))))
                        del replicated
                        multihost_utils.sync_global_devices(f"export-{group}-{name}")

                    completed = _writer_step(
                        partial(_store_shard, root, group, export_id, expected_names, tensors, Path(directory))
                    )
                    del tensors
                    gc.collect()
                if completed is not None:
                    records.append(completed)
                    weight_map.update({name: completed["filename"] for name in expected_names})
                    logger.info("%s %s", "Verified" if reuse else "Uploaded", completed["filename"])

        def finish():
            _write_json(root / "config.json", hf_config)
            _write_json(root / INDEX_FILENAME, {"metadata": {"total_size": payload_size}, "weight_map": weight_map})
            _write_json(
                root / MANIFEST_FILENAME,
                {
                    "export_id": export_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "request": request,
                    "authoritative_weight_tree": weights_key,
                    "authoritative_weight_dtypes": authoritative_dtypes,
                    "effective_weight_dtype": "bfloat16",
                    "pending_qb_rule": "applied once by restore_weights before BF16 conversion",
                    "pending_qb_betas_sha256": pending_hash,
                    "tensor_count": len(weight_map),
                    "total_safetensors_bytes": sum(record["bytes"] for record in records),
                    "shards": records,
                    "mesh": dict(mesh.shape),
                    "process_count": jax.process_count(),
                },
            )

        _writer_step(finish)


@draccus.wrap()
def main(config: ExportConfig) -> None:
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    export(config)


if __name__ == "__main__":
    main()
