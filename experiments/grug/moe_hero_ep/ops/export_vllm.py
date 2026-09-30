# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one pinned native Hero checkpoint as BF16 split-expert HF weights."""

import gc
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ShardSpec:
    root: StoragePath
    group: str
    export_id: str
    tensor_names: list[str]

    @property
    def path(self) -> StoragePath:
        return self.root / f"model-{self.group}.safetensors"

    @property
    def progress_path(self) -> StoragePath:
        return self.root / f".export-progress-{self.group}.json"


@dataclass(frozen=True)
class ShardRecord:
    export_id: str
    filename: str
    bytes: int
    sha256: str
    tensor_names: list[str]


def _sha256(path: StoragePath) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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
    """Return the action result on process zero, None elsewhere; propagate writer failure to all ranks."""
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


def _split_names(name: str, num_experts: int) -> list[str]:
    match = EXPERT_BANK.fullmatch(name)
    if match is None:
        return [name]
    return [f"{match[1]}.{i}.{match[2]}.weight" for i in range(num_experts)]


def _split_experts(name: str, value: np.ndarray) -> dict[str, np.ndarray]:
    if EXPERT_BANK.fullmatch(name) is None:
        return {name: value}
    if value.ndim != 3:
        raise ValueError(f"Expected a 3D routed expert bank: {name} {value.shape}")
    return dict(zip(_split_names(name, value.shape[0]), value, strict=True))


def _completed_shard(spec: ShardSpec) -> ShardRecord | None:
    if not spec.progress_path.exists():
        return None
    record = draccus.decode(ShardRecord, json.loads(spec.progress_path.read_text()))
    if (
        record.export_id != spec.export_id
        or record.filename != spec.path.name
        or record.tensor_names != spec.tensor_names
    ):
        raise ValueError(f"Shard identity changed: {spec.progress_path}")
    if not spec.path.exists() or spec.path.size() != record.bytes or _sha256(spec.path) != record.sha256:
        raise ValueError(f"Shard integrity check failed: {spec.path}")
    return record


def _store_shard(spec: ShardSpec, tensors: dict[str, np.ndarray], local_root: Path) -> ShardRecord:
    if sorted(tensors) != spec.tensor_names:
        raise ValueError(f"Incomplete tensor mapping for {spec.group}")
    local = local_root / spec.path.name
    save_file(tensors, local, metadata={"format": "pt"})
    size, checksum = local.stat().st_size, _sha256(StoragePath(str(local)))
    # An upload without progress is uncommitted and may be rewritten after interruption.
    # Committed shards were already verified by _completed_shard.
    spec.path.upload_from(str(local))
    record = ShardRecord(spec.export_id, spec.path.name, size, checksum, spec.tensor_names)
    _write_json(spec.progress_path, asdict(record))
    local.unlink()
    return record


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

        weight_map: dict[str, str] = {}
        records: list[ShardRecord] = []
        payload_size = sum(x.size * x.dtype.itemsize for x in state_dict.values())
        with TemporaryDirectory(prefix="hero-export-") as directory:
            for group, source_names in groups.items():
                expected_names = []
                for name in source_names:
                    expected_names.extend(_split_names(name, config.model.num_experts))
                spec = ShardSpec(root, group, export_id, sorted(expected_names))
                completed = _writer_step(partial(_completed_shard, spec))
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

                    completed = _writer_step(partial(_store_shard, spec, tensors, Path(directory)))
                    del tensors
                    gc.collect()
                if completed is not None:
                    records.append(completed)
                    weight_map.update({name: completed.filename for name in spec.tensor_names})
                    logger.info("%s %s", "Verified" if reuse else "Uploaded", completed.filename)

        def finish():
            _write_json(root / "config.json", hf_config)
            _write_json(root / INDEX_FILENAME, {"metadata": {"total_size": payload_size}, "weight_map": weight_map})
            _write_json(
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

        _writer_step(finish)


@draccus.wrap()
def main(config: ExportConfig) -> None:
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    export(config)


if __name__ == "__main__":
    main()
