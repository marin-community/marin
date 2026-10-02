# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded HF shard writing, with optional verified resume receipts."""

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
import jax
import numpy as np
from fsspec.asyn import get_loop
from fsspec.asyn import sync as fsspec_sync
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import save_file

from levanter.utils.byte_budget import HostByteBudget


def on_export_writer[T](action: Callable[[], T]) -> T | None:
    """Run rank-zero I/O and broadcast failure before any rank enters its next gather.

    Every initialized JAX process calls this on its main thread in the same order.
    Return the result on process zero and None elsewhere. Worker threads never call it.
    """
    error = None
    result = None
    if jax.process_index() == 0:
        try:
            result = action()
        except Exception as exc:
            error = exc
    if not multihost_utils.broadcast_one_to_all(np.asarray(error is None)):
        if error is not None:
            raise error
        raise RuntimeError("Export I/O failed on process zero")
    return result


def write_export_json(path: StoragePath, value: dict) -> None:
    """Create immutable export metadata, or verify identical existing bytes."""
    contents = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = conditional_object(str(path))
    existing = target.read()
    if existing is not None:
        if existing.data != contents:
            raise FileExistsError(f"Refusing to overwrite {path}")
        return
    target.write(contents, expected_version=None)


def _sha256(path: StoragePath) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class HFShardRecord:
    export_id: str
    filename: str
    bytes: int
    sha256: str
    tensor_names: list[str]


@dataclass(frozen=True)
class HFShardProgress:
    """Opt into verified shard reuse for one immutable request and one exporting gang.

    The caller checks its destination/request policy before exporting and publishes
    completion metadata after exporting. Receipts commit only uploaded shards;
    uncommitted uploads may be rewritten. Corrupt or mismatched receipts stop resume.
    """

    root: StoragePath
    export_id: str

    def _path(self, filename: str) -> StoragePath:
        group = filename.removeprefix("model-").removesuffix(".safetensors")
        return self.root / f".export-progress-{group}.json"

    def completed(self, filename: str, tensor_names: list[str]) -> HFShardRecord | None:
        path = self._path(filename)
        if not path.exists():
            return None
        record = draccus.decode(HFShardRecord, json.loads(path.read_text()))
        if record.export_id != self.export_id or record.filename != filename or record.tensor_names != tensor_names:
            raise ValueError(f"Shard identity changed: {path}")
        shard = self.root / filename
        if not shard.exists() or shard.size() != record.bytes or _sha256(shard) != record.sha256:
            raise ValueError(f"Shard integrity check failed: {shard}")
        return record

    def commit(self, record: HFShardRecord) -> None:
        write_export_json(self._path(record.filename), asdict(record))


def save_hf_shards(
    shards: Mapping[str, Mapping[str, jax.Array | jax.ShapeDtypeStruct]],
    load_shard: Callable[[tuple[str, ...]], Mapping[str, jax.Array]],
    path: str,
    *,
    export_host_budget_bytes: int,
    max_concurrent_shards: int,
    progress: HFShardProgress | None = None,
    tensor_names: Mapping[str, tuple[str, ...]] | None = None,
    upload_to_hf: Callable[[str, str], None] | None = None,
) -> list[HFShardRecord]:
    """Gather and save a fixed HF shard layout on all initialized JAX processes.

    All ranks iterate shards and tensor keys in the same order and load matching
    keys, shapes and dtypes.
    Device staging requires space for one full tensor. tensor_names expands a
    tensor's first axis into named outputs. Only process zero retains host shards or writes.

    Reserve twice each shard's payload for host arrays and serialization buffers;
    an oversized shard runs alone. One writer bounds staging to one shard plus
    serialization buffers. One writer finishes before inspecting the next shard.
    Concurrent writer failures propagate to every rank through matched collectives.

    Without progress, existing shard files are overwritten. With progress, verify
    identity, names, size and freshly computed SHA-256 before skipping any gather.
    Return ordered verified/uploaded receipts on process zero, an empty list elsewhere.
    upload_to_hf receives a temporary directory containing exactly one shard and
    its filename, on a process-zero worker thread, and must not enter collectives.
    """
    root = StoragePath(path)
    budget = HostByteBudget(export_host_budget_bytes)
    pending: deque[Future[HFShardRecord | None]] = deque()
    records: list[HFShardRecord] = []

    def drain_one() -> None:
        record = pending.popleft().result()
        if record is not None:
            records.append(record)

    def reserve(num_bytes: int) -> None:
        if len(pending) >= max_concurrent_shards:
            drain_one()
        fsspec_sync(get_loop(), budget.acquire, num_bytes)
        try:
            # A writer may have failed while acquisition waited for its bytes.
            for future in pending:
                if future.done():
                    future.result()
        except BaseException:
            budget.release(num_bytes)
            raise

    def write_shard(filename: str, tensors: dict[str, np.ndarray], reserved_bytes: int) -> HFShardRecord | None:
        try:
            with TemporaryDirectory(prefix="hf-export-") as directory:
                local = Path(directory) / filename
                save_file(tensors, local, metadata={"format": "pt"})
                record = None
                if progress is not None:
                    record = HFShardRecord(
                        progress.export_id,
                        filename,
                        local.stat().st_size,
                        _sha256(StoragePath(str(local))),
                        sorted(tensors),
                    )
                (root / filename).upload_from(str(local))
                if upload_to_hf is not None:
                    upload_to_hf(directory, filename)
                if progress is not None and record is not None:
                    progress.commit(record)
                return record
        finally:
            budget.release(reserved_bytes)

    with ThreadPoolExecutor(max_workers=max_concurrent_shards, thread_name_prefix="hf_export") as pool:
        for filename, shapes in shards.items():
            names = sorted(
                output for key in shapes for output in (tensor_names[key] if tensor_names is not None else (key,))
            )
            if progress is not None:
                record = on_export_writer(lambda: progress.completed(filename, names))
                reuse = multihost_utils.broadcast_one_to_all(np.asarray(record is not None))
                if reuse:
                    if record is not None:
                        records.append(record)
                    continue

            reserved_bytes = 2 * sum(value.size * value.dtype.itemsize for value in shapes.values())
            on_export_writer(lambda: reserve(reserved_bytes))
            tensors: dict[str, np.ndarray] = {}
            try:
                weights = load_shard(tuple(shapes))
                for key in shapes:
                    replicated = jax.sharding.reshard(weights[key], P())
                    host = np.asarray(multihost_utils.process_allgather(replicated, tiled=True), order="C")
                    del replicated
                    if jax.process_index() == 0:
                        outputs = tensor_names[key] if tensor_names is not None else (key,)
                        if outputs == (key,):
                            tensors[key] = host
                        else:
                            tensors.update(zip(outputs, host, strict=True))
                    del host
                del weights

                def submit() -> None:
                    if sorted(tensors) != names:
                        raise ValueError(f"Incomplete tensor mapping for {filename}")
                    pending.append(pool.submit(write_shard, filename, tensors, reserved_bytes))

                on_export_writer(submit)
            except BaseException:
                if jax.process_index() == 0:
                    budget.release(reserved_bytes)
                raise
            del tensors
            if max_concurrent_shards == 1:
                on_export_writer(drain_one)

        def finish() -> None:
            while pending:
                drain_one()

        on_export_writer(finish)

    # Reused shards can be encountered before earlier asynchronous writes finish.
    by_filename = {record.filename: record for record in records}
    return [by_filename[filename] for filename in shards] if progress is not None and jax.process_index() == 0 else []
