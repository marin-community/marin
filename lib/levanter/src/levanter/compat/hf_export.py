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


def run_on_export_writer[T](action: Callable[[], T]) -> T | None:
    """Run a rank-zero action and report its failure collectively.

    Every initialized JAX process calls this on its main thread in the same order.
    Worker threads never call it. Asynchronous errors propagate when the action
    observes them, for example by reading a completed writer future.
    Return the result on process zero and None elsewhere.
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
class HFShardResume:
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


class _HFShardWriter:
    """Own rank-zero write state for one export.

    The main thread manages pending futures and returned receipts. Workers write
    shards and release their bytes; these methods never enter collectives.
    The caller owns reserved bytes until submit succeeds; the worker releases them
    afterward. The surrounding executor context waits for workers on every exit.
    """

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        path: str,
        export_host_budget_bytes: int,
        max_concurrent_shards: int,
        resume: HFShardResume | None,
        upload_to_hf: Callable[[str, str], None] | None,
    ) -> None:
        self._pool = pool
        self._root = StoragePath(path)
        self._budget = HostByteBudget(export_host_budget_bytes)
        self._max_concurrent_shards = max_concurrent_shards
        self._resume = resume
        self._upload_to_hf = upload_to_hf
        self._pending: deque[Future[HFShardRecord | None]] = deque()
        self._records: list[HFShardRecord] = []

    def resume_shard(self, filename: str, tensor_names: list[str]) -> bool:
        """Verify and retain an existing receipt, returning whether it can be reused."""
        if self._resume is None:
            return False
        record = self._resume.completed(filename, tensor_names)
        if record is None:
            return False
        self._records.append(record)
        return True

    def drain_one(self) -> None:
        """Wait for one pending write and retain its receipt, propagating its failure."""
        record = self._pending.popleft().result()
        if record is not None:
            self._records.append(record)

    def reserve(self, num_bytes: int) -> None:
        """Wait for capacity and observe completed writer failures before returning."""
        if len(self._pending) >= self._max_concurrent_shards:
            self.drain_one()
        fsspec_sync(get_loop(), self._budget.acquire, num_bytes)
        try:
            # A writer may have failed while acquisition waited for its bytes.
            for future in self._pending:
                if future.done():
                    future.result()
        except BaseException:
            self.release(num_bytes)
            raise

    def release(self, num_bytes: int) -> None:
        self._budget.release(num_bytes)

    def submit(
        self,
        filename: str,
        tensors: dict[str, np.ndarray],
        tensor_names: list[str],
        reserved_bytes: int,
    ) -> None:
        """Validate names and transfer the reservation to a worker on success."""
        if sorted(tensors) != tensor_names:
            raise ValueError(f"Incomplete tensor mapping for {filename}")
        self._pending.append(self._pool.submit(self._write_shard, filename, tensors, reserved_bytes))

    def finish(self) -> list[HFShardRecord]:
        """Return receipts after observing all pending writes."""
        while self._pending:
            self.drain_one()
        return self._records

    def _write_shard(self, filename: str, tensors: dict[str, np.ndarray], reserved_bytes: int) -> HFShardRecord | None:
        try:
            with TemporaryDirectory(prefix="hf-export-") as directory:
                local = Path(directory) / filename
                save_file(tensors, local, metadata={"format": "pt"})
                record = None
                if self._resume is not None:
                    record = HFShardRecord(
                        self._resume.export_id,
                        filename,
                        local.stat().st_size,
                        _sha256(StoragePath(str(local))),
                        sorted(tensors),
                    )
                (self._root / filename).upload_from(str(local))
                if self._upload_to_hf is not None:
                    self._upload_to_hf(directory, filename)
                if self._resume is not None and record is not None:
                    self._resume.commit(record)
                return record
        finally:
            self.release(reserved_bytes)


def save_hf_shards(
    shards: Mapping[str, Mapping[str, jax.Array | jax.ShapeDtypeStruct]],
    load_shard: Callable[[tuple[str, ...]], Mapping[str, jax.Array]],
    path: str,
    *,
    export_host_budget_bytes: int,
    max_concurrent_shards: int,
    resume: HFShardResume | None = None,
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
    Asynchronous writer errors propagate when the main thread observes completed
    futures at matched coordination points; intervening gathers can finish first.

    Without resume, existing shard files are overwritten. With resume, verify
    identity, names, size and freshly computed SHA-256 before skipping any gather.
    Return ordered receipts on process zero when resume is enabled; return an empty list otherwise.
    upload_to_hf receives a temporary directory containing exactly one shard and
    its filename, on a process-zero worker thread, and must not enter collectives.
    """
    with ThreadPoolExecutor(max_workers=max_concurrent_shards, thread_name_prefix="hf_export") as pool:
        writer = _HFShardWriter(pool, path, export_host_budget_bytes, max_concurrent_shards, resume, upload_to_hf)
        for filename, shapes in shards.items():
            names = sorted(
                output for key in shapes for output in (tensor_names[key] if tensor_names is not None else (key,))
            )
            if resume is not None:
                reused = run_on_export_writer(lambda: writer.resume_shard(filename, names))
                reuse = multihost_utils.broadcast_one_to_all(np.asarray(bool(reused)))
                if reuse:
                    continue

            reserved_bytes = 2 * sum(value.size * value.dtype.itemsize for value in shapes.values())
            run_on_export_writer(lambda: writer.reserve(reserved_bytes))
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

                run_on_export_writer(lambda: writer.submit(filename, tensors, names, reserved_bytes))
            except BaseException:
                if jax.process_index() == 0:
                    writer.release(reserved_bytes)
                raise
            del tensors
            if max_concurrent_shards == 1:
                run_on_export_writer(writer.drain_one)

        records = run_on_export_writer(writer.finish)

    if resume is None or records is None:
        return []
    # Reused shards can be encountered before earlier asynchronous writes finish.
    by_filename = {record.filename: record for record in records}
    return [by_filename[filename] for filename in shards]
