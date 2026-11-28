# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import collections
import contextlib
import ctypes
import errno
import logging
import math
import os
import pathlib
import socket
import sys
import threading
import time
import urllib.parse
import zlib
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache, partial
from typing import Any, Callable, Literal, Optional, Sequence

import equinox
import haliax as hax
import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tensorstore as ts
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from jax._src.mesh import get_concrete_mesh
from jax._src.sharding import IndivisibleError
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding
from jaxtyping import PyTree

from rigging.filesystem.cross_region import record_transfer
from rigging.filesystem.storage_path import StoragePath, prefix_join

from levanter._debug_logging import flush_debug_output
from levanter.checkpoint_manifest import CheckpointArray, build_manifest, read_manifest, write_manifest
from levanter.utils import jax_utils
from levanter.utils.byte_budget import HostByteBudget

logger = logging.getLogger(__name__)

ARRAY_DRIVER = "zarr3"
FileIoLocking = Literal["os", "lockfile", "none"]
KVSTORE_DRIVER = "ocdbt"
_FILE_KVSTORE_DRIVER = "file"
# TensorStore's file driver writes a key through "<key>.__lock" and renames it into place.
_LOCK_FILE_SUFFIX = ".__lock"
# Orders manifest commits across processes. TensorStore's lock excludes writers but does not
# refresh another client's cached directory entry, so a commit can rename over a newer manifest.
_COMMIT_ORDER_LOCK_NAME = ".commit-order.lock"
_COMMIT_ORDER_TIMEOUT_S = 600
# A live save refreshes this file. The next save removes lock files only after it is gone or stale.
_WRITER_LEASE_NAME = ".writer-lease"
_WRITER_LEASE_REFRESH_S = 30
_WRITER_LEASE_STALE_S = 90
_AT_FDCWD = -100
_AT_STATX_FORCE_SYNC = 0x2000
_STATX_BASIC_STATS = 0x7FF
# JAX's memory kind for host memory a device can address. Offloaded optimizer state lives here.
_HOST_MEMORY_KIND = "pinned_host"
_PAGEABLE_HOST_MEMORY_KIND = "unpinned_host"
_GPU_PLATFORM = "gpu"
# Chunks a save stages at once. The budget is per process, so a node holds it once per local
# process, and a process carries about four times what it holds in flight
# (_STAGED_BYTE_OVERHEAD) on top of its resident shard of the offloaded state. A save blocks
# training only for the bytes it must stage past this budget. In-flight bytes are bounded by a
# process's share of the write, so a budget above that share never binds.
_DEFAULT_STAGED_CHUNKS = 32
# Host memory a staged byte occupies while its write is in flight, for reporting only.
_STAGED_BYTE_OVERHEAD = 4
_JEMALLOC_LIBRARY_NAME = "jemalloc"


def _malloc_trim() -> None:
    """Ask glibc to return unused heap pages to the OS."""
    if sys.platform != "linux":
        return

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    released = libc.malloc_trim(0)
    logger.info("malloc_trim after checkpoint commits returned %d", released)


def _trim_host_memory_after_commits(commit_futures: Sequence[ts.Future]) -> None:
    """Schedule one heap trim after every local TensorStore commit finishes."""
    if _JEMALLOC_LIBRARY_NAME in os.environ.get("LD_PRELOAD", ""):
        return

    remaining = len(commit_futures)
    if remaining == 0:
        return

    lock = threading.Lock()

    def commit_finished(_: ts.Future) -> None:
        nonlocal remaining
        with lock:
            remaining -= 1
            all_finished = remaining == 0
        if all_finished:
            _malloc_trim()

    for future in commit_futures:
        future.add_done_callback(commit_finished)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f}GiB"


def _estimate_array_nbytes(array: Any) -> int:
    size = getattr(array, "size", None)
    dtype = getattr(array, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if size is None or itemsize is None:
        return 0
    return int(size) * int(itemsize)


def build_kvstore_spec(path: str) -> dict:
    """Build a tensorstore kvstore spec for an S3, GCS, or local URI.

    tensorstore reads neither AWS_ENDPOINT_URL nor AWS_DEFAULT_REGION from the environment.
    Custom endpoints use virtual-hosted-style addressing, which CoreWeave requires.
    """
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        spec: dict = {"driver": "s3", "path": parsed.path.lstrip("/")}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            # Virtual-hosted style: bucket becomes a subdomain of the endpoint
            # host and the ``bucket`` field is omitted (tensorstore #285).
            scheme, _, host = endpoint.partition("://")
            spec["endpoint"] = f"{scheme}://{bucket}.{host}"
        else:
            spec["bucket"] = bucket
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            spec["aws_region"] = region
        elif endpoint:
            # Custom endpoint with no explicit region: use a placeholder to prevent
            # tensorstore from trying (and failing) to discover the region via HEAD bucket.
            spec["aws_region"] = "us-east-1"

        # TensorStore's default chain does not reliably load a mounted shared
        # credentials file in a task pod. Its profile provider also requires
        # a readable config path, even if that config is empty.
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            spec["aws_credentials"] = {"type": "environment"}
        elif credentials_file := os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
            spec["aws_credentials"] = {
                "type": "profile",
                "profile": os.environ.get("AWS_PROFILE") or "default",
                "credentials_file": credentials_file,
                "config_file": os.environ.get("AWS_CONFIG_FILE") or os.devnull,
            }

        return spec
    elif parsed.scheme == "gs":
        return {"driver": "gcs", "bucket": parsed.netloc, "path": parsed.path.lstrip("/")}
    elif parsed.scheme in ("", "file"):
        file_path = urllib.parse.unquote(parsed.path) if parsed.scheme == "file" else path
        return {"driver": _FILE_KVSTORE_DRIVER, "path": os.path.abspath(file_path)}
    else:
        raise ValueError(f"Unsupported URI scheme for tensorstore: {parsed.scheme!r} in {path!r}")


def _slice_shard_on_device(data, axis: int, start: int, limit: int):
    """Slice a single-device shard under a one-device mesh.

    A training mesh rejects the single-device operand. Orbax does the same.
    """
    shard_mesh = jax.sharding.Mesh(np.array(list(data.sharding.device_set)), ("shard",))
    with jax.sharding.set_mesh(shard_mesh):
        return jax.lax.slice_in_dim(data, start_index=start, limit_index=limit, axis=axis)


async def _transfer_shard_to_pageable_host(shard, local_slice: tuple[int, int, int] | None = None) -> np.ndarray:
    """Return a detached pageable-host snapshot, restricted to ``local_slice``."""
    data = shard.data if local_slice is None else _slice_shard_on_device(shard.data, *local_slice)

    if getattr(data.sharding, "memory_kind", None) == _HOST_MEMORY_KIND:
        # Already in host memory, and copying it to host again would alias rather than move.
        return np.array(data, copy=True)

    # np.array(data) populates a GPU array's host cache, including for disposable slices. Going
    # through a pageable CPU array lets the CUDA DMA pool reuse its internal pinned bounce buffer.
    if data.device.platform == _GPU_PLATFORM:
        cpu_device = jax.local_devices(backend="cpu")[0]
        pageable_sharding = SingleDeviceSharding(cpu_device, memory_kind=_PAGEABLE_HOST_MEMORY_KIND)
        staged = jax.device_put(data, pageable_sharding)
        try:
            # Let the other staging coroutines enqueue their transfers before materialization blocks.
            await asyncio.sleep(0)
            # The private NumPy snapshot must outlive the disposable JAX staging array.
            return np.array(staged, copy=True)
        finally:
            staged.delete()

    data.copy_to_host_async()
    # Yield so the remaining shards' copies can be enqueued before this one blocks.
    await asyncio.sleep(0)
    # TensorStore may retain this array. It must not reference the buffer training donates next.
    return np.array(data, copy=True)


@dataclass(frozen=True)
class TensorStoreWriteConfig:
    """How a checkpoint save divides work across the processes that hold the state."""

    max_write_replicas: int = 1024
    """Cap on how many replicas of an array write part of it. 1 disables replica splitting."""

    min_replica_slice_bytes: int = 16 * 1024**2
    """Do not split an array when doing so would give each replica less than this."""

    max_chunk_bytes: int = 512 * 1024**2
    """Upper bound on one zarr3 chunk. Bounds the size of a single object store write."""

    max_staged_host_bytes: int = _DEFAULT_STAGED_CHUNKS * max_chunk_bytes
    """Host memory this process may hold in staged snapshots at once.

    A save whose share fits returns while the commits drain. A larger save rolls through.
    Under ``lockfile`` locking, a share that does not fit is one OCDBT manifest commit per wave
    of this budget. A share that fits commits each chunk as it is staged.
    """

    cache_pool_bytes: int = 1024**3
    """Soft limit for each TensorStore write cache."""

    data_copy_concurrency: int = 16
    """Maximum CPU concurrency TensorStore uses to copy and encode checkpoint data."""

    file_io_locking: FileIoLocking | None = None
    """How TensorStore's file driver locks a key while committing it.

    None picks ``lockfile`` when several processes save to a local path, and TensorStore's default
    otherwise. The default advisory OS locks do not exclude writers on different NFS clients, which
    then race on the OCDBT manifest commit (``Failed to rename ... manifest.ocdbt.__lock``); an
    exclusively created lock file is enforced by the NFS server. A lock file outlives a process that
    crashes mid-commit and would then block every later save to that directory, so a ``lockfile``
    save first removes lock files abandoned there by an earlier, interrupted save.
    """

    def __post_init__(self) -> None:
        if self.max_write_replicas < 1:
            raise ValueError(f"max_write_replicas must be at least 1, got {self.max_write_replicas}")
        for name in ("min_replica_slice_bytes", "max_chunk_bytes", "max_staged_host_bytes", "data_copy_concurrency"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.cache_pool_bytes < 0:
            raise ValueError(f"cache_pool_bytes must be non-negative, got {self.cache_pool_bytes}")


# Every TensorStore concurrency knob a checkpoint read can reach, across kvstore drivers.
_CONCURRENCY_RESOURCES = (
    "data_copy_concurrency",
    "s3_request_concurrency",
    "gcs_request_concurrency",
    "http_request_concurrency",
    "file_io_concurrency",
)


class ReplicaRestoreMode(StrEnum):
    """How a restore obtains shards held by more than one device."""

    EVERY_REPLICA = "every_replica"
    ONE_REPLICA = "one_replica"


@dataclass(frozen=True)
class TensorStoreReadConfig:
    """How a restore reads shards back."""

    max_in_flight_bytes: int = 16 * 1024**3
    """Target for transient staging memory per process.

    One shard larger than this target proceeds alone so the restore cannot deadlock.
    """

    request_concurrency: int = 128
    """Concurrent object-store requests. TensorStore defaults to 32, which on one GB200 rack
    read a hero checkpoint in 125.6s against 96.9s at 128."""

    replica_mode: ReplicaRestoreMode = ReplicaRestoreMode.ONE_REPLICA
    """Whether every replica reads or the restore attempts one reader per replicated shard.

    The one-reader mode falls back to per-replica reads on TPU and unsupported shardings.
    """

    def __post_init__(self) -> None:
        if self.max_in_flight_bytes <= 0:
            raise ValueError(f"max_in_flight_bytes must be positive, got {self.max_in_flight_bytes}")
        if self.request_concurrency <= 0:
            raise ValueError(f"request_concurrency must be positive, got {self.request_concurrency}")


def _tensorstore_read_context(config: TensorStoreReadConfig) -> ts.Context:
    """JAX's default context carrying this restore's concurrency."""
    spec = ts_impl._TS_CONTEXT.spec.to_json()
    # A driver ignores a resource it does not use, so set every store's knob together.
    for resource in _CONCURRENCY_RESOURCES:
        spec[resource] = {"limit": config.request_concurrency}
    return ts.Context(spec)


def _file_io_locking(config: TensorStoreWriteConfig, checkpoint_dir: str) -> FileIoLocking | None:
    if config.file_io_locking is not None:
        return config.file_io_locking
    if jax.process_count() > 1 and build_kvstore_spec(checkpoint_dir)["driver"] == _FILE_KVSTORE_DRIVER:
        return "lockfile"
    return None


def _checkpoint_directory(checkpoint_dir: str) -> str | None:
    kvstore = build_kvstore_spec(checkpoint_dir)
    if kvstore["driver"] != _FILE_KVSTORE_DRIVER:
        return None
    return kvstore["path"]


def _delete_lock_files(directory: str) -> None:
    if not os.path.isdir(directory):
        return
    for lock_file in pathlib.Path(directory).rglob(f"*{_LOCK_FILE_SUFFIX}"):
        logger.warning("Removing lock file %s abandoned by an interrupted save", lock_file)
        lock_file.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_writer_lease(path: str) -> tuple[str, int] | None:
    try:
        text = pathlib.Path(path).read_text()
    except FileNotFoundError:
        return None
    host, _, pid = text.strip().partition(" ")
    if not host or not pid.isdigit():
        return None
    return host, int(pid)


def _writer_lease_is_live(path: str) -> bool:
    """A lease is live when its local process still exists, or a remote one refreshed it recently."""
    parsed = _read_writer_lease(path)
    if parsed is None:
        return False
    host, pid = parsed
    if host == socket.gethostname():
        # This process is between saves when it reads its own lease. A crashed save has a different pid.
        if pid == os.getpid():
            return False
        return _pid_alive(pid)
    try:
        age = time.time() - os.stat(path).st_mtime
    except FileNotFoundError:
        return False
    return age < _WRITER_LEASE_STALE_S


def _locks_may_be_removed(directory: str) -> bool:
    """False while a live save holds the directory. A stale or absent lease leaves the locks abandoned."""
    path = os.path.join(directory, _WRITER_LEASE_NAME)
    return not (os.path.exists(path) and _writer_lease_is_live(path))


class _WriterLease:
    """Names the save that currently owns a checkpoint directory."""

    def __init__(self, directory: str):
        self.path = os.path.join(directory, _WRITER_LEASE_NAME)
        self._directory = directory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _create(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
        finally:
            os.close(fd)

    def _refresh(self) -> None:
        while not self._stop.wait(_WRITER_LEASE_REFRESH_S):
            try:
                os.utime(self.path)
            except FileNotFoundError:
                return

    def try_acquire(self) -> bool:
        """Take the directory if its lease is absent or stale. A live lease is left untouched."""
        if not _locks_may_be_removed(self._directory):
            return False
        if os.path.lexists(self.path):
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
        try:
            self._create()
        except FileExistsError:
            return False
        # The previous writer is gone. Its lock files can no longer belong to a live commit.
        _delete_lock_files(self._directory)
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return True

    def acquire(self) -> None:
        deadline = time.monotonic() + _COMMIT_ORDER_TIMEOUT_S
        while not self.try_acquire():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for the checkpoint writer holding {self.path}")
            time.sleep(0.05)

    def release(self) -> None:
        self._stop.set()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _claim_writer_lease(checkpoint_dir: str) -> _WriterLease | None:
    directory = _checkpoint_directory(checkpoint_dir)
    if directory is None:
        return None
    os.makedirs(directory, exist_ok=True)
    lease = _WriterLease(directory)
    lease.acquire()
    return lease


def _release_lease_when_commits_finish(lease: _WriterLease, commit_futures: Sequence[ts.Future]) -> None:
    if not commit_futures:
        lease.release()
        return
    remaining = len(commit_futures)
    lock = threading.Lock()

    def finished(future: ts.Future) -> None:
        nonlocal remaining
        del future
        with lock:
            remaining -= 1
            if remaining == 0:
                lease.release()

    for future in commit_futures:
        future.add_done_callback(finished)


def _prepare_file_io_locking(
    config: TensorStoreWriteConfig, checkpoint_dir: str
) -> tuple[FileIoLocking | None, _WriterLease | None]:
    """Resolve the save's locking mode and claim the directory before removing stale lock files.

    Process 0 takes the writer lease, then deletes lock files. A lease whose process is still
    running, or whose remote holder refreshed it within ``_WRITER_LEASE_STALE_S``, blocks that
    sweep. The other processes wait at the barrier until the sweep has finished.
    """
    file_io_locking = _file_io_locking(config, checkpoint_dir)
    lease = None
    if file_io_locking == "lockfile" and jax.process_index() == 0:
        lease = _claim_writer_lease(checkpoint_dir)
    if file_io_locking == "lockfile":
        jax_utils.barrier_sync()
    return file_io_locking, lease


def _revalidate_directory(directory: str) -> None:
    """Ask the file server for ``directory``'s current entries.

    An NFS client answers later lookups from its directory cache and can keep opening the inode
    from before another client replaced the file. ``statx(AT_STATX_FORCE_SYNC)`` refreshes that
    cache. A kernel without ``statx`` leaves the cache as it is.
    """
    buffer = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    libc.statx.restype = ctypes.c_int
    if libc.statx(_AT_FDCWD, os.fsencode(directory), _AT_STATX_FORCE_SYNC, _STATX_BASIC_STATS, buffer) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.ENOSYS:
        return
    raise OSError(error, os.strerror(error), directory)


class _HeldCommit:
    def __init__(self, path: str, fd: int):
        self.path = path
        self._fd: int | None = fd

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        os.close(fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


class _CommitOrder:
    """Serialize one directory's manifest commits across processes.

    The holder refreshes the directory, then publishes. The next holder therefore rewrites the
    manifest the previous holder left, rather than a copy cached before that publish.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self._path = os.path.join(directory, _COMMIT_ORDER_LOCK_NAME)

    def acquire(self) -> _HeldCommit:
        deadline = time.monotonic() + _COMMIT_ORDER_TIMEOUT_S
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting to commit {self.directory}; {self._path} is still held")
                time.sleep(0.02)
        os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
        return _HeldCommit(self._path, fd)

    @contextlib.contextmanager
    def hold(self):
        held = self.acquire()
        try:
            _revalidate_directory(self.directory)
            yield
        finally:
            held.release()


def _prepare_commit_order(checkpoint_dir: str) -> _CommitOrder | None:
    """Order manifest commits when several processes share a filesystem checkpoint.

    Process 0 removes a lock left by a save that died mid-commit. Every process then waits, so
    no one is holding the lock that removal deletes. Call this only after the writer lease is held,
    so a live save is not the owner of that lock.
    """
    directory = _checkpoint_directory(checkpoint_dir)
    if jax.process_count() <= 1 or directory is None:
        return None
    if jax.process_index() == 0 and os.path.isdir(directory):
        abandoned = os.path.join(directory, _COMMIT_ORDER_LOCK_NAME)
        if os.path.lexists(abandoned):
            logger.warning("Removing commit lock %s abandoned by an interrupted save", abandoned)
            os.unlink(abandoned)
    jax_utils.barrier_sync()
    return _CommitOrder(directory)


def _tensorstore_write_context(config: TensorStoreWriteConfig, file_io_locking: FileIoLocking | None) -> ts.Context:
    spec = ts_impl._TS_CONTEXT.spec.to_json()
    spec["cache_pool"] = {"total_bytes_limit": config.cache_pool_bytes}
    spec["cache_pool#remote"] = {"total_bytes_limit": config.cache_pool_bytes}
    spec["data_copy_concurrency"] = {"limit": config.data_copy_concurrency}
    if file_io_locking is not None:
        spec["file_io_locking"] = {"mode": file_io_locking}
    return ts.Context(spec)


@dataclass(frozen=True)
class _WritePlan:
    """Which bytes of one global array each process writes, and the chunk grid they land on.

    With no ``split_axis``, ``writer_replica`` writes each shard whole. Otherwise replica
    ``r`` writes block ``r`` along ``split_axis``.
    """

    chunk_shape: tuple[int, ...]
    split_axis: int | None
    write_replicas: int
    block: int
    """Length of one replica's slice along ``split_axis``; unused when there is no split."""
    replicas: int = 1
    """How many processes hold each shard. Exceeds ``write_replicas`` when a split was rejected."""
    writer_replica: int = 0
    """Which replica writes each shard whole; unused when there is a split."""


@dataclass(frozen=True)
class _ShardWrite:
    """One device's share of one array: where it lands, and which part of the shard it is."""

    index: tuple
    """Index into the global array."""
    slice_axis: int | None
    """Axis the shard is sliced along, or None to write the whole shard."""
    slice_start: int
    slice_limit: int

    @property
    def local_slice(self) -> tuple[int, int, int] | None:
        if self.slice_axis is None:
            return None
        return self.slice_axis, self.slice_start, self.slice_limit


def _hashable_index(index) -> tuple:
    return tuple((entry.start, entry.stop) if isinstance(entry, slice) else entry for entry in index)


def _uniform_replica_count(sharding: Sharding, shape: tuple[int, ...]) -> int | None:
    """Replicas per index, or None when indices disagree (no single split factor applies)."""
    counts: collections.Counter = collections.Counter()
    for index in sharding.devices_indices_map(shape).values():
        counts[_hashable_index(index)] += 1

    distinct = set(counts.values())
    if len(distinct) != 1:
        return None
    return distinct.pop()


def _shard_shape(sharding: Sharding, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The per-device shard shape; raises when the sharding does not divide the array evenly."""
    try:
        return tuple(sharding.shard_shape(shape))
    except IndivisibleError as e:
        raise ValueError(
            f"Cannot checkpoint an array of shape {shape} under {sharding}: the sharding does not "
            "divide it evenly, so its shards have different shapes."
        ) from e


def _is_safe_to_slice(sharding: Sharding, shard_shape: tuple[int, ...]) -> bool:
    """Whether a shard can be sliced on device without risking a hang.

    Slicing a small pinned-host array can hang on layout requirements (b/417243451).
    """
    if getattr(sharding, "memory_kind", None) != _HOST_MEMORY_KIND:
        return True
    return len(shard_shape) >= 2 and math.prod(shard_shape) % 1024 == 0 and shard_shape[-1] % 128 == 0


def _capped_chunk_shape(local_shape: tuple[int, ...], itemsize: int, max_chunk_bytes: int) -> tuple[int, ...]:
    """Halve ``local_shape`` down to ``max_chunk_bytes``, keeping it an exact divisor.

    Only even axes halve, so every writer's region is a whole number of chunks. A zero-length
    axis becomes a chunk of 1, which zarr3 requires.
    """
    chunk = [max(size, 1) for size in local_shape]
    while math.prod(chunk) * itemsize > max_chunk_bytes:
        divisible = [axis for axis, size in enumerate(chunk) if size % 2 == 0]
        if not divisible:
            break
        chunk[max(divisible, key=lambda axis: chunk[axis])] //= 2
    return tuple(chunk)


def plan_array_write(path: str, array, config: TensorStoreWriteConfig) -> _WritePlan:
    """Decide how ``array`` is divided among the processes holding it.

    Split along the first shard axis that divides evenly by the replica count, after Orbax's
    ``replica_slices.py``. Every process must reach the same answer, so this depends only on
    ``path`` and the global shape and sharding. A replica count dividing no axis takes the
    widest smaller one that does. The sole writer for an un-splittable array comes from ``path``.
    """
    shape = tuple(array.shape)
    itemsize = array.dtype.itemsize

    if not isinstance(array, jax.Array):
        # Unsharded host array: one process writes all of it.
        return _WritePlan(
            chunk_shape=_capped_chunk_shape(shape, itemsize, config.max_chunk_bytes),
            split_axis=None,
            write_replicas=1,
            block=0,
        )

    sharding = array.sharding
    shard_shape = _shard_shape(sharding, shape)
    replica_count = _uniform_replica_count(sharding, shape)
    single_writer = _WritePlan(
        chunk_shape=_capped_chunk_shape(shard_shape, itemsize, config.max_chunk_bytes),
        split_axis=None,
        write_replicas=1,
        block=0,
        replicas=replica_count or 1,
        # Every replica holds the whole shard, so any may write it.
        writer_replica=zlib.crc32(path.encode()) % replica_count if replica_count else 0,
    )

    if not _is_safe_to_slice(sharding, shard_shape):
        return single_writer

    if replica_count is None:
        return single_writer

    for replicas in range(min(replica_count, config.max_write_replicas), 1, -1):
        split_axis = next((axis for axis, size in enumerate(shard_shape) if size % replicas == 0), None)
        if split_axis is None:
            continue
        # Fewer writers give each a larger slice, so retry under the floor.
        if math.prod(shard_shape) * itemsize // replicas < config.min_replica_slice_bytes:
            continue

        block = shard_shape[split_axis] // replicas
        local_shape = shard_shape[:split_axis] + (block,) + shard_shape[split_axis + 1 :]
        return _WritePlan(
            chunk_shape=_capped_chunk_shape(local_shape, itemsize, config.max_chunk_bytes),
            split_axis=split_axis,
            write_replicas=replicas,
            block=block,
            replicas=replica_count,
        )

    return single_writer


def _shard_write_region(shard, plan: _WritePlan) -> _ShardWrite | None:
    """What this device writes for this array, or None when it writes nothing."""
    if plan.split_axis is None:
        if shard.replica_id != plan.writer_replica:
            return None
        return _ShardWrite(index=shard.index, slice_axis=None, slice_start=0, slice_limit=0)

    if shard.replica_id >= plan.write_replicas:
        return None

    axis = plan.split_axis
    local_start = shard.replica_id * plan.block
    start = (shard.index[axis].start or 0) + local_start
    return _ShardWrite(
        index=shard.index[:axis] + (slice(start, start + plan.block),) + shard.index[axis + 1 :],
        slice_axis=axis,
        slice_start=local_start,
        slice_limit=local_start + plan.block,
    )


def _process_staged_bytes(array, plan: _WritePlan) -> int:
    """Bytes this process stages for ``array``."""
    if not isinstance(array, jax.Array):
        return _estimate_array_nbytes(array) if jax.process_index() == 0 else 0
    return sum(
        _estimate_array_nbytes(shard.data) // plan.write_replicas
        for shard in array.addressable_shards
        if _shard_write_region(shard, plan) is not None
    )


def _create_ocdbt_spec(
    checkpoint_root: str,
    array_path: str | None,
    *,
    entry: "CheckpointArray | None" = None,
) -> dict:
    """Build a TensorStore spec over an OCDBT kvstore.

    ``entry`` pins the zarr3 chunk grid, so concurrent writers never share a chunk. Reads omit
    it and take the grid from storage.
    """
    spec: dict[str, Any] = {
        "driver": ARRAY_DRIVER,
        "kvstore": {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint_root)},
    }

    if array_path:
        spec["kvstore"]["path"] = array_path

    if entry is not None:
        spec["metadata"] = {
            "shape": list(entry.shape),
            "data_type": entry.dtype,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(entry.chunk_shape)}},
        }

    return spec


async def _list_ocdbt_keys(checkpoint_root: str) -> list[str]:
    """List all keys in an OCDBT TensorStore kvstore."""
    kvstore_spec = _create_ocdbt_spec(checkpoint_root, array_path=None)["kvstore"]
    kvstore = await ts.KvStore.open(kvstore_spec)
    keys_bytes = await kvstore.list()
    return [key.decode("utf-8") for key in keys_bytes]


def _is_named_or_none(x):
    return x is None or is_named_array(x)


def _flatten_serializable_leaves(pytree) -> tuple[list[str], list[Any]]:
    """Flatten a pytree to (storage path, array) pairs, dropping leaves with nothing to store.

    Paths come from tree position with dots turned into slashes: ``model.layers.0.w_q``
    becomes ``model/layers/0/w_q``. Python scalars become arrays.
    """
    leaf_key_paths = jax_utils.leaf_key_paths(pytree, is_leaf=is_named_array)
    assert len(jax.tree.leaves(leaf_key_paths, is_leaf=is_named_array)) == len(
        jax.tree.leaves(pytree, is_leaf=is_named_array)
    )

    paths = jtu.tree_map(lambda key_path: "/".join(key_path.split(".")), leaf_key_paths)

    # make a dataclass since tuples are pytrees
    @dataclass
    class Pair:
        path: str
        leaf: Any

    zipped = jax.tree.map(lambda x, y: Pair(x, y), paths, pytree, is_leaf=lambda x: x is None)
    paired_leaves = jax.tree.leaves(zipped)

    def to_array(leaf):
        if is_named_array(leaf):
            return leaf.array
        if isinstance(leaf, (int, float, bool, complex)):
            return jnp.array(leaf)
        return leaf

    with_arrays = [(pair.path, to_array(pair.leaf)) for pair in paired_leaves]
    keep = [(path, array) for path, array in with_arrays if equinox.is_array_like(array)]
    return [path for path, _ in keep], [array for _, array in keep]


def _log_write_share(
    paths: Sequence[str],
    arrays: Sequence[Any],
    plans: Sequence[_WritePlan],
    total_array_bytes: int,
) -> None:
    """Report this process's share, and how much of it more processes would not relieve."""
    staged = [_process_staged_bytes(array, plan) for array, plan in zip(arrays, plans)]
    solo = sorted(
        (
            (nbytes, path)
            for path, plan, nbytes in zip(paths, plans, staged)
            if nbytes and plan.write_replicas == 1 and plan.replicas > 1
        ),
        reverse=True,
    )
    capped = sum(nbytes for plan, nbytes in zip(plans, staged) if nbytes and 1 < plan.write_replicas < plan.replicas)
    logger.info(
        "Checkpoint gives this process %s of %s across %d of %d arrays; %s of that is %d arrays "
        "it writes alone and %s is split no further than the replica cap%s",
        _format_gib(sum(staged)),
        _format_gib(total_array_bytes),
        sum(1 for nbytes in staged if nbytes),
        len(arrays),
        _format_gib(sum(nbytes for nbytes, _ in solo)),
        len(solo),
        _format_gib(capped),
        "".join(f", {path} at {_format_gib(nbytes)}" for nbytes, path in solo[:3]),
    )


def tree_serialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    manager: Optional[array_ser.GlobalAsyncCheckpointManager] = None,
    *,
    commit_callback: Optional[Callable] = None,
    on_local_commit: Optional[Callable[[str], None]] = None,
    on_staged: Optional[Callable[[int], None]] = None,
    debug_checkpointer: bool = False,
    debug_log_flush: Callable[[logging.Logger], None] | None = flush_debug_output,
    write_config: Optional[TensorStoreWriteConfig] = None,
) -> int:
    """Serialize a PyTree and return the peak host bytes staged by this process."""
    write_config = write_config or TensorStoreWriteConfig()

    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()
        manager_was_none = True
    else:
        manager_was_none = False

    paths, arrays = _flatten_serializable_leaves(pytree)

    total_array_bytes = sum(_estimate_array_nbytes(array) for array in arrays)
    largest_path: str | None = None
    largest_array_bytes = 0
    for path, array in zip(paths, arrays):
        array_bytes = _estimate_array_nbytes(array)
        if array_bytes > largest_array_bytes:
            largest_path = path
            largest_array_bytes = array_bytes

    if commit_callback is None:
        commit_callback = lambda: logger.info("Committed checkpoint to Tensorstore")  # noqa

    if debug_checkpointer:
        logger.info(
            "Checkpoint tensorstore serialize start: dir=%s arrays=%d total=%s largest=%s (%s)",
            checkpoint_dir,
            len(arrays),
            _format_gib(total_array_bytes),
            largest_path or "<none>",
            _format_gib(largest_array_bytes),
        )
        if debug_log_flush is not None:
            debug_log_flush(logger)

    plans = [plan_array_write(path, array, write_config) for path, array in zip(paths, arrays)]
    _log_write_share(paths, arrays, plans, total_array_bytes)
    entries = [
        CheckpointArray(
            path=path,
            shape=tuple(array.shape),
            dtype=jnp.dtype(array.dtype).name,
            chunk_shape=plan.chunk_shape,
        )
        for path, array, plan in zip(paths, arrays, plans)
    ]
    tspecs = [_create_ocdbt_spec(checkpoint_dir, entry.path, entry=entry) for entry in entries]

    if jax.process_index() == 0:
        write_manifest(
            checkpoint_dir, build_manifest(entries, array_driver=ARRAY_DRIVER, kvstore_driver=KVSTORE_DRIVER)
        )

    # Pre-charge the cross-region budget: tensorstore bypasses fsspec, so CrossRegionGuardedFS
    # never sees these bytes. No-op for a local or same-region checkpoint_dir.
    record_transfer(total_array_bytes, checkpoint_dir)

    if debug_checkpointer:
        split = sum(1 for plan in plans if plan.split_axis is not None)
        logger.info(
            "Checkpoint tensorstore serialize starting writes for %s: %d/%d arrays replica-split",
            checkpoint_dir,
            split,
            len(plans),
        )
        if debug_log_flush is not None:
            debug_log_flush(logger)

    staged_host_bytes = _serialize_arrays(
        arrays,
        tspecs,
        plans,
        manager,
        write_config,
        str(checkpoint_dir),
        commit_callback,
        on_local_commit,
        on_staged,
    )

    if debug_checkpointer:
        logger.info("Checkpoint tensorstore serialize handed off async commit for %s", checkpoint_dir)
        if debug_log_flush is not None:
            debug_log_flush(logger)

    if manager_was_none:
        manager.wait_until_finished()

    return staged_host_bytes


class _Wave:
    """One transaction's worth of checkpoint writes. Committing it publishes the manifest once."""

    def __init__(self, txn: ts.Transaction, bound: ts.KvStore):
        self.txn = txn
        self.bound = bound
        self.pending = 0
        self.bytes = 0
        self.close_requested = False
        self.error: BaseException | None = None
        self.stores: dict[str, ts.Future] = {}
        self.writes: list[ts.WriteFutures] = []
        promise, self.future = ts.Promise.new()
        self._promise = promise

    def open_store(self, path: str, metadata: dict, context: ts.Context, *, create: bool) -> ts.Future:
        store = self.stores.get(path)
        if store is None:
            spec = {"driver": ARRAY_DRIVER, "metadata": metadata}
            kvstore = _array_kvstore(self.bound, path)
            if create:
                store = ts.open(spec, kvstore=kvstore, create=True, open=True, context=context)
            else:
                # Another process published this metadata. Reading it back reads the manifest
                # without the lock, and on NFS that fails with ESTALE when another client's
                # commit replaces the manifest during the read.
                store = ts.open(spec, kvstore=kvstore, assume_metadata=True, context=context)
            self.stores[path] = store
        return store

    def succeed(self) -> None:
        self._promise.set_result(None)

    def fail(self, error: BaseException) -> None:
        self._promise.set_exception(error)


class _ManifestWaves:
    """Group this process's checkpoint writes into one OCDBT commit per staging wave.

    Every array in the save shares one OCDBT database, so each ``write().commit`` rewrites
    ``manifest.ocdbt``. On a file kvstore that rewrite takes the file lock. A wave instead
    opens every array on one atomic transactional view of that database and commits it once,
    after every write is staged in it. The next wave starts when the host-byte budget has a
    waiter and this wave's writes are staged, or when every write has been staged.
    """

    def __init__(
        self,
        base: ts.KvStore,
        context: ts.Context,
        gate: HostByteBudget,
        expected_writes: int,
        *,
        create_metadata: bool,
        commit_order: _CommitOrder | None,
    ):
        self._base = base
        self._context = context
        self._gate = gate
        self._expected = expected_writes
        self._create_metadata = create_metadata
        self._commit_order = commit_order
        self._finished = 0
        self._wave: _Wave | None = None
        self._lock = threading.Lock()
        self.commits: list[ts.Future] = []

    def reserve(self, num_bytes: int) -> _Wave:
        """Count ``num_bytes`` in the open wave. Call only after the budget has been acquired.

        The caller must not await between acquiring the budget and this call. The event loop
        would otherwise run a waiter that closes the wave before this write joins it.
        """
        with self._lock:
            wave = self._ensure_wave()
            wave.pending += 1
            wave.bytes += num_bytes
            return wave

    def cancel(self, wave: _Wave, num_bytes: int, error: BaseException) -> None:
        with self._lock:
            wave.pending -= 1
            wave.bytes -= num_bytes
            if wave.error is None:
                wave.error = error
            self._finished += 1
            self._gate.release(num_bytes)
            self._close_if_ready(wave)

    def launch(self, wave: _Wave, data, path: str, metadata: dict, region: _ShardWrite | None) -> None:
        with self._lock:
            store_future = wave.open_store(path, metadata, self._context, create=self._create_metadata)
        store_future.force()

        def opened(fut: ts.Future) -> None:
            try:
                store = fut.result()
                target = store if region is None else store[region.index]
                # TensorStore's referenced-source fast path can commit only fill values
                # for large bfloat16 chunks. Let it copy the staged snapshot instead.
                write = target.write(data)
                write.commit.force()
            except BaseException as error:
                self._write_finished(wave, error)
                return
            with self._lock:
                wave.writes.append(write)
            # A transactional write's commit means it is staged in the transaction, not stored.
            write.commit.add_done_callback(lambda fut: self._write_finished(wave, _future_error(fut)))

        store_future.add_done_callback(opened)

    def close_blocked_wave(self) -> None:
        """Close the wave a budget waiter is stuck behind, once its writes are staged."""
        with self._lock:
            wave = self._wave
            if wave is not None:
                wave.close_requested = True
                self._close_if_ready(wave)

    def _write_finished(self, wave: _Wave, error: BaseException | None) -> None:
        with self._lock:
            if error is not None and wave.error is None:
                wave.error = error
            wave.pending -= 1
            self._finished += 1
            self._close_if_ready(wave)

    def _ensure_wave(self) -> _Wave:
        if self._wave is None:
            # A non-atomic transaction commits metadata and chunks as separate manifest versions.
            # An atomic one over the single OCDBT database is one version, and all or nothing.
            txn = ts.Transaction(atomic=True)
            wave = _Wave(txn, self._base.with_transaction(txn))
            self._wave = wave
            self.commits.append(wave.future)
        return self._wave

    def _close_if_ready(self, wave: _Wave) -> None:
        if wave.pending != 0 or wave is not self._wave:
            return
        # Waiters waking from a previous commit may still fit in this wave.
        # Only close early if this wave has already blocked a new reservation.
        if self._finished != self._expected and not wave.close_requested:
            return
        self._wave = None
        held = wave.bytes
        wave.bytes = 0
        if wave.error is not None:
            wave.txn.abort()
            wave.fail(wave.error)
            self._gate.release(held)
            return
        held_commit = None
        try:
            if self._commit_order is not None:
                held_commit = self._commit_order.acquire()
                _revalidate_directory(self._commit_order.directory)
            commit = wave.txn.commit_async()
        except BaseException as error:
            if held_commit is not None:
                held_commit.release()
            wave.fail(error)
            self._gate.release(held)
            return

        def committed(fut: ts.Future) -> None:
            try:
                fut.result()
            except BaseException as error:
                wave.fail(error)
            else:
                wave.succeed()
            finally:
                if held_commit is not None:
                    held_commit.release()
            self._gate.release(held)

        commit.add_done_callback(committed)


@dataclass(frozen=True)
class _CheckpointWrite:
    """One staged write: a host array, or one device's slice of a sharded array."""

    path: str
    metadata: dict
    num_bytes: int
    region: _ShardWrite | None
    stage: Callable


def _checkpoint_writes(
    arrays: Sequence[Any],
    tspecs: Sequence[dict],
    plans: Sequence[_WritePlan],
) -> list[_CheckpointWrite]:
    """The writes this process contributes. Other processes skip arrays they do not own."""
    jobs: list[_CheckpointWrite] = []
    for array, tspec, plan in zip(arrays, tspecs, plans):
        metadata = tspec["metadata"]
        path = tspec["kvstore"].get("path", "")
        if not isinstance(array, jax.Array):
            # Unsharded and identical everywhere, so process 0 writes all of it.
            if jax.process_index() != 0:
                continue

            def stage_host(array=array):
                async def stage():
                    # The caller owns this buffer and may mutate it after the save returns.
                    return np.array(array, copy=True)

                return stage

            jobs.append(
                _CheckpointWrite(
                    path=path,
                    metadata=metadata,
                    num_bytes=_estimate_array_nbytes(array),
                    region=None,
                    stage=stage_host(),
                )
            )
            continue

        for shard in array.addressable_shards:
            region = _shard_write_region(shard, plan)
            if region is None:
                continue

            def stage_shard(shard=shard, region=region):
                async def stage():
                    return await _transfer_shard_to_pageable_host(shard, region.local_slice)

                return stage

            jobs.append(
                _CheckpointWrite(
                    path=path,
                    metadata=metadata,
                    num_bytes=_estimate_array_nbytes(shard.data) // plan.write_replicas,
                    region=region,
                    stage=stage_shard(),
                )
            )
    return jobs


def _array_kvstore(base: ts.KvStore, path: str) -> ts.KvStore:
    """The OCDBT prefix ``_create_ocdbt_spec`` gives the array at ``path``.

    A non-empty path needs the trailing separator, or ``zarr.json`` is appended onto the array
    name. A bare-array PyTree has the empty path and lives at the root, not under ``/``.
    """
    if not path:
        return base
    return base / f"{path}/"


def _commit_array_metadata(
    base: ts.KvStore,
    context: ts.Context,
    tspecs: Sequence[dict],
    commit_order: _CommitOrder | None,
) -> None:
    """Publish every array's metadata in one transaction.

    Two processes must not create the same ``zarr.json`` in separate transactions: the second
    commit fails with ``ALREADY_EXISTS``. One process publishes the metadata, then every process
    commits only the chunks it owns.
    """

    def publish() -> None:
        txn = ts.Transaction(atomic=True)
        bound = base.with_transaction(txn)
        opened: list[ts.Future] = []
        seen: set[str] = set()
        for tspec in tspecs:
            path = tspec["kvstore"].get("path", "")
            if path in seen:
                continue
            seen.add(path)
            opened.append(
                ts.open(
                    {"driver": ARRAY_DRIVER, "metadata": tspec["metadata"]},
                    kvstore=_array_kvstore(bound, path),
                    create=True,
                    open=True,
                    context=context,
                )
            )
        for future in opened:
            future.result()
        if opened:
            txn.commit_sync()

    if commit_order is None:
        publish()
        return
    with commit_order.hold():
        publish()


def _future_error(fut: ts.Future) -> BaseException | None:
    try:
        fut.result()
    except BaseException as error:
        return error
    return None


def _publish_array_metadata(
    checkpoint_dir: str,
    context: ts.Context,
    tspecs: Sequence[dict],
    commit_order: _CommitOrder | None,
) -> None:
    """Process 0 commits every array's metadata, and the others wait before opening it.

    Opening ``zarr.json`` reads the manifest without the lock. On NFS that read fails with
    ESTALE when another client's commit replaces the manifest, so the others then open with
    the metadata this published instead of reading it back.
    """
    if jax.process_index() == 0:
        base = ts.KvStore.open(
            {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint_dir)},
            context=context,
        ).result()
        _commit_array_metadata(base, context, tspecs, commit_order)
    jax_utils.barrier_sync()


def _write_each_chunk(
    jobs: Sequence[_CheckpointWrite],
    tspecs: Sequence[dict],
    config: TensorStoreWriteConfig,
    context: ts.Context,
    checkpoint_dir: str,
    commit_order: _CommitOrder | None,
    *,
    published_metadata: bool,
) -> tuple[HostByteBudget, list[ts.Future]]:
    """Commit every chunk on its own as soon as it is staged."""
    gate = HostByteBudget(config.max_staged_host_bytes)
    commit_futures: list[ts.Future] = []
    tspec_by_path = {tspec["kvstore"].get("path", ""): tspec for tspec in tspecs}
    stores: dict[str, ts.Future] = {}
    base: ts.KvStore | None = None

    def store_for(path: str) -> ts.Future:
        store = stores.get(path)
        if store is None:
            if base is None:
                store = ts.open(ts.Spec(tspec_by_path[path]), create=True, open=True, context=context)
            else:
                # assume_metadata skips the unlocked manifest read that fails with ESTALE while
                # another client commits.
                spec = {"driver": ARRAY_DRIVER, "metadata": tspec_by_path[path]["metadata"]}
                store = ts.open(spec, kvstore=_array_kvstore(base, path), assume_metadata=True, context=context)
            # Start I/O without waiting; staging never depends on open completion.
            store.force()
            stores[path] = store
        return store

    async def issue_write(job: _CheckpointWrite):
        store_future = store_for(job.path)
        await gate.acquire(job.num_bytes)
        try:
            data = await job.stage()
        except BaseException:
            gate.release(job.num_bytes)
            raise

        promise, commit_future = ts.Promise.new()
        commit_futures.append(commit_future)
        # Fires on failure as well as success, so a failed commit cannot strand the budget.
        commit_future.add_done_callback(lambda _: gate.release(job.num_bytes))

        def store_opened(future: ts.Future) -> None:
            try:
                store = future.result()
                target = store if job.region is None else store[job.region.index]
                # TensorStore's referenced-source fast path can commit only fill values
                # for large bfloat16 chunks. Let it copy the staged snapshot instead.
                write = target.write(data)
                write.commit.add_done_callback(write_finished)
                if commit_order is None:
                    write.commit.force()
                    return
                order = commit_order
            except BaseException as error:
                promise.set_exception(error)
                return

            def publish_latest() -> None:
                try:
                    # Refresh and rewrite the manifest another client last published. Releasing
                    # the lock before result() returns lets the next client reopen that file.
                    with order.hold():
                        write.commit.force()
                        write.commit.result()
                except BaseException as error:
                    # A commit that started reports through write_finished. One that did not
                    # must still settle the promise, or the save waits on it forever.
                    if not commit_future.done():
                        promise.set_exception(error)

            threading.Thread(target=publish_latest, daemon=True).start()

        def write_finished(future: ts.Future) -> None:
            error = _future_error(future)
            if error is None:
                promise.set_result(None)
            else:
                promise.set_exception(error)

        store_future.add_done_callback(store_opened)

    async def write_all():
        nonlocal base
        # Open every store before any staging wait, so one array's open never queues behind
        # another array's budget wait.
        if published_metadata:
            base = await ts.KvStore.open(
                {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint_dir)},
                context=context,
            )
        for job in jobs:
            store_for(job.path)
        await asyncio.gather(*(issue_write(job) for job in jobs))

    asyncio.run(write_all())
    return gate, commit_futures


def _write_in_manifest_waves(
    jobs: Sequence[_CheckpointWrite],
    tspecs: Sequence[dict],
    config: TensorStoreWriteConfig,
    checkpoint_dir: str,
    context: ts.Context,
    commit_order: _CommitOrder | None,
) -> tuple[HostByteBudget, list[ts.Future]]:
    """Commit the chunks staged together as one atomic manifest commit per wave."""
    # Assigned before the first acquire. A waiter calls it once the gate exists.
    waves: _ManifestWaves | None = None

    def close_blocked_wave() -> None:
        if waves is not None:
            waves.close_blocked_wave()

    gate = HostByteBudget(config.max_staged_host_bytes, on_blocked=close_blocked_wave)

    async def stage_write(job: _CheckpointWrite, committer: _ManifestWaves):
        await gate.acquire(job.num_bytes)
        # Join the wave before the next await. See ``_ManifestWaves.reserve``.
        wave = committer.reserve(job.num_bytes)
        try:
            data = await job.stage()
        except BaseException as error:
            committer.cancel(wave, job.num_bytes, error)
            raise
        committer.launch(wave, data, job.path, job.metadata, job.region)

    async def open_base() -> ts.KvStore:
        return await ts.KvStore.open(
            {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint_dir)},
            context=context,
        )

    async def write_all():
        nonlocal waves
        # Concurrent transactions that both create an array's metadata fail with ALREADY_EXISTS.
        # One process publishes it, and the data waves below only write disjoint chunks.
        share_metadata = jax.process_count() > 1 and bool(tspecs)
        if share_metadata:
            if jax.process_index() == 0:
                _commit_array_metadata(await open_base(), context, tspecs, commit_order)
            jax_utils.barrier_sync()
        if not jobs:
            return
        waves = _ManifestWaves(
            await open_base(),
            context,
            gate,
            len(jobs),
            create_metadata=not share_metadata,
            commit_order=commit_order,
        )
        await asyncio.gather(*(stage_write(job, waves) for job in jobs))

    asyncio.run(write_all())
    return gate, [] if waves is None else waves.commits


def _serialize_arrays(
    arrays: Sequence[Any],
    tspecs: Sequence[dict],
    plans: Sequence[_WritePlan],
    manager: array_ser.GlobalAsyncCheckpointManager,
    config: TensorStoreWriteConfig,
    checkpoint_dir: str,
    commit_callback: Callable,
    on_local_commit: Optional[Callable[[str], None]],
    on_staged: Optional[Callable[[int], None]],
) -> int:
    """Write every array according to its plan and start the asynchronous commit.

    Returns once this process has copied its data out. ``manager`` joins the commits and
    barriers on the other processes.

    Returns:
        The peak host bytes staged by this process.
    """
    manager.wait_until_finished()

    file_io_locking, writer_lease = _prepare_file_io_locking(config, checkpoint_dir)
    handed_off = False
    try:
        commit_order = _prepare_commit_order(checkpoint_dir)

        # JAX's process-lifetime context accumulates caches across saves, since each save writes a
        # new OCDBT database (#6785). Give each save bounded caches and copy concurrency of its own.
        context = _tensorstore_write_context(config, file_io_locking)
        jobs = _checkpoint_writes(arrays, tspecs, plans)
        # Each process chooses from its own share; both lock-file paths reach one metadata barrier.
        use_waves = file_io_locking == "lockfile" and sum(job.num_bytes for job in jobs) > config.max_staged_host_bytes
        if use_waves:
            gate, commit_futures = _write_in_manifest_waves(
                jobs, tspecs, config, checkpoint_dir, context, commit_order
            )
        else:
            # A fitting multi-process save still commits each chunk on its own, and those commits
            # replace the manifest while another process may be opening it.
            published_metadata = file_io_locking == "lockfile" and jax.process_count() > 1 and bool(tspecs)
            if published_metadata:
                _publish_array_metadata(checkpoint_dir, context, tspecs, commit_order)
            gate, commit_futures = _write_each_chunk(
                jobs,
                tspecs,
                config,
                context,
                checkpoint_dir,
                commit_order,
                published_metadata=published_metadata,
            )
        if writer_lease is not None:
            _release_lease_when_commits_finish(writer_lease, commit_futures)
            handed_off = True
        logger.info(
            "Checkpoint staged %.2f GiB peak against a %.2f GiB budget across %d writes "
            "in %d manifest commits (about %.0f GiB of host memory while in flight)",
            gate.peak_bytes / 1024**3,
            config.max_staged_host_bytes / 1024**3,
            len(jobs),
            len(commit_futures),
            _STAGED_BYTE_OVERHEAD * gate.peak_bytes / 1024**3,
        )
        staged_host_bytes = gate.peak_bytes

        _trim_host_memory_after_commits(commit_futures)

        # Private to AsyncManager. Its own `serialize` calls these.
        manager._add_futures(commit_futures)
        if on_staged is not None:
            on_staged(staged_host_bytes)
        if on_local_commit is not None:
            remaining = len(commit_futures)
            callback_lock = threading.Lock()
            local_status = "local_completed"

            def local_write_finished(future: ts.Future) -> None:
                nonlocal remaining, local_status
                try:
                    future.result()
                    failed = False
                except BaseException:
                    failed = True
                with callback_lock:
                    if failed:
                        local_status = "local_failed"
                    remaining -= 1
                    if remaining == 0:
                        on_local_commit(local_status)

            for future in commit_futures:
                future.add_done_callback(local_write_finished)
            if not commit_futures:
                on_local_commit(local_status)
        manager._start_async_commit(commit_callback)
        return staged_host_bytes
    finally:
        if writer_lease is not None and not handed_off:
            writer_lease.release()


def _sharding_from_leaf(leaf, axis_mapping, mesh) -> Optional[jax.sharding.Sharding]:
    def _concretize_sharding(sharding: jax.sharding.Sharding) -> jax.sharding.Sharding:
        # `eqx.filter_eval_shape` can produce `NamedSharding(mesh=AbstractMesh(...))`, but JAX array
        # deserialization requires a concrete device assignment (i.e., a concrete Mesh).
        if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(sharding.mesh, jax.sharding.AbstractMesh):
            concrete_mesh = mesh or hax.partitioning._get_mesh()
            if isinstance(concrete_mesh, jax.sharding.AbstractMesh) or concrete_mesh is None or concrete_mesh.empty:
                # Fall back to JAX's concrete mesh getter when available.
                concrete_mesh = get_concrete_mesh()

            if concrete_mesh is not None and not concrete_mesh.empty:
                # Preserve memory_kind: an offloaded run's leaves carry `pinned_host`, and dropping it
                # here would silently reload the state into device memory.
                return jax.sharding.NamedSharding(concrete_mesh, sharding.spec, memory_kind=sharding.memory_kind)
        return sharding

    if is_named_array(leaf):
        if not is_jax_array_like(leaf.array):
            return None
        return hax.partitioning.sharding_for_axis(leaf.axes, axis_mapping, mesh)
    elif hasattr(leaf, "sharding") and getattr(leaf, "sharding") is not None:
        return _concretize_sharding(leaf.sharding)
    elif is_jax_array_like(leaf):
        return _fully_replicated_sharding(mesh)
    elif isinstance(leaf, (bool, float, complex, int, np.ndarray)):
        return _fully_replicated_sharding(mesh)
    else:
        logger.warning(f"Unknown leaf type {type(leaf)}")
        return None


def _fully_replicated_sharding(mesh):
    return hax.partitioning.sharding_for_axis((), {}, mesh)


def _replica_staging_sharding(sharding: Sharding, shape: tuple[int, ...]) -> NamedSharding | None:
    """Expose uniform replicas on a leading axis in the platform's default compute memory, if any."""
    if not isinstance(sharding, NamedSharding):
        return None
    if any(device.platform == "tpu" for device in sharding.device_set):
        # JAX 0.11 aborts when a TPU collective has a pinned-host input or output.
        return None

    used_axes = set()
    for partition in sharding.spec:
        if isinstance(partition, str):
            used_axes.add(partition)
        elif isinstance(partition, tuple):
            used_axes.update(partition)
    replica_axes = tuple(
        axis for axis in sharding.mesh.axis_names if axis not in used_axes and sharding.mesh.shape[axis] > 1
    )
    if not replica_axes:
        return None

    replica_count = math.prod(sharding.mesh.shape[axis] for axis in replica_axes)
    if _uniform_replica_count(sharding, shape) != replica_count:
        return None

    staging_spec = PartitionSpec(replica_axes, *sharding.spec)
    return NamedSharding(sharding.mesh, staging_spec)


def _replica_count(staging_sharding: NamedSharding) -> int:
    replica_axes = staging_sharding.spec[0]
    if isinstance(replica_axes, str):
        replica_axes = (replica_axes,)
    return math.prod(staging_sharding.mesh.shape[axis] for axis in replica_axes)


@dataclass(frozen=True)
class _LeafReadPlan:
    path: str
    store: ts.TensorStore
    sharding: Sharding
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    shard_shape: tuple[int, ...]
    temporary_bytes: int
    staging_sharding: NamedSharding | None
    replica_count: int


@dataclass(frozen=True)
class _RestoreReadResult:
    leaves: list[jax.Array]
    store_bytes: int
    peak_bytes: int


async def _leaf_read_plan(
    path: str,
    sharding: Sharding,
    tensorstore_spec: dict,
    *,
    context: ts.Context,
    config: TensorStoreReadConfig,
) -> _LeafReadPlan:
    store = await ts.open(tensorstore_spec, open=True, context=context)
    shape = tuple(store.shape)
    dtype = np.dtype(jax.dtypes.canonicalize_dtype(store.dtype.numpy_dtype))
    shard_shape = tuple(sharding.shard_shape(shape))
    shard_bytes = math.prod(shard_shape) * np.dtype(dtype).itemsize
    max_read_bytes = shard_bytes
    seen_indices = set()
    # Use the maximum over global shard indices so every process forms identical batches.
    for index in sharding.devices_indices_map(shape).values():
        hashable_index = _hashable_index(index)
        if hashable_index in seen_indices:
            continue
        seen_indices.add(hashable_index)
        requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
        restricted_domain = store.domain.intersect(requested_domain)
        max_read_bytes = max(max_read_bytes, ts_impl.estimate_read_memory_footprint(store, restricted_domain))
    if store.dtype.numpy_dtype != dtype:
        max_read_bytes += shard_bytes

    staging_sharding = None
    if config.replica_mode == ReplicaRestoreMode.ONE_REPLICA:
        staging_sharding = _replica_staging_sharding(sharding, shape)
    replica_count = _replica_count(staging_sharding) if staging_sharding is not None else 1
    temporary_bytes = max_read_bytes
    if staging_sharding is not None:
        # The collective input remains live after the pageable TensorStore destination is copied.
        temporary_bytes += shard_bytes
    return _LeafReadPlan(
        path=path,
        store=store,
        sharding=sharding,
        shape=shape,
        dtype=dtype,
        shard_shape=shard_shape,
        temporary_bytes=temporary_bytes * len(sharding.addressable_devices),
        staging_sharding=staging_sharding,
        replica_count=replica_count,
    )


async def _read_store_shard(plan: _LeafReadPlan, index, host_array: np.ndarray) -> int:
    """Read one shard into host memory and return the TensorStore bytes fetched."""
    requested_domain = ts.IndexTransform(input_shape=plan.shape)[index].domain
    restricted_domain = plan.store.domain.intersect(requested_domain)
    if plan.store.dtype.numpy_dtype == plan.dtype:
        destination = host_array
    else:
        destination = np.empty(plan.shard_shape, dtype=plan.store.dtype.numpy_dtype)
    await ts.array(destination)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(
        plan.store[restricted_domain]
    )
    if destination is not host_array:
        host_array[...] = destination.astype(plan.dtype)
    return restricted_domain.size * np.dtype(plan.store.dtype.numpy_dtype).itemsize


async def _stage_leaf(plan: _LeafReadPlan) -> tuple[jax.Array, int]:
    target_indices = plan.sharding.addressable_devices_indices_map(plan.shape)
    staging_sharding = plan.staging_sharding
    replica_indices_by_device = {}
    if staging_sharding is None:
        local_shape = plan.shard_shape
        output_shape = plan.shape
        output_sharding = plan.sharding
    else:
        output_shape = (plan.replica_count, *plan.shape)
        output_sharding = staging_sharding
        for device, index in staging_sharding.addressable_devices_indices_map(output_shape).items():
            assert index is not None
            replica_indices_by_device[device] = index[0]
        local_shape = tuple(staging_sharding.shard_shape(output_shape))
        assert local_shape == (1, *plan.shard_shape)

    async def stage_shard(device: jax.Device, index, replica_index) -> tuple[jax.Array, int]:
        reader = True
        if replica_index is not None:
            assert isinstance(replica_index, slice)
            replica_start = replica_index.start or 0
            assert replica_index.stop - replica_start == 1
            shard_key = f"{plan.path}:{_hashable_index(index)!r}".encode()
            reader = replica_start == zlib.crc32(shard_key) % plan.replica_count

        host_array = np.zeros(local_shape, dtype=plan.dtype)
        store_bytes = 0
        if reader:
            destination = host_array if staging_sharding is None else host_array.reshape(plan.shard_shape)
            store_bytes = await _read_store_shard(plan, index, destination)
        target = SingleDeviceSharding(device, memory_kind=output_sharding.memory_kind)
        array = jax.device_put(host_array, target)
        array.block_until_ready()
        return array, store_bytes

    shard_args = [(device, index, replica_indices_by_device.get(device)) for device, index in target_indices.items()]
    staged = await asyncio.gather(*(stage_shard(*args) for args in shard_args))
    shards, store_bytes = zip(*staged)
    array = jax.make_array_from_single_device_arrays(output_shape, output_sharding, list(shards))
    return array, sum(store_bytes)


def _restore_replica_axis(value: jax.Array) -> jax.Array:
    bits = jax.lax.bitcast_convert_type(value, jnp.uint8)
    # Exactly one replica contributes each value, so bytewise sum preserves every checkpoint bit.
    bits = jnp.sum(bits, axis=0, dtype=bits.dtype)
    return jax.lax.bitcast_convert_type(bits, value.dtype)


@lru_cache(maxsize=16)
def _replica_reducer(staging_sharding: NamedSharding, target_sharding: Sharding):
    return jax.jit(_restore_replica_axis, in_shardings=staging_sharding, out_shardings=target_sharding)


def _finish_leaf(plan: _LeafReadPlan, staged: jax.Array) -> jax.Array:
    if plan.staging_sharding is None:
        return staged
    # The jit stages pinned-host inputs through device memory for the all-reduce, then honors
    # the original memory kind on output. Only this leaf's shard needs to enter device memory.
    restored = _replica_reducer(plan.staging_sharding, plan.sharding)(staged)
    restored.block_until_ready()
    return restored


def _read_batches(plans: Sequence[_LeafReadPlan], limit_bytes: int) -> list[list[_LeafReadPlan]]:
    batches: list[list[_LeafReadPlan]] = []
    current: list[_LeafReadPlan] = []
    current_bytes = 0
    for plan in plans:
        if current and current_bytes + plan.temporary_bytes > limit_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(plan)
        current_bytes += plan.temporary_bytes
    if current:
        batches.append(current)
    return batches


def _deserialize_leaves(
    paths: list[str],
    shardings: list[Sharding],
    tensorstore_specs: list[dict],
    config: TensorStoreReadConfig,
) -> list:
    """Restore leaves into their requested shardings and memory kinds."""
    context = _tensorstore_read_context(config)

    async def read_all():
        plans = await asyncio.gather(
            *(
                _leaf_read_plan(path, sharding, spec, context=context, config=config)
                for path, sharding, spec in zip(paths, shardings, tensorstore_specs)
            )
        )
        leaves = []
        store_bytes = 0
        peak_bytes = 0
        # Every process completes these collectives in path order. Issuing them as soon as each
        # asynchronous read finishes would let ranks enter different all-reduces and deadlock.
        for batch in _read_batches(plans, config.max_in_flight_bytes):
            staged = await asyncio.gather(*(_stage_leaf(plan) for plan in batch))
            peak_bytes = max(peak_bytes, sum(plan.temporary_bytes for plan in batch))
            staged_arrays: list[jax.Array | None] = [array for array, _ in staged]
            staged_store_bytes = [leaf_store_bytes for _, leaf_store_bytes in staged]
            del staged
            for index, plan in enumerate(batch):
                array = staged_arrays[index]
                assert array is not None
                leaves.append(_finish_leaf(plan, array))
                staged_arrays[index] = None
                store_bytes += staged_store_bytes[index]
        return _RestoreReadResult(leaves=leaves, store_bytes=store_bytes, peak_bytes=peak_bytes)

    started = time.time()
    result = asyncio.run(read_all())
    elapsed = time.time() - started
    materialized_bytes = sum(shard.data.nbytes for leaf in result.leaves for shard in leaf.addressable_shards)
    logger.info(
        "Restore read %s from TensorStore and materialized %s across %d arrays in %.1fs "
        "(%.2f GiB/s), peak %s of a %s budget",
        _format_gib(result.store_bytes),
        _format_gib(materialized_bytes),
        len(result.leaves),
        elapsed,
        result.store_bytes / 1024**3 / max(elapsed, 1e-9),
        _format_gib(result.peak_bytes),
        _format_gib(config.max_in_flight_bytes),
    )
    return result.leaves


def _restore_ocdbt(
    checkpoint_root: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    read_config: TensorStoreReadConfig,
    allow_missing: bool,
) -> tuple[list, list[int]]:
    """Restore arrays from an OCDBT checkpoint."""
    manifest = read_manifest(checkpoint_root)
    if manifest is not None:
        # Listing the kvstore walks every chunk key.
        present = manifest.array_paths
    else:
        keys = asyncio.run(_list_ocdbt_keys(checkpoint_root))
        present = {key[: -len("/zarr.json")] for key in keys if key.endswith("/zarr.json")}

    paths_to_load = []
    indices_to_load = []
    shardings_to_load = []
    missing_paths = []
    missing_indices = []

    for i in real_indices:
        path = paths[i]

        if path not in present:
            missing_paths.append(path)
            missing_indices.append(i)
            continue

        paths_to_load.append(path)
        indices_to_load.append(i)
        shardings_to_load.append(shardings_leaves[i])

    # Check for missing paths
    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing {len(missing_paths)} arrays in OCDBT checkpoint {checkpoint_root}: {missing_paths}."
                f" The checkpoint holds {sorted(present)}"
            )
        else:
            to_log = f"Several keys were missing from the OCDBT checkpoint {checkpoint_root}:"
            leaf_paths = jtu.tree_leaves(leaf_key_paths, is_leaf=_is_named_or_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    tspecs_to_load = []
    for path in paths_to_load:
        spec = _create_ocdbt_spec(checkpoint_root, path)
        tspecs_to_load.append(spec)

    deser_leaves = _deserialize_leaves(paths_to_load, shardings_to_load, tspecs_to_load, read_config)
    return deser_leaves, indices_to_load


def _restore_old_ts(
    checkpoint_dir: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    read_config: TensorStoreReadConfig,
    allow_missing: bool,
) -> tuple[list, list[int]]:
    """Restore arrays from an old (non-OCDBT) tensorstore checkpoint."""
    paths = [prefix_join(checkpoint_dir, p) for p in paths]

    paths_to_load = []
    indices_to_load = []
    shardings_to_load = []

    missing_paths = []
    missing_indices = []

    for i in real_indices:
        path = paths[i]

        if not StoragePath(path).exists():
            missing_paths.append(path)
            missing_indices.append(i)
            continue

        paths_to_load.append(path)
        indices_to_load.append(i)
        shardings_to_load.append(shardings_leaves[i])

    # Check for missing paths
    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(f"Missing paths: {missing_paths}")
        else:
            to_log = f"Several keys were missing from the checkpoint directory {checkpoint_dir}:"
            leaf_paths = jtu.tree_leaves(leaf_key_paths, is_leaf=_is_named_or_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    tspecs_to_load = [array_ser.get_tensorstore_spec(path) for path in paths_to_load]
    deser_leaves = _deserialize_leaves(paths_to_load, shardings_to_load, tspecs_to_load, read_config)
    return deser_leaves, indices_to_load


def tree_deserialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    axis_mapping: Optional[ResourceMapping] = None,
    mesh: Optional[Mesh] = None,
    *,
    allow_missing: bool = False,
    read_config: Optional[TensorStoreReadConfig] = None,
):
    """Deserialize a checkpoint into the shape of ``pytree``.

    ``pytree`` may hold ShapeDtypeStructs from ``eval_shape``; ``axis_mapping`` and ``mesh``
    supply the shardings. ``allow_missing`` keeps absent leaves as they are.
    """
    read_config = read_config or TensorStoreReadConfig()

    # Pre-charge the cross-region budget from the exemplar pytree, an upper bound under
    # `allow_missing=True`. See the save path.
    estimated_bytes = sum(
        _estimate_array_nbytes(leaf.array if is_named_array(leaf) else leaf)
        for leaf in jtu.tree_leaves(pytree, is_leaf=_is_named_or_none)
        if leaf is not None
    )
    record_transfer(estimated_bytes, checkpoint_dir)

    shardings: PyTree[Optional[Sharding]] = jtu.tree_map(
        partial(_sharding_from_leaf, axis_mapping=axis_mapping, mesh=mesh), pytree, is_leaf=_is_named_or_none
    )

    # TODO: support ShapeDtypeStructs that are not NamedArrays
    leaf_key_paths = jax_utils.leaf_key_paths(shardings, is_leaf=_is_named_or_none)
    paths = jtu.tree_map(lambda kp: "/".join(kp.split(".")), leaf_key_paths)
    paths = jtu.tree_leaves(paths, is_leaf=lambda x: x is None)

    shardings_leaves, shardings_structure = jtu.tree_flatten(shardings, is_leaf=_is_named_or_none)

    assert len(shardings_leaves) == len(paths)
    # ok, so, jax really doesn't want any Nones in the leaves here, so we need to temporarily partition the pytree
    real_indices = [i for i, x in enumerate(shardings_leaves) if x is not None]

    # The checkpoint code has munged our paths to add the subpath in explicitly to the `checkpoint_dir`.
    # For OCDBT, we need to determine the actual root and then adjust the requests tensor paths accordingly.
    def find_checkpoint_root(path):
        """Find the checkpoint root by looking for metadata.json"""
        current = path
        while current and current != os.path.dirname(current):
            metadata_path = prefix_join(current, "metadata.json")
            if StoragePath(metadata_path).exists():
                return current
            current = os.path.dirname(current)
        return path  # fallback to original path

    checkpoint_root = find_checkpoint_root(checkpoint_dir)
    manifest = read_manifest(checkpoint_root)
    if manifest is not None:
        if manifest.array_driver != ARRAY_DRIVER:
            raise ValueError(
                f"Checkpoint {checkpoint_root} stores arrays with the {manifest.array_driver!r} driver, "
                f"but this build of levanter reads {ARRAY_DRIVER!r}."
            )
        is_ocdbt_checkpoint = manifest.kvstore_driver == KVSTORE_DRIVER
    else:
        # A manifest-less checkpoint predates the manifest. Sniff the OCDBT database itself.
        is_ocdbt_checkpoint = StoragePath(prefix_join(checkpoint_root, "manifest.ocdbt")).exists()

    if is_ocdbt_checkpoint:
        subpath = os.path.relpath(checkpoint_dir, start=find_checkpoint_root(checkpoint_dir))
        if subpath != ".":
            logger.info("Adjusting paths for OCDBT checkpoint with subpath: %s", subpath)
            paths = [os.path.join(subpath, p) for p in paths]
        deser_leaves, indices_to_load = _restore_ocdbt(
            checkpoint_root, paths, real_indices, shardings_leaves, leaf_key_paths, read_config, allow_missing
        )
    else:
        deser_leaves, indices_to_load = _restore_old_ts(
            checkpoint_dir, paths, real_indices, shardings_leaves, leaf_key_paths, read_config, allow_missing
        )

    # now we need to recreate the original structure
    out_leaves = jax.tree.leaves(pytree, is_leaf=_is_named_or_none)
    assert len(out_leaves) == len(shardings_leaves)
    # out_leaves = [None] * len(shardings_leaves)
    for i, x in zip(indices_to_load, deser_leaves):
        out_leaves[i] = x

    deser_arrays = jtu.tree_unflatten(shardings_structure, out_leaves)

    # deser_arrays only has arrays for the deserialized arrays, but we need named arrays for at least some.
    # The original pytree has the structure we want, so we'll use that to rebuild the named arrays
    def _rebuild_named_array(like, array):
        if is_named_array(array):
            return array

        if is_named_array(like):
            return hax.NamedArray(array, like.axes)
        else:
            return array

    return jtu.tree_map(_rebuild_named_array, pytree, deser_arrays, is_leaf=_is_named_or_none)
