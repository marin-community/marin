# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import collections
import logging
import math
import os
import urllib.parse
import zlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence

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
from jax.sharding import Mesh, Sharding
from jaxtyping import PyTree

from rigging.filesystem.cross_region import record_transfer
from rigging.filesystem.storage_path import StoragePath, prefix_join

from levanter._debug_logging import flush_debug_output
from levanter.checkpoint_manifest import CheckpointArray, build_manifest, read_manifest, write_manifest
from levanter.utils import jax_utils

logger = logging.getLogger(__name__)

ARRAY_DRIVER = "zarr3"
KVSTORE_DRIVER = "ocdbt"
# JAX's memory kind for host memory a device can address. Offloaded optimizer state lives here.
_HOST_MEMORY_KIND = "pinned_host"
# Chunks a save stages at once. Four processes at 32 GiB each exhausted a GB200 node.
_DEFAULT_STAGED_CHUNKS = 32
# Host memory a staged byte occupies while its write is in flight, for reporting only.
_STAGED_BYTE_OVERHEAD = 4


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

        # Supplying credentials explicitly reduces noisy AWS CRT logs in containers.
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            spec["aws_credentials"] = {"type": "environment"}

        return spec
    elif parsed.scheme == "gs":
        return {"driver": "gcs", "bucket": parsed.netloc, "path": parsed.path.lstrip("/")}
    elif parsed.scheme in ("", "file"):
        return {"driver": "file", "path": os.path.abspath(path)}
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
    """Snapshot a shard into pageable host memory, restricted to ``local_slice``.

    The TPU runtime never returns JAX's pinned staging to the OS (#6924). State already in
    host memory is snapshotted in place.
    """
    data = shard.data if local_slice is None else _slice_shard_on_device(shard.data, *local_slice)
    if getattr(data.sharding, "memory_kind", None) != _HOST_MEMORY_KIND:
        data.copy_to_host_async()
        # Yield so the remaining shards' copies can be enqueued before this one blocks.
        await asyncio.sleep(0)
    # TensorStore may retain this array. It must not reference the buffer training donates next.
    return np.array(data, copy=True)


@dataclass(frozen=True)
class TensorStoreWriteConfig:
    """How a checkpoint save divides work across the processes that hold the state."""

    max_write_replicas: int = 64
    """Cap on how many replicas of an array write part of it. 1 disables replica splitting."""

    min_replica_slice_bytes: int = 16 * 1024**2
    """Do not split an array when doing so would give each replica less than this."""

    max_chunk_bytes: int = 512 * 1024**2
    """Upper bound on one zarr3 chunk. Bounds the size of a single object store write."""

    max_staged_host_bytes: int = _DEFAULT_STAGED_CHUNKS * max_chunk_bytes
    """Host memory this process may hold in staged snapshots at once.

    A save whose share fits returns while the commits drain. A larger save rolls through.
    """

    def __post_init__(self) -> None:
        if self.max_write_replicas < 1:
            raise ValueError(f"max_write_replicas must be at least 1, got {self.max_write_replicas}")
        for name in ("min_replica_slice_bytes", "max_chunk_bytes", "max_staged_host_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


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


class _HostStagingGate:
    """Track staged bytes until TensorStore has committed and released them.

    Shard writes pass ``can_reference_source_data_indefinitely=True``. The copy future
    resolves while the snapshot is still referenced.
    """

    def __init__(self, limit_bytes: int):
        self._limit = limit_bytes
        self._in_flight = 0
        self._peak = 0
        self._released = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def peak_bytes(self) -> int:
        return self._peak

    async def acquire(self, num_bytes: int) -> None:
        # Built before the save's loop exists, so bind on first use.
        self._loop = asyncio.get_running_loop()
        # A snapshot larger than the whole budget proceeds alone; it can never be admitted.
        while self._in_flight and self._in_flight + num_bytes > self._limit:
            self._released.clear()
            await self._released.wait()
        self._in_flight += num_bytes
        self._peak = max(self._peak, self._in_flight)

    def release(self, num_bytes: int) -> None:
        """Callable from any thread; TensorStore resolves commits off the loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._release_on_loop, num_bytes)

    def _release_on_loop(self, num_bytes: int) -> None:
        # Every mutation lands on the loop thread, so acquire never observes a partial update.
        self._in_flight -= num_bytes
        self._released.set()


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
    debug_checkpointer: bool = False,
    write_config: Optional[TensorStoreWriteConfig] = None,
):
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
        flush_debug_output(logger)

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
        flush_debug_output(logger)

    _serialize_arrays(arrays, tspecs, plans, manager, write_config, commit_callback)

    if debug_checkpointer:
        logger.info("Checkpoint tensorstore serialize handed off async commit for %s", checkpoint_dir)
        flush_debug_output(logger)

    if manager_was_none:
        manager.wait_until_finished()


def _serialize_arrays(
    arrays: Sequence[Any],
    tspecs: Sequence[dict],
    plans: Sequence[_WritePlan],
    manager: array_ser.GlobalAsyncCheckpointManager,
    config: TensorStoreWriteConfig,
    commit_callback: Callable,
) -> None:
    """Write every array according to its plan and start the asynchronous commit.

    Returns once this process has copied its data out. ``manager`` joins the commits and
    barriers on the other processes.
    """
    manager.wait_until_finished()

    # JAX's process-lifetime ts.Context accumulates caches across saves, since each save
    # writes a new OCDBT database (#6785). Cloning the spec keeps its pools and limits.
    context = ts.Context(ts_impl._TS_CONTEXT.spec)
    gate = _HostStagingGate(config.max_staged_host_bytes)
    commit_futures: list[ts.Future] = []

    async def issue_write(num_bytes: int, stage):
        """Admit ``num_bytes`` to the staging budget, then stage and issue one write."""
        await gate.acquire(num_bytes)
        try:
            write = await stage()
        except BaseException:
            gate.release(num_bytes)
            raise
        commit_futures.append(write.commit)
        # Fires on failure as well as success, so a failed commit cannot strand the budget.
        write.commit.add_done_callback(lambda _: gate.release(num_bytes))
        await write.copy

    async def write_host_array(store, array):
        # Unsharded and identical everywhere, so process 0 writes all of it.
        if jax.process_index() != 0:
            return

        async def stage():
            # The caller owns this buffer and may mutate it, so TensorStore must copy.
            return store.write(np.asarray(array), can_reference_source_data_indefinitely=False)

        await issue_write(_estimate_array_nbytes(array), stage)

    async def write_shard(store, shard, plan):
        region = _shard_write_region(shard, plan)
        if region is None:
            return

        async def stage():
            data = await _transfer_shard_to_pageable_host(shard, region.local_slice)
            # The snapshot is private and never mutated, so TensorStore may hold the reference.
            return store[region.index].write(data, can_reference_source_data_indefinitely=True)

        await issue_write(_estimate_array_nbytes(shard.data) // plan.write_replicas, stage)

    async def write_one(array, tspec, plan):
        store = await ts.open(ts.Spec(tspec), create=True, open=True, context=context)
        if not isinstance(array, jax.Array):
            await write_host_array(store, array)
            return
        await asyncio.gather(*(write_shard(store, shard, plan) for shard in array.addressable_shards))

    async def write_all():
        await asyncio.gather(*(write_one(a, s, p) for a, s, p in zip(arrays, tspecs, plans)))

    asyncio.run(write_all())
    logger.info(
        "Checkpoint staged %.2f GiB peak against a %.2f GiB budget across %d writes "
        "(about %.0f GiB of host memory while in flight)",
        gate.peak_bytes / 1024**3,
        config.max_staged_host_bytes / 1024**3,
        len(commit_futures),
        _STAGED_BYTE_OVERHEAD * gate.peak_bytes / 1024**3,
    )

    # Private to AsyncManager. Its own `serialize` calls these.
    manager._add_futures(commit_futures)
    manager._start_async_commit(commit_callback)


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
                return jax.sharding.NamedSharding(concrete_mesh, sharding.spec)
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


def _restore_ocdbt(
    checkpoint_root: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    manager: array_ser.GlobalAsyncCheckpointManager,
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

    deser_leaves = manager.deserialize(shardings=shardings_to_load, tensorstore_specs=tspecs_to_load)
    return deser_leaves, indices_to_load


def _restore_old_ts(
    checkpoint_dir: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    manager: array_ser.GlobalAsyncCheckpointManager,
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

    deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load, concurrent_gb=300)
    return deser_leaves, indices_to_load


def tree_deserialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    axis_mapping: Optional[ResourceMapping] = None,
    mesh: Optional[Mesh] = None,
    manager: Optional[array_ser.GlobalAsyncCheckpointManager] = None,
    *,
    allow_missing: bool = False,
):
    """Deserialize a checkpoint into the shape of ``pytree``.

    ``pytree`` may hold ShapeDtypeStructs from ``eval_shape``; ``axis_mapping`` and ``mesh``
    supply the shardings. ``allow_missing`` keeps absent leaves as they are.
    """
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()

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
            checkpoint_root, paths, real_indices, shardings_leaves, leaf_key_paths, manager, allow_missing
        )
    else:
        deser_leaves, indices_to_load = _restore_old_ts(
            checkpoint_dir, paths, real_indices, shardings_leaves, leaf_key_paths, manager, allow_missing
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
