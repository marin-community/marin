# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host snapshot and rebind of the full state before the first train step."""

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from jax._src.array import ArrayImpl


@dataclass
class HostArray:
    aval: object
    sharding: object
    committed: bool
    shards: list


@dataclass(frozen=True)
class HostRebindConfig:
    """Optional first-step host rebind and memory observation location."""

    cgroup_path: str = "/sys/fs/cgroup"
    post_rebind_barrier_timeout_seconds: float = 1800.0


def state_snapshot(state):
    leaves, tree = jax.tree.flatten(state)
    saved = []
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            shards = [(shard.data.sharding, np.array(shard.data, copy=True)) for shard in leaf.addressable_shards]
            saved.append(HostArray(leaf.aval, leaf.sharding, leaf.committed, shards))
        else:
            saved.append(leaf)
    return tree, saved


def restore_state(snapshot):
    tree, leaves = snapshot
    values = []
    for leaf in leaves:
        if isinstance(leaf, HostArray):
            # Public constructors change committed or weak-type metadata, checked below.
            buffers = [jax.device_put(data, sharding) for sharding, data in leaf.shards]
            leaf = ArrayImpl(leaf.aval, leaf.sharding, buffers, committed=leaf.committed)
        values.append(leaf)
    return jax.block_until_ready(jax.tree.unflatten(tree, values))


def signature(tree):
    rows = []
    paths, definition = jax.tree.flatten_with_path(tree)
    for path, leaf in paths:
        key = jax.tree_util.keystr(path)
        if isinstance(leaf, jax.Array):
            rows.append(
                (
                    key,
                    tuple(leaf.shape),
                    str(leaf.dtype),
                    str(leaf.sharding),
                    leaf.sharding.memory_kind,
                    leaf.committed,
                    leaf.weak_type,
                    tuple((shard.device.id, repr(shard.index)) for shard in leaf.addressable_shards),
                )
            )
        else:
            rows.append((key, type(leaf).__name__, repr(leaf)))
    return str(definition), rows


def buffer_layout(state, batch):
    shards = []
    for prefix, tree in (("state", state), ("batch", batch)):
        paths, _ = jax.tree.flatten_with_path(tree)
        for path, leaf in paths:
            if isinstance(leaf, jax.Array):
                for shard in leaf.addressable_shards:
                    label = prefix + jax.tree_util.keystr(path)
                    shards.append((label, shard.device.id, shard.data))

    pending = [path for path, _, shard in shards if not shard.is_ready()]
    by_buffer = defaultdict(list)
    for path, device, shard in shards:
        pointer = shard.unsafe_buffer_pointer()
        if pointer:
            by_buffer[device, pointer].append(path)
    aliases = [{"device": device, "leaves": paths} for (device, _), paths in by_buffer.items() if len(paths) > 1]
    return {"local_shards": len(shards), "pending": pending, "alias_groups": aliases}


def task_cgroup_sample(path: Path) -> dict[str, int | str | bool | None]:
    """Report rank RSS and cgroup memory without assuming cgroup membership."""
    sample = {
        "rank_rss_bytes": None,
        "rank_rss_error": None,
        "cgroup_path": str(path),
        "cgroup_current_bytes": None,
        "cgroup_limit_bytes": None,
        "cgroup_error": None,
        "task_scope_verified": False,
        "process_index": jax.process_index(),
        "pid": os.getpid(),
    }
    try:
        sample["rank_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError) as error:
        sample["rank_rss_error"] = type(error).__name__
    try:
        sample["cgroup_current_bytes"] = int((path / "memory.current").read_text().strip())
        maximum = (path / "memory.max").read_text().strip()
        sample["cgroup_limit_bytes"] = None if maximum == "max" else int(maximum)
    except (OSError, ValueError) as error:
        sample["cgroup_error"] = type(error).__name__
    return sample


def rebind_first_state(state, batch, *, cgroup_path: Path, observe):
    """Snapshot, release and restore state while sampling memory.

    Cgroup membership and task-wide scope must be verified independently.
    """
    initial = signature((state, batch))
    layout = buffer_layout(state, batch)
    if any(
        any(p.startswith("state") for p in group["leaves"]) and any(p.startswith("batch") for p in group["leaves"])
        for group in layout["alias_groups"]
    ):
        raise RuntimeError("state and batch share a buffer")
    observe("before_snapshot", task_cgroup_sample(cgroup_path))
    snapshot = state_snapshot(state)
    sizes = [data.nbytes for leaf in snapshot[1] if isinstance(leaf, HostArray) for _, data in leaf.shards]
    sample = task_cgroup_sample(cgroup_path)
    sample.update(
        snapshot_bytes=sum(sizes),
        snapshot_shards=len(sizes),
        largest_shard_bytes=max(sizes, default=0),
    )
    observe("after_snapshot", sample)
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, jax.Array) and not leaf.is_deleted():
            leaf.delete()
    if any(isinstance(leaf, jax.Array) and not leaf.is_deleted() for leaf in jax.tree.leaves(state)):
        raise RuntimeError("state buffers survived deletion")
    restored = restore_state(snapshot)
    del snapshot
    if signature((restored, batch)) != initial:
        raise ValueError("state restore changed the actual stock input signature")
    return restored
