# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered host-memory snapshot of a training state.

The snapshot copies a sharded JAX train-state pytree off the accelerators into
raw host bytes so a freshly re-exec'd trainer process can resume in seconds
without touching the durable (GCS) checkpoint. It is deliberately transformation
free: leaves are pulled to host with ``jax.device_get`` and pushed back with
``jax.device_put`` using the target leaf's own sharding.

Scope: this is the single-node store. ``jax.device_get`` gathers each sharded
leaf into one host array, so save throughput is bounded by a single serialized
device-to-host copy and the whole state must fit in one host's RAM. The
multi-node / large-model tier must instead copy each host's *local* shards in
parallel (no gather) so save scales past one host; see the TODO in ``save``.

Layout under ``root`` (tmpfs under ``/dev/shm``)::

    root/
      latest            # {"generation": G, "slot": S} written last, atomically
      slot_0/{meta.json,data.bin}
      slot_1/{meta.json,data.bin}

Crash consistency comes from double buffering plus commit ordering: a save
writes the *inactive* slot (the one ``latest`` does not point at), flushes it,
then atomically replaces ``latest``. A crash before that replace leaves the
previous generation fully intact, so ``latest`` never names a torn slot.

Restore never carries a pickled treedef across the process boundary. The caller
passes a freshly built ``template`` state (same config → identical structure and
shardings); the snapshot supplies only the raw leaf bytes, positionally, and the
structure comes from the template.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass

import jax
import numpy as np
from jaxtyping import PyTree

from levanter.recovery.types import DEFAULT_TMPFS_DIR


logger = logging.getLogger(__name__)

_MAGIC = "levanter-host-snapshot-v1"
_NUM_SLOTS = 2
_POINTER_NAME = "latest"
_META_NAME = "meta.json"
_DATA_NAME = "data.bin"
# Per-run snapshots live under <base>/<_SNAPSHOT_SUBDIR>/<run_id>; the pin marker
# is a sibling <run_id>.root file.
_SNAPSHOT_SUBDIR = "levanter-recovery"


@dataclass(frozen=True)
class SaveStats:
    generation: int
    step: int
    num_leaves: int
    bytes_written: int
    seconds: float
    root: str


@dataclass(frozen=True)
class RestoreResult:
    generation: int
    step: int
    num_leaves: int
    bytes_read: int
    seconds: float


def _slot_dir(root: str, slot: int) -> str:
    return os.path.join(root, f"slot_{slot}")


@dataclass(frozen=True)
class _Pointer:
    generation: int
    slot: int


class HostSnapshot:
    """Manage a double-buffered host snapshot rooted at a directory."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        for slot in range(_NUM_SLOTS):
            os.makedirs(_slot_dir(root, slot), exist_ok=True)

    # -- introspection ---------------------------------------------------

    def _read_pointer(self) -> _Pointer | None:
        path = os.path.join(self.root, _POINTER_NAME)
        try:
            with open(path) as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return _Pointer(generation=int(raw["generation"]), slot=int(raw["slot"]))

    def _resolve_committed(self) -> tuple[_Pointer, dict] | None:
        """Return the newest slot whose metadata loads cleanly, if any.

        The pointer is authoritative, but fall back to scanning both slots so a
        damaged pointer cannot strand an otherwise-good snapshot.
        """
        candidates: list[tuple[int, int, dict]] = []
        pointer = self._read_pointer()
        ordered_slots = (
            [pointer.slot] + [s for s in range(_NUM_SLOTS) if s != pointer.slot]
            if pointer is not None
            else list(range(_NUM_SLOTS))
        )
        for slot in ordered_slots:
            meta = self._try_read_meta(slot)
            if meta is not None:
                candidates.append((int(meta["generation"]), slot, meta))
        if not candidates:
            return None
        generation, slot, meta = max(candidates, key=lambda c: c[0])
        return _Pointer(generation=generation, slot=slot), meta

    def _try_read_meta(self, slot: int) -> dict | None:
        path = os.path.join(_slot_dir(self.root, slot), _META_NAME)
        try:
            with open(path) as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if meta.get("magic") != _MAGIC or not meta.get("complete", False):
            return None
        data_path = os.path.join(_slot_dir(self.root, slot), _DATA_NAME)
        if not os.path.exists(data_path):
            return None
        return meta

    def exists(self) -> bool:
        return self._resolve_committed() is not None

    def latest_step(self) -> int | None:
        resolved = self._resolve_committed()
        return None if resolved is None else int(resolved[1]["step"])

    # -- save ------------------------------------------------------------

    def save(self, state: PyTree, *, step: int) -> SaveStats:
        """Copy ``state`` to host and commit it to the inactive slot."""
        start = time.perf_counter()
        pointer = self._read_pointer()
        generation = 1 if pointer is None else pointer.generation + 1
        # Write the slot the pointer does NOT currently name, so the live
        # snapshot survives a crash partway through this write.
        target_slot = (pointer.slot + 1) % _NUM_SLOTS if pointer is not None else 0

        leaves, _ = jax.tree_util.tree_flatten(state)
        # Single-node path: device_get GATHERS each sharded leaf onto one host, so
        # throughput is one serialized D2H copy and the state must fit in one host's
        # RAM. TODO: copy leaf.addressable_shards in parallel (no gather) so save
        # scales past one host and one D2H link (the multi-node / large-model tier).
        host_leaves = jax.device_get(leaves)

        slot_dir = _slot_dir(self.root, target_slot)
        data_path = os.path.join(slot_dir, _DATA_NAME)
        meta_leaves: list[dict] = []
        offset = 0
        with open(data_path, "wb") as f:
            for leaf in host_leaves:
                host = np.asarray(leaf)
                # np.ascontiguousarray promotes 0-d arrays to shape (1,), so record
                # the original shape/nbytes and use the contiguous copy only for bytes.
                np.ascontiguousarray(host).tofile(f)
                meta_leaves.append(
                    {
                        "shape": list(host.shape),
                        "dtype": str(np.dtype(host.dtype)),
                        "offset": offset,
                        "nbytes": int(host.nbytes),
                    }
                )
                offset += int(host.nbytes)
            f.flush()
            os.fsync(f.fileno())

        meta = {
            "magic": _MAGIC,
            "generation": generation,
            "step": int(step),
            "num_leaves": len(meta_leaves),
            "leaves": meta_leaves,
            "complete": True,
        }
        self._write_json_atomic(os.path.join(slot_dir, _META_NAME), meta)
        # Commit: flip the pointer last, atomically.
        self._write_json_atomic(
            os.path.join(self.root, _POINTER_NAME),
            {"generation": generation, "slot": target_slot},
        )

        seconds = time.perf_counter() - start
        stats = SaveStats(
            generation=generation,
            step=int(step),
            num_leaves=len(meta_leaves),
            bytes_written=offset,
            seconds=seconds,
            root=self.root,
        )
        logger.info(
            "Host snapshot gen=%d step=%d wrote %.2f GiB in %.2fs to slot %d (%s)",
            generation,
            step,
            offset / 2**30,
            seconds,
            target_slot,
            self.root,
        )
        return stats

    # -- restore ---------------------------------------------------------

    def restore_into(self, template: PyTree) -> tuple[PyTree, RestoreResult] | None:
        """Return ``template`` with its leaves overwritten from the snapshot.

        ``template`` must be a freshly built state with the same structure and
        shardings as the saved one. Returns ``None`` when no committed snapshot
        exists, so the caller falls back to init / durable checkpoint.
        """
        resolved = self._resolve_committed()
        if resolved is None:
            return None
        pointer, meta = resolved
        start = time.perf_counter()

        template_leaves, treedef = jax.tree_util.tree_flatten(template)
        meta_leaves = meta["leaves"]
        if len(template_leaves) != len(meta_leaves):
            raise ValueError(
                f"snapshot leaf count {len(meta_leaves)} != template leaf count "
                f"{len(template_leaves)}; structure mismatch (config changed?)"
            )

        data_path = os.path.join(_slot_dir(self.root, pointer.slot), _DATA_NAME)
        new_leaves = []
        bytes_read = 0
        with open(data_path, "rb") as f:
            for template_leaf, leaf_meta in zip(template_leaves, meta_leaves, strict=True):
                shape = tuple(leaf_meta["shape"])
                if shape != tuple(template_leaf.shape):
                    raise ValueError(f"snapshot leaf shape {shape} != template {tuple(template_leaf.shape)}")
                nbytes = int(leaf_meta["nbytes"])
                f.seek(int(leaf_meta["offset"]))
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise ValueError(f"snapshot data truncated: expected {nbytes} bytes, got {len(buf)}")
                # Reconstruct with the template's dtype/shape (positional bytes);
                # this sidesteps serializing exotic dtypes like bfloat16.
                host = np.frombuffer(buf, dtype=template_leaf.dtype).reshape(shape)
                new_leaves.append(jax.device_put(host, template_leaf.sharding))
                bytes_read += nbytes

        restored = jax.tree_util.tree_unflatten(treedef, new_leaves)
        jax.block_until_ready(restored)
        seconds = time.perf_counter() - start
        result = RestoreResult(
            generation=pointer.generation,
            step=int(meta["step"]),
            num_leaves=len(new_leaves),
            bytes_read=bytes_read,
            seconds=seconds,
        )
        logger.info(
            "Restored host snapshot gen=%d step=%d (%.2f GiB) in %.2fs from slot %d",
            pointer.generation,
            result.step,
            bytes_read / 2**30,
            seconds,
            pointer.slot,
        )
        return restored, result

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _write_json_atomic(path: str, payload: dict) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def estimate_state_nbytes(state: PyTree) -> int:
    """Sum ``nbytes`` over the array leaves of ``state`` (one snapshot copy)."""
    leaves, _ = jax.tree_util.tree_flatten(state)
    total = 0
    for leaf in leaves:
        total += int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize
    return total


def snapshot_root(run_id: str, tmpfs_base: str = DEFAULT_TMPFS_DIR) -> str:
    """Return the host-snapshot root for a run.

    Snapshots always live in tmpfs (host RAM); there is no spill tier. The path is
    deterministic from ``run_id``, so every restart of a run agrees on it without a
    pin marker. If a state's double buffer does not fit in tmpfs the save fails
    loudly on write, which is the intended signal that the tier is undersized.
    """
    return os.path.join(tmpfs_base, _SNAPSHOT_SUBDIR, run_id)


def remove_snapshot(run_id: str, tmpfs_base: str = DEFAULT_TMPFS_DIR) -> None:
    """Delete a run's snapshot root so a finished run frees its tmpfs.

    A missing root is ignored: nothing may have been saved yet.
    """
    shutil.rmtree(snapshot_root(run_id, tmpfs_base), ignore_errors=True)
