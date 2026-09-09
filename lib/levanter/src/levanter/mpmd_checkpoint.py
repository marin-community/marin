# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard-local checkpoints for JaxPP MPMD array trees."""

import json
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from levanter.checkpoint import discover_latest_checkpoint, load_checkpoint
from levanter.checkpoint import save_checkpoint as save_levanter_checkpoint
from levanter.tensorstore_serialization import (
    ReplicaRestoreMode,
    TensorStoreReadConfig,
)
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath, prefix_join

try:
    import jaxpp.api as jaxpp  # pyrefly: ignore[missing-import]  # Optional pipeline extra.
except ModuleNotFoundError as error:
    if error.name != "jaxpp":
        raise
    jaxpp = None

State = TypeVar("State")

_FORMAT_VERSION = 1
_METADATA_FILE = "metadata.json"
_LATEST_FILE = "latest.json"


def checkpoint_arrays(state: State) -> State:
    """Expose an array tree's device buffers without moving any shards."""

    def unwrap(value):
        if jaxpp is None or not isinstance(value, jaxpp.MpmdArray):
            return value
        local = value.to_mpmd_local_array
        arrays = [] if local is None else local if isinstance(local, list) else [local]
        buffers = {shard.device: shard.data for array in arrays for shard in array.addressable_shards}
        return jax.make_array_from_single_device_arrays(
            value.shape,
            value.sharding,
            [buffers[device] for device in value.sharding.mesh.devices.flat if device in buffers],
            dtype=value.dtype,
        )

    return jax.tree.map(unwrap, state)


def _array_layout(state) -> list[dict]:
    return [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "spec": list(value.sharding.spec),
            "mesh": dict(value.sharding.mesh.shape),
            "processes": [device.process_index for device in value.sharding.mesh.devices.flat],
        }
        for path, value in jax.tree_util.tree_flatten_with_path(state)[0]
    ]


def save_checkpoint(root: str, state: State, *, step: int, metadata: dict) -> str:
    """Commit a completed training step after every process finishes its shard writes."""
    identifier = multihost_utils.broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8))
    path = prefix_join(root, f"step-{step:012d}-{bytes(identifier).hex()}")
    arrays = checkpoint_arrays(state)
    checkpoint_metadata = {
        "mpmd_version": _FORMAT_VERSION,
        "user_metadata": metadata,
        "arrays": _array_layout(arrays),
    }

    def commit():
        with atomic_rename(prefix_join(root, _LATEST_FILE)) as temporary_path:
            StoragePath(temporary_path).write_text(json.dumps({"checkpoint": StoragePath(path).name, "step": step}))

    save_levanter_checkpoint(
        arrays, step, path, commit_callback=commit, is_temporary=False, metadata=checkpoint_metadata
    )
    return path


def restore_checkpoint(
    root: str,
    state: State,
    shardings,
    *,
    validate_metadata: Callable[[dict[str, Any]], None],
) -> tuple[State, int]:
    """Restore the newest committed step, or return fresh state for an empty root.

    The compiled step supplies the exact MPMD placement. Array layouts and
    process topology must match. The caller validates its application metadata
    before any array data is read.
    """
    latest_path = StoragePath(prefix_join(root, _LATEST_FILE))
    latest = None
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        metadata_path = StoragePath(root) / latest["checkpoint"] / _METADATA_FILE
    else:
        # Recover a fully written first checkpoint if publication of latest.json
        # was interrupted. Normal resumes need no checkpoint-directory listing.
        checkpoint_path = discover_latest_checkpoint(root)
        if checkpoint_path is None:
            return state, 0
        metadata_path = StoragePath(checkpoint_path) / _METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    if latest is not None and latest["step"] != metadata["step"]:
        raise ValueError(f"Checkpoint step disagrees with latest.json: {metadata_path}")
    arrays = checkpoint_arrays(state)
    expected_layout = json.loads(json.dumps(_array_layout(arrays)))
    if metadata["mpmd_version"] != _FORMAT_VERSION or metadata["arrays"] != expected_layout:
        raise ValueError(f"Checkpoint array layout or topology does not match: {metadata_path}")
    validate_metadata(metadata["user_metadata"])
    path = str(metadata_path.parent)
    # Levanter's reader expects at least one addressable shard per input array.
    # Other stages remain empty descriptors and never enter the read plan.
    local_arrays = jax.tree.map(
        lambda value: (
            jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
            if value.sharding.addressable_devices
            else None
        ),
        arrays,
    )
    restored = load_checkpoint(
        local_arrays,
        path,
        # Avoid restore collectives across processes that do not own this stage.
        read_config=TensorStoreReadConfig(replica_mode=ReplicaRestoreMode.EVERY_REPLICA),
    )
    restored = jax.tree.map(
        lambda original, loaded: original if loaded is None else loaded,
        arrays,
        restored,
        is_leaf=lambda value: value is None,
    )

    def wrap(value, target):
        if isinstance(target, NamedSharding):
            return value
        if jaxpp is None:
            raise ImportError("MPMD restore requires jaxpp")
        buffers = {shard.device: shard.data for shard in value.addressable_shards}
        local_arrays = []
        for mesh_id in sorted(target.mesh_ids):
            sharding = NamedSharding(target.mpmd_mesh.unstack[mesh_id], target.spec)
            if sharding.addressable_devices:
                local_arrays.append(
                    jax.make_array_from_single_device_arrays(
                        value.shape,
                        sharding,
                        [buffers[device] for device in sharding.mesh.devices.flat if device in buffers],
                        dtype=value.dtype,
                    )
                )
        return jaxpp.MpmdArray(local_arrays, target, shape=value.shape, dtype=value.dtype)

    return jax.tree.map(wrap, restored, shardings), int(metadata["step"])
