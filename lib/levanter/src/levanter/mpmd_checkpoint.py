# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard-local checkpoints for JaxPP MPMD array trees."""

from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import PyTree
from levanter.checkpoint import load_checkpoint
from levanter.tensorstore_serialization import (
    ReplicaRestoreMode,
    TensorStoreReadConfig,
)

try:
    import jaxpp.api as jaxpp  # pyrefly: ignore[missing-import]  # Optional pipeline extra.
except ModuleNotFoundError as error:
    if error.name != "jaxpp":
        raise
    jaxpp = None

State = TypeVar("State")


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


def unstack_checkpoint_layers(value: jax.Array) -> tuple[jax.Array, ...]:
    """Slice a replicated leading layer axis without work on nonowning processes."""
    spec = (*value.sharding.spec, None, None)
    assert spec[0] is None
    sharding = NamedSharding(value.sharding.mesh, PartitionSpec(spec[1]))
    return tuple(
        jax.make_array_from_single_device_arrays(
            value.shape[1:], sharding, [shard.data[index] for shard in value.addressable_shards], dtype=value.dtype
        )
        for index in range(value.shape[0])
    )


def stack_checkpoint_layers(values: tuple[jax.Array, ...], sharding: NamedSharding) -> jax.Array:
    """Reassemble layer buffers under a destination stage sharding."""
    buffers = [{shard.device: shard.data for shard in value.addressable_shards} for value in values]
    return jax.make_array_from_single_device_arrays(
        (len(values), *values[0].shape),
        sharding,
        [
            jnp.stack([buffer[device] for buffer in buffers])
            for device in sharding.mesh.devices.flat
            if device in buffers[0]
        ],
        dtype=values[0].dtype,
    )


def restore_checkpoint(state: State, checkpoint_path: str, shardings: PyTree) -> State:
    """Load a normal Levanter checkpoint into the destination's MPMD placement.

    The exemplar state supplies global array shapes and destination shardings.
    The checkpoint's source mesh and partitioning may differ. Nonowning
    processes keep empty array descriptors, and each replica reads its own
    shards without collectives across pipeline stages. Scalar leaves are read
    on every device before wrapping the destination MPMD placement; ordinary
    JAX scalar leaves retain that replication for subsequent stage splitting.

    Args:
        state: Destination array tree, including MPMD arrays or abstract arrays.
        checkpoint_path: Concrete checkpoint directory from Levanter discovery.
        shardings: Destination sharding tree, including JaxPP MPMD shardings.

    Returns:
        Restored state with the destination's sharding and MPMD array types.
    """
    arrays = checkpoint_arrays(state)
    # Canonical scalars may be shared by several destination stages. Read them
    # on every device so each stage can reuse its local copy after repartitioning.
    scalar_sharding = NamedSharding(Mesh(np.array(jax.devices()), ("checkpoint",)), PartitionSpec())
    arrays = jax.tree.map(
        lambda value: (
            jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=scalar_sharding) if value.shape == () else value
        ),
        arrays,
    )
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
        checkpoint_path,
        # Avoid restore collectives across processes that do not own this stage.
        read_config=TensorStoreReadConfig(replica_mode=ReplicaRestoreMode.EVERY_REPLICA),
    )
    restored = jax.tree.map(
        lambda original, loaded: original if loaded is None else loaded,
        arrays,
        restored,
        is_leaf=lambda value: value is None,
    )

    return wrap_checkpoint_arrays(restored, shardings)


def wrap_checkpoint_arrays(state: State, shardings: PyTree) -> State:
    """Wrap restored device buffers in the destination MPMD array types.

    Arrays must already reside on the target devices; this does not reshard or
    copy them. Ordinary JAX shardings leave arrays unchanged.
    """

    def wrap(value, target):
        if isinstance(target, jax.sharding.Sharding):
            return value
        if jaxpp is None:
            raise ImportError("MPMD restore requires jaxpp")
        buffers = (
            {shard.device: shard.data for shard in value.addressable_shards}
            if value.sharding.addressable_devices
            else {}
        )
        local_arrays = []
        for mesh_id in sorted(target.mesh_ids):
            sharding = NamedSharding(target.mpmd_mesh.unstack[mesh_id], target.spec, memory_kind=target.memory_kind)
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

    return jax.tree.map(wrap, state, shardings)
