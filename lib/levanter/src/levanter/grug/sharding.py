# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import P
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec, get_abstract_mesh, get_mesh, reshard

# Convenience shorthand for batch sharding. Keep this aligned with Levanter's
# default distributed batch mapping, which includes the cross-slice axis.
Pbatch = P(("replica_dcn", "data"))
Pembed_vocab = P("model", Pbatch[0])
Plm_head = P(Pbatch[0], "model")
Plogits = P(Pbatch[0], None, "model")


def unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P(None))


def current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def mesh_has_axis(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> bool:
    if mesh is None or mesh.empty:
        return False
    return axis_name in mesh.shape


def mesh_axis_size(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        return 1
    return int(mesh.shape.get(axis_name, 1))


def batch_axes(mesh: Mesh | jax.sharding.AbstractMesh | None) -> tuple[str, ...]:
    axes = tuple(axis for axis in ("replica_dcn", "data", "expert") if mesh_has_axis(mesh, axis))
    if axes:
        return axes
    return ("data",)


def batch_spec(mesh: Mesh | jax.sharding.AbstractMesh | None) -> PartitionSpec:
    return P(batch_axes(mesh))


def batch_spec_from_x(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None) -> PartitionSpec:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return P(spec[0])
    return batch_spec(mesh)


def is_replicated_spec(spec: PartitionSpec) -> bool:
    return all(axis is None for axis in spec)


def value_spec_or_default(
    x: jax.Array,
    default: PartitionSpec,
    *,
    replace_replicated: bool = False,
) -> PartitionSpec:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and not (replace_replicated and is_replicated_spec(spec)):
        return spec
    return default


def reshard_for_init(x: jax.Array, spec: PartitionSpec) -> jax.Array:
    mesh = current_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, NamedSharding(mesh, spec))


def reshard_for_shard_map(
    x: jax.Array,
    mesh: Mesh | jax.sharding.AbstractMesh | None,
    spec: PartitionSpec,
) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


def _compact_grug_mesh_shape(
    *,
    process_count: int,
    local_device_count: int,
    expert_axis_size: int,
    replica_axis_size: int,
    model_axis_size: int,
) -> tuple[int, ...]:
    if process_count <= 0:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if local_device_count <= 0:
        raise ValueError(f"local_device_count must be positive, got {local_device_count}")
    if expert_axis_size <= 0:
        raise ValueError(f"expert_axis_size must be positive, got {expert_axis_size}")
    if replica_axis_size <= 0:
        raise ValueError(f"replica_axis_size must be positive, got {replica_axis_size}")
    if model_axis_size <= 0:
        raise ValueError(f"model_axis_size must be positive, got {model_axis_size}")

    global_device_count = process_count * local_device_count
    fixed_axes = replica_axis_size * expert_axis_size * model_axis_size
    if global_device_count % fixed_axes != 0:
        raise ValueError(
            f"global_device_count ({global_device_count}) must be divisible by "
            f"replica_axis_size ({replica_axis_size}) * expert_axis_size ({expert_axis_size}) * "
            f"model_axis_size ({model_axis_size})"
        )

    data_axis_size = global_device_count // fixed_axes
    if expert_axis_size == 1:
        return (replica_axis_size, data_axis_size, model_axis_size)
    return (replica_axis_size, data_axis_size, expert_axis_size, model_axis_size)


def compact_grug_mesh(
    *,
    expert_axis_size: int = 1,
    replica_axis_size: int | None = None,
    model_axis_size: int = 1,
) -> Mesh:
    """Return the compact explicit mesh used by raw Grug PartitionSpecs."""
    if replica_axis_size is None:
        replica_axis_size = jax.process_count()

    shape = _compact_grug_mesh_shape(
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
        expert_axis_size=expert_axis_size,
        replica_axis_size=replica_axis_size,
        model_axis_size=model_axis_size,
    )
    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    axis_names = ("replica_dcn", "data", "model")
    axis_types = (AxisType.Explicit, AxisType.Explicit, AxisType.Explicit)
    if expert_axis_size != 1:
        axis_names = ("replica_dcn", "data", "expert", "model")
        axis_types = (AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit)

    return Mesh(devices, axis_names, axis_types=axis_types)
