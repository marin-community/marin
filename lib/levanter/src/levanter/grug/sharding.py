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


def _current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def _mesh_has_axis(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> bool:
    if mesh is None or mesh.empty:
        return False
    return axis_name in mesh.shape


def _mesh_axis_size(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        return 1
    return int(mesh.shape.get(axis_name, 1))


def _batch_axes(mesh: Mesh | jax.sharding.AbstractMesh | None) -> tuple[str, ...]:
    """Axes that partition the flat token dim (batch and, if present, context).

    Used for token-space psums and shard_map in_specs on tensors already flattened
    to ``[T = B*S, ...]``. Adding "context" here matches the flat-token sharding
    that context-parallel produces when the seq dim is sharded on "context".
    """
    axes = tuple(axis for axis in ("replica_dcn", "data", "expert", "context") if _mesh_has_axis(mesh, axis))
    if axes:
        return axes
    return ("data",)


def _batch_spec(mesh: Mesh | jax.sharding.AbstractMesh | None) -> PartitionSpec:
    return P(_batch_axes(mesh))


def _axis_names(entry) -> tuple[str, ...]:
    """Flatten one PartitionSpec entry into the mesh axis names it covers."""
    if entry is None:
        return ()
    return tuple(str(name) for name in entry) if isinstance(entry, tuple) else (str(entry),)


def _spec_of(x: jax.Array) -> PartitionSpec | None:
    """The PartitionSpec ``x`` carries, whether it is a concrete array or a traced value.

    A tracer exposes no ``.sharding``, so probing the value alone makes every caller inside
    ``jax.jit`` fall through to a mesh-derived default -- which silently re-partitions a token axis
    the caller had deliberately sharded some other way. ``jax.typeof`` reaches the aval's sharding
    and works in both cases.
    """
    for candidate in (jax.typeof(x), x):
        spec = getattr(getattr(candidate, "sharding", None), "spec", None)
        if spec is not None and len(spec) > 0:
            return spec
    return None


def _batch_spec_from_x(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None) -> PartitionSpec:
    spec = _spec_of(x)
    if spec is not None and spec[0] is not None:
        return P(spec[0])
    return _batch_spec(mesh)


def _is_replicated_spec(spec: PartitionSpec) -> bool:
    return all(axis is None for axis in spec)


def _value_spec_or_default(x: jax.Array, default: PartitionSpec, *, replace_replicated: bool = False) -> PartitionSpec:
    spec = _spec_of(x)
    if spec is not None and not (replace_replicated and _is_replicated_spec(spec)):
        return spec
    return default


def _drop_absent_mesh_axes(mesh: Mesh | jax.sharding.AbstractMesh, spec: PartitionSpec) -> PartitionSpec:
    """Replace mesh-absent axes in ``spec`` with ``None`` (replicated).

    ``compact_grug_mesh`` keeps every axis, but meshes built by tests and other tools name
    only the axes they use, and a spec naming an absent one would raise. An absent axis has
    size 1, so replicating along it is equivalent to sharding over it.
    """

    def keep(entry):
        if entry is None:
            return None
        names = entry if isinstance(entry, tuple) else (entry,)
        kept = tuple(name for name in names if name in mesh.shape)
        if not kept:
            return None
        return kept if len(kept) > 1 else kept[0]

    return P(*(keep(entry) for entry in spec))


def _reshard_for_init(x: jax.Array, spec: PartitionSpec) -> jax.Array:
    mesh = _current_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, NamedSharding(mesh, _drop_absent_mesh_axes(mesh, spec)))


def _reshard_for_shard_map(
    x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None, spec: PartitionSpec
) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


_GRUG_MESH_AXIS_NAMES: tuple[str, ...] = ("replica_dcn", "data", "context", "expert", "model")


def _compact_grug_mesh_shape(
    *,
    process_count: int,
    local_device_count: int,
    expert_axis_size: int,
    replica_axis_size: int,
    model_axis_size: int,
    context_axis_size: int = 1,
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
    if context_axis_size <= 0:
        raise ValueError(f"context_axis_size must be positive, got {context_axis_size}")

    global_device_count = process_count * local_device_count
    fixed_axes = replica_axis_size * expert_axis_size * model_axis_size * context_axis_size
    if global_device_count % fixed_axes != 0:
        raise ValueError(
            f"global_device_count ({global_device_count}) must be divisible by "
            f"replica_axis_size ({replica_axis_size}) * context_axis_size ({context_axis_size}) * "
            f"expert_axis_size ({expert_axis_size}) * model_axis_size ({model_axis_size})"
        )

    data_axis_size = global_device_count // fixed_axes
    return (replica_axis_size, data_axis_size, context_axis_size, expert_axis_size, model_axis_size)


def compact_grug_mesh(
    *,
    expert_axis_size: int = 1,
    replica_axis_size: int | None = None,
    model_axis_size: int = 1,
    context_axis_size: int = 1,
) -> Mesh:
    """Return the compact explicit mesh used by raw Grug PartitionSpecs.

    The mesh is always ``(replica_dcn, data, context, expert, model)``; length-1
    axes are kept so downstream PartitionSpecs can name any axis unconditionally.
    ``data`` absorbs whatever the other axes leave free, so a 32-process job with
    4 local devices can build an effective ``(4, 2, 1, 16, 1)`` Grug mesh.

    ``context_axis_size`` only sizes the ``context`` axis. Placing the sequence
    dimension on it belongs to the model and attention layers; until they do,
    raising it above 1 just narrows ``data`` and leaves token-space reductions
    counting shards no activation is actually split across.
    """
    if replica_axis_size is None:
        replica_axis_size = jax.process_count()

    shape = _compact_grug_mesh_shape(
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
        expert_axis_size=expert_axis_size,
        replica_axis_size=replica_axis_size,
        model_axis_size=model_axis_size,
        context_axis_size=context_axis_size,
    )
    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    axis_types = tuple(AxisType.Explicit for _ in _GRUG_MESH_AXIS_NAMES)
    return Mesh(devices, _GRUG_MESH_AXIS_NAMES, axis_types=axis_types)
