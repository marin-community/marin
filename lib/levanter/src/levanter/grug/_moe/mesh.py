# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Mesh and sharding helpers for Grug MoE dispatch."""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard


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


def _batch_spec(mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    if _mesh_has_axis(mesh, "expert"):
        return P(("data", "expert"))
    return P(("data",))


def _batch_spec_from_x(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return P(spec[0])
    return _batch_spec(mesh)


def _is_replicated_spec(spec: P) -> bool:
    return all(axis is None for axis in spec)


def _value_spec_or_default(x: jax.Array, default: P, *, replace_replicated: bool = False) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and not (replace_replicated and _is_replicated_spec(spec)):
        return spec
    return default


def _reshard_for_init(x: jax.Array, spec: P) -> jax.Array:
    mesh = _current_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, NamedSharding(mesh, spec))


def _reshard_for_shard_map(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None, spec: P) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x
