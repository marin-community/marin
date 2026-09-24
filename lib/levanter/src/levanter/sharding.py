# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sharding inspection shared by model code and kernel wrappers."""

import jax
from jax.sharding import AbstractMesh, Mesh, PartitionSpec


def partition_spec_of(array: jax.Array) -> PartitionSpec | None:
    """Read a concrete array or tracer's partition spec, or None if unavailable.

    Concrete arrays retain Auto-axis placement in their own sharding. A tracer's
    abstract value cannot recover Auto-axis placement, so an empty tracer spec
    does not prove that the array is replicated. Empty specs are preserved.
    """
    if isinstance(array, jax.Array) and not isinstance(array, jax.core.Tracer):
        sharding = array.sharding
    else:
        sharding = jax.typeof(array).sharding
    return getattr(sharding, "spec", None)


def partitioning_axes(entry: str | tuple[str, ...] | None, mesh: Mesh | AbstractMesh | None) -> tuple[str, ...]:
    """Return the named mesh axes of size greater than one in a spec entry.

    Missing axes count as non-partitioning. This inspects placement; it does not
    validate the spec against the mesh. An absent mesh or an unconstrained entry
    has no known partitioning axes.
    """
    if mesh is None or mesh.empty or entry is None or entry is PartitionSpec.UNCONSTRAINED:
        return ()
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    return tuple(name for name in names if mesh.shape.get(name, 1) > 1)
