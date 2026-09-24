# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.sharding import partitioning_axes, partition_spec_of


@pytest.mark.skipif(jax.device_count() < 2, reason="Requires two devices")
@pytest.mark.parametrize("axis_type", [AxisType.Auto, AxisType.Explicit])
def test_partition_inspection_preserves_concrete_and_traced_placement(axis_type):
    mesh = Mesh(np.array(jax.devices()[:2]).reshape(2, 1), ("context", "model"), axis_types=(axis_type, axis_type))
    with jax.set_mesh(mesh):
        x = jax.device_put(np.arange(8).reshape(4, 2), NamedSharding(mesh, P("context", "model")))
        spec = partition_spec_of(x)
        assert spec == P("context", "model"), spec
        assert partitioning_axes(spec[0], mesh) == ("context",)
        assert partitioning_axes(spec[1], mesh) == ()
        assert partitioning_axes(("model", "context", "absent"), mesh) == ("context",)

        def traced(value):
            traced_spec = partition_spec_of(value)
            if axis_type == AxisType.Explicit:
                assert traced_spec == P("context", "model")
            else:
                # Auto-axis placement is unavailable during tracing.
                assert traced_spec is not None
                assert all(entry is None for entry in traced_spec)
            return value + 1

        np.testing.assert_array_equal(jax.jit(traced)(x), np.arange(8).reshape(4, 2) + 1)


def test_partition_inspection_distinguishes_empty_spec_from_unavailable_spec():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Auto,))
    replicated = jax.device_put(np.arange(8), NamedSharding(mesh, P()))
    assert partition_spec_of(replicated) == P()
    unplaced = jax.device_put(np.arange(8), jax.devices()[0])
    assert partition_spec_of(unplaced) is None
