# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.sharding import full_partition_spec, partitioned_dims
from levanter.testing.cpu_devices import run_on_cpu_devices


def test_full_spec_keeps_length_one_axes_that_partitioned_dims_drops():
    mesh = AbstractMesh(axis_sizes=(2, 1), axis_names=("expert", "data"), axis_types=(AxisType.Explicit,) * 2)
    x = jax.ShapeDtypeStruct((4, 8, 2), jnp.float32, sharding=NamedSharding(mesh, P(("data", "expert"))))

    seen = {}

    def inspect(value):
        seen["full"] = full_partition_spec(value)
        seen["dims"] = partitioned_dims(value, mesh)
        # An array laid out by the full spec combines with `value`; one without `data` would not.
        return jnp.zeros(value.shape, value.dtype, out_sharding=seen["full"]) * value

    with use_abstract_mesh(mesh):
        jax.eval_shape(inspect, x)

    assert seen["full"] == P(("data", "expert"), None, None)
    assert seen["dims"] == (("expert",), (), ())


def test_partition_inspection_preserves_concrete_auto_and_explicit_tracer_placement():
    run_on_cpu_devices(
        """
        import jax
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.sharding import partitioning_axes, partition_spec_of

        for axis_type in (AxisType.Auto, AxisType.Explicit):
            mesh = Mesh(np.array(jax.devices()).reshape(2, 1), ("context", "model"),
                        axis_types=(axis_type, axis_type))
            with jax.set_mesh(mesh):
                x = jax.device_put(np.arange(8).reshape(4, 2),
                                   NamedSharding(mesh, P("context", "model")))
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

                replicated = jax.device_put(np.arange(8), NamedSharding(mesh, P()))
                assert partition_spec_of(replicated) == P()

        unplaced = jax.device_put(np.arange(8), jax.devices()[0])
        assert partition_spec_of(unplaced) is None
        """,
        device_count=2,
    )
