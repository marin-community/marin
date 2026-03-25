# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from jax._src import pjit
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas import autotune_utils


def test_autotune_utils_wraps_in_shard_map_for_global_named_sharding():
    mesh = Mesh(jax.devices(), ("data",))
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32), NamedSharding(mesh, P("data", None)))
    y = jax.device_put(jnp.zeros((4,), dtype=jnp.int32), NamedSharding(mesh, P("data")))
    w = jax.device_put(jnp.ones((8, 16), dtype=jnp.float32), NamedSharding(mesh, P(None, None)))

    fn = lambda x_value, labels_value, w_value: x_value[:, 0] + labels_value.astype(x_value.dtype) + w_value[0, 0]

    assert not autotune_utils.value_uses_manual_sharding(x)
    assert not autotune_utils.value_uses_manual_sharding(y)
    assert not autotune_utils.value_uses_manual_sharding(w)

    wrapped = autotune_utils.maybe_wrap_in_shard_map(
        fn,
        args=(x, y, w),
        out_specs=P("data"),
    )

    assert wrapped is not fn
    out = wrapped(x, y, w)
    assert out.shape == (4,)
    assert jnp.array_equal(out, fn(x, y, w))


def test_autotune_utils_skip_nested_shard_map_for_manual_sharding():
    mesh = Mesh(jax.devices(), ("data",))
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32), NamedSharding(mesh, P("data", None)))
    y = jax.device_put(jnp.zeros((4,), dtype=jnp.int32), NamedSharding(mesh, P("data")))
    w = jax.device_put(jnp.ones((8, 16), dtype=jnp.float32), NamedSharding(mesh, P(None, None)))

    fn = lambda x_value, labels_value, w_value: x_value[:, 0] + labels_value.astype(x_value.dtype) + w_value[0, 0]
    seen_wrapped_identity: list[bool] = []

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("data", None), P("data"), P(None, None)),
        out_specs=P("data"),
        check_vma=False,
    )
    def _capture(local_x, local_y, local_w):
        assert autotune_utils.value_uses_manual_sharding(local_x)
        assert autotune_utils.value_uses_manual_sharding(local_y)
        assert autotune_utils.value_uses_manual_sharding(local_w)
        wrapped = autotune_utils.maybe_wrap_in_shard_map(
            fn,
            args=(local_x, local_y, local_w),
            out_specs=P("data"),
        )
        seen_wrapped_identity.append(wrapped is fn)
        return wrapped(local_x, local_y, local_w)

    out = _capture(x, y, w)
    out.block_until_ready()

    assert seen_wrapped_identity == [True]
    assert out.shape == (4,)


def test_shape_dtype_struct_for_benchmark_drops_manual_sharding_from_shard_map_tracer():
    mesh = Mesh(jax.devices(), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32), sharding)

    seen_manual: list[bool] = []
    seen_shapes: list[tuple[int, ...]] = []
    seen_structs: list[jax.ShapeDtypeStruct] = []

    @jax.shard_map(mesh=mesh, in_specs=P("data", None), out_specs=P("data", None), check_vma=False)
    def _capture(local_x):
        seen_manual.append(autotune_utils.value_uses_manual_sharding(local_x))
        seen_shapes.append(local_x.shape)
        with pytest.raises(AssertionError):
            pjit.pjit_check_aval_sharding([local_x.aval.sharding], [local_x.aval], ["x"], "arg", False)
        seen_structs.append(autotune_utils.shape_dtype_struct_for_benchmark(local_x))
        return local_x

    _capture(x).block_until_ready()

    assert seen_manual == [True]
    assert len(seen_shapes) == 1
    assert len(seen_structs) == 1
    struct = seen_structs[0]
    assert struct.shape == seen_shapes[0]
    assert struct.dtype == jnp.float32
    assert getattr(struct, "sharding", None) is None


def test_benchmark_lowering_args_preserve_tracers():
    mesh = Mesh(jax.devices(), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32), sharding)

    seen_passthrough: list[bool] = []

    @jax.shard_map(mesh=mesh, in_specs=P("data", None), out_specs=P("data", None), check_vma=False)
    def _capture(local_x):
        lowering_args = autotune_utils.benchmark_lowering_args(local_x)
        seen_passthrough.append(lowering_args[0] is local_x)
        return local_x

    _capture(x).block_until_ready()

    assert seen_passthrough == [True]
