# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import jax
from jax._src import pjit
import jax.numpy as jnp
import numpy as np
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


def test_compile_benchmark_fn_offloads_from_real_shard_map_tracers(monkeypatch: pytest.MonkeyPatch):
    partition_spec = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    x = jax.device_put(
        jnp.ones((4, 8), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, partition_spec("data", None)),
    )
    y = jax.device_put(
        jnp.zeros((4,), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, partition_spec("data")),
    )
    w = jax.device_put(
        jnp.ones((8, 16), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, partition_spec(None, None)),
    )
    seen_lower_args: list[tuple[object, ...]] = []
    seen_thread_ids: list[int] = []
    seen_mesh_empty: list[bool] = []
    caller_thread_id = threading.get_ident()

    class FakeLowered:
        def compile(self):
            return None

    class FakeJitted:
        def lower(self, *args):
            seen_lower_args.append(args)
            seen_thread_ids.append(threading.get_ident())
            seen_mesh_empty.append(autotune_utils.mesh_lib.thread_resources.env.physical_mesh.empty)
            return FakeLowered()

    monkeypatch.setattr(autotune_utils.jax, "jit", lambda fn: FakeJitted())

    def benchmark_from_shard_map(x_shard, y_shard, w_shard):
        benchmark_fn = lambda x_value, labels_value, w_value: (
            x_value[:, 0] + labels_value.astype(x_value.dtype) + w_value[0, 0]
        )
        lowering_args = autotune_utils.benchmark_lowering_args(x_shard, y_shard, w_shard)
        return autotune_utils.compile_benchmark_fn(
            benchmark_fn=benchmark_fn,
            lowering_args=lowering_args,
            args=(x_shard, y_shard, w_shard),
        )

    mapped = jax.shard_map(
        benchmark_from_shard_map,
        mesh=mesh,
        in_specs=(partition_spec("data", None), partition_spec("data"), partition_spec(None, None)),
        out_specs=partition_spec(),
        check_vma=True,
    )

    score = mapped(x, y, w)
    assert float(score) >= 0.0
    assert len(seen_lower_args) == 1
    lower_x, lower_y, lower_w = seen_lower_args[0]
    assert lower_x.shape == (4, 8)
    assert lower_y.shape == (4,)
    assert lower_w.shape == (8, 16)
    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != caller_thread_id
    assert seen_mesh_empty == [True]
