# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array

import haliax as hax
from haliax import Axis, NamedArray
from haliax.partitioning import (
    ResourceAxis,
    axis_mapping,
    named_jit,
    set_mesh,
    pspec_for,
)
from test_utils import skip_if_not_enough_devices


class MyModule(eqx.Module):
    named: NamedArray
    unnamed1: Array
    static_field: int = eqx.static_field()


Dim1 = Axis("dim1", 8)
Dim2 = Axis("dim2", 16)
Dim3 = Axis("dim3", 32)

resource_map = {
    "dim2": ResourceAxis.DATA,
    "dim3": ResourceAxis.MODEL,
}


def test_pspec_for_named_axes():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping(resource_map), mesh:
        mod = MyModule(named=hax.ones((Dim1, Dim2, Dim3)), unnamed1=jnp.ones(Dim2.size), static_field=1)

        specs: MyModule = pspec_for(mod, preserve_existing_shardings=False)

        spec = PartitionSpec(None, ResourceAxis.DATA, ResourceAxis.MODEL)

        assert specs.named == spec
        assert specs.unnamed1 == PartitionSpec(None)


class ArrayModule(eqx.Module):
    arr: Array = hax.field(axis_names=("dim2", "dim3"))


def test_pspec_for_plain_array_axis_names():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping(resource_map), mesh:
        mod = ArrayModule(jnp.ones((Dim2.size, Dim3.size)))

        specs: ArrayModule = pspec_for(mod, preserve_existing_shardings=False)

        assert specs.arr == PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL)


class NestedArrayModule(eqx.Module):
    inner: ArrayModule


def test_pspec_for_plain_array_axis_names_nested_module():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping(resource_map), mesh:
        mod = NestedArrayModule(ArrayModule(jnp.ones((Dim2.size, Dim3.size))))

        specs: NestedArrayModule = pspec_for(mod, preserve_existing_shardings=False)

        assert specs.inner.arr == PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL)


class MyModuleInit(eqx.Module):
    named: NamedArray
    unnamed1: Array
    named2: NamedArray
    static_field: int = eqx.static_field()

    def __init__(self):
        self.named = hax.ones((Dim2, Dim3))
        self.unnamed1 = jnp.ones(())
        self.named2 = hax.ones(Dim3)
        self.static_field = 1


@skip_if_not_enough_devices(4)
def test_pjit_class_init():
    with axis_mapping(resource_map):
        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod = named_jit(MyModuleInit)()

        assert mod.named.array.shape == (Dim2.size, Dim3.size)

        assert mod.unnamed1.shape == ()
        assert mod.named2.array.shape == (Dim3.size,)


@skip_if_not_enough_devices(4)
def test_pjit_class_nested_init():
    with axis_mapping(resource_map):

        class Mod2(eqx.Module):
            inner: MyModuleInit

            def __init__(self):
                self.inner = MyModuleInit()

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod2 = named_jit(Mod2)()

        mod = mod2.inner
        assert mod.named.array.shape == (Dim2.size, Dim3.size)
        assert mod.unnamed1.shape == ()
        assert mod.named2.array.shape == (Dim3.size,)


def test_pjit_class_init_with_args():
    with axis_mapping(resource_map):

        class ModWithArgs(eqx.Module):
            array: NamedArray
            array2: NamedArray

            def __init__(self, in_array: NamedArray):
                self.array = in_array
                self.array2 = hax.zeros(Dim3)

        devices = jax.devices()
        mesh = jax.make_mesh((len(devices), 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
        with set_mesh(mesh):
            mod = named_jit(ModWithArgs)(hax.shard(hax.ones((Dim1, Dim2))))
        assert isinstance(mod, ModWithArgs)
        assert mod.array.array.shape == (Dim1.size, Dim2.size)
        assert mod.array2.array.shape == (Dim3.size,)


def test_infer_resource_partition_gda_bug():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):

        def foo():
            return hax.zeros((Dim1, Dim2, Dim3))

        pjit_foo = named_jit(foo, resource_map)
        r = pjit_foo()
        assert r.axes == (Dim1, Dim2, Dim3)

        def bar(x):
            return x

        # this won't work with GDAs
        pjit_bar = named_jit(bar, resource_map)
        r = pjit_bar(r)
        assert r.axes == (Dim1, Dim2, Dim3)


@skip_if_not_enough_devices(4)
def test_shard_with_axis_mapping_outside_pjit():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
        x = hax.ones((Dim1, Dim2))
        y = hax.ones((Dim2, Dim3))

        x = hax.shard(x, resource_map)
        assert x.array.sharding == NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA))

        y = hax.shard(y, resource_map)
        assert y.array.sharding == NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL))


def test_named_jit_works_without_axis_resources():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):

        def foo(x):
            return x

        pjit_foo = named_jit(foo)
        r = pjit_foo(hax.ones((Dim1, Dim2)))

        assert r.array.sharding.is_fully_replicated

        def foo2(x):
            return hax.shard(x, resource_map)

        pjit_foo2 = named_jit(foo2)
        r2 = pjit_foo2(hax.ones((Dim1, Dim2)))

        assert r2.array.sharding.is_equivalent_to(NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA)), ndim=2)


@skip_if_not_enough_devices(4)
def test_shard_with_axis_mapping_inside_jit():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
        x = hax.ones((Dim1, Dim2))
        y = hax.ones((Dim2, Dim3))

        def assert_inside_pjit(arr, expected: NamedSharding):
            def assert_eq(x, y):
                assert x == y

            jax.debug.inspect_array_sharding(arr.array, callback=lambda x: assert_eq(x, expected))

        @named_jit(out_axis_resources=resource_map)
        def do_shard(x, y):
            x = hax.shard(x, resource_map)
            assert_inside_pjit(x, NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA)))

            y = hax.shard(y, resource_map)
            assert_inside_pjit(y, NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL)))

            return x, y

        x, y = do_shard(x, y)

        assert x.array.sharding == NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA))
        assert y.array.sharding == NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL))


def test_shard_scalar_in_module():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            scalar: jnp.ndarray

            def __init__(self):
                self.scalar = jnp.zeros(
                    (),
                )

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod = named_jit(MyModule)()
            assert mod.scalar.sharding.is_fully_replicated


def test_shard_plain_array_in_module():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            array: jnp.ndarray

            def __init__(self):
                self.array = jnp.zeros((8, 8))

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod = named_jit(MyModule)()
            assert mod.array.sharding.is_fully_replicated


def test_named_jit_with_donation():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            array: jnp.ndarray
            array2: jnp.ndarray

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod = named_jit(MyModule, donate_args=(True, False))(jnp.zeros((8, 8)), jnp.zeros((8, 16)))
            assert mod.array.sharding.is_fully_replicated


def test_named_jit_with_donation_nested_pytrees():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            array: jnp.ndarray
            array2: jnp.ndarray

        class MyModule2(eqx.Module):
            mod: MyModule
            mod2: MyModule

        def init(a1, a2):
            return MyModule2(MyModule(a1, a2), MyModule(a1, a2))

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            mod = named_jit(init, donate_args=(True, False))(jnp.zeros((8, 8)), jnp.zeros((8, 16)))
            assert mod.mod.array.sharding.is_fully_replicated


def test_jit_lower_doesnt_blow_up():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            array: jnp.ndarray
            array2: jnp.ndarray

        class MyModule2(eqx.Module):
            mod: MyModule
            mod2: MyModule

        def init(a1, a2):
            return MyModule2(MyModule(a1, a2), MyModule(a1, a2))

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh, set_mesh(mesh):
            jit_init = named_jit(init, donate_args=(True, False))
            lowered = jit_init.lower(jnp.zeros((8, 8)), jnp.zeros((8, 16)))
            assert lowered
            lowered.cost_analysis()
            lowered.as_text()


def test_cross_device_sharding():
    # this doesn't actually do anything interesting on CPU
    cpu_device = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu_device):
        x = hax.ones((Dim1, Dim2))

    with (
        axis_mapping(resource_map),
        Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)),
    ):
        x = hax.shard(x, resource_map)
        z = hax.ones((Dim1, Dim3))

        x_devices = x.array.devices()
        z_devices = z.array.devices()

        assert set(d.platform for d in x_devices) == set(d.platform for d in z_devices)


def test_named_jit_no_in_axis_resources():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping(resource_map), set_mesh(mesh):

        class MyModule(eqx.Module):
            array: NamedArray

            def __init__(self):
                self.array = hax.ones((Dim1, Dim2))

        data = hax.ones((Dim1, Dim2))
        data = hax.shard(data, {})

        @named_jit(axis_resources=resource_map)
        def fn(data):
            mod = MyModule()
            return mod.array

        r = fn(data)
        assert r.array.sharding.device_set == set(jax.devices())
