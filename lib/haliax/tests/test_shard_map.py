# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, axis_mapping
from test_utils import skip_if_not_enough_devices

Dim = Axis("dim", 8)


@skip_if_not_enough_devices(2)
def test_shard_map_basic():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(x):
        return x + 1

    sm = hax.shard_map(fn, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x)
    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + 1)


@skip_if_not_enough_devices(2)
def test_shard_map_pytree_multidim_output():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    B = Axis("b", 8)
    C = Axis("c", 4)
    D = Axis("d", 2)

    def fn(x):
        return {
            "expanded": hax.broadcast_axis(x, D),
            "twice": x + x,
        }

    x = hax.ones((B, C))
    sm = hax.shard_map(fn, mesh=mesh, check_rep=False)
    with axis_mapping({"b": ResourceAxis.DATA}), mesh:
        out = sm(x)

    assert isinstance(out, dict)
    assert out["expanded"].axes == (D, B, C)
    assert out["twice"].axes == (B, C)
    expected_expanded = jnp.broadcast_to(x.array, (D.size, B.size, C.size))
    assert jnp.allclose(out["expanded"].array, expected_expanded)
    assert jnp.allclose(out["twice"].array, x.array * 2)


@skip_if_not_enough_devices(2)
def test_shard_map_multiple_args():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(a, b):
        return a + b

    sm = hax.shard_map(fn, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    y = hax.arange(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x, y)

    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + y.array)


@skip_if_not_enough_devices(2)
def test_shard_map_decorator_usage():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    @hax.shard_map(mesh=mesh, check_rep=False)
    def fn(x):
        return x + 5

    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = fn(x)

    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + 5)


@skip_if_not_enough_devices(2)
def test_shard_map_decorator_no_kwargs():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    @hax.shard_map
    def fn(x):
        return x - 1

    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), hax.partitioning.set_mesh(mesh):
        out = fn(x)

    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array - 1)
