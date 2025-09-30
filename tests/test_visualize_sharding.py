# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, axis_mapping, named_jit
from test_utils import skip_if_not_enough_devices
from haliax.debug import visualize_shardings

Dim1 = Axis("dim1", 8)
Dim2 = Axis("dim2", 8)
Dim3 = Axis("dim3", 2)

resource_map = {
    "dim1": ResourceAxis.DATA,
    "dim2": ResourceAxis.MODEL,
    "dim3": ResourceAxis.REPLICA,
}


def test_visualize_shardings_runs(capsys):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(-1, 1, 1),
        (ResourceAxis.DATA, ResourceAxis.MODEL, ResourceAxis.REPLICA),
    )
    with axis_mapping(resource_map), mesh:
        arr = hax.ones((Dim1, Dim2, Dim3))
        visualize_shardings(arr)

    out = capsys.readouterr().out
    assert "dim1" in out and "dim2" in out and "dim3" in out


@skip_if_not_enough_devices(2, reason="JAX will claim it's unsharded if there's only one device")
def test_visualize_shardings_inside_jit(capsys):
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))

    @named_jit
    def fn(x):
        x = hax.shard(x)
        visualize_shardings(x)
        return x

    with axis_mapping({"dim1": ResourceAxis.DATA}), mesh:
        x = hax.ones({"dim1": 8 * len(jax.devices())})
        x = hax.shard(x)
        fn(x)

    out = capsys.readouterr().out
    assert "dim1" in out


def test_visualize_shardings_plain_array(capsys):
    x = jnp.ones((4, 4))
    visualize_shardings(x)
    out = capsys.readouterr().out
    assert out.strip() != ""


@skip_if_not_enough_devices(2)
def test_visualize_shardings_model_axis(capsys):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping({"dim1": ResourceAxis.DATA, "dim2": ResourceAxis.MODEL}), mesh:
        arr = hax.ones((Dim1, Dim2))
        visualize_shardings(arr)

    out = capsys.readouterr().out
    assert "dim2" in out
