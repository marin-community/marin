# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec, get_abstract_mesh

# Prefer 8 CPU devices when available to exercise multi-axis meshes.
try:
    jax.config.update("jax_num_cpu_devices", 8)
except Exception:
    pass

import haliax as hax
from haliax import Axis
from haliax.partitioning import axis_mapping, set_mesh
from test_utils import skip_if_not_enough_devices


def _build_explicit_mesh():
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",), axis_types=(AxisType.Explicit,))
    return mesh


def test_set_mesh_defaults_to_explicit_axis_types():
    mesh = _build_explicit_mesh()
    with set_mesh(mesh):
        abstract = get_abstract_mesh()
        assert abstract is not None
        assert abstract.axis_types == (AxisType.Explicit,)


@skip_if_not_enough_devices(2)
def test_dot_accepts_out_sharding():
    mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, PartitionSpec("data"))
    resource_map = {"data": "data", "b": None}

    Data = Axis("data", len(jax.devices()))
    B = Axis("b", 2)
    x = hax.ones((Data, B))
    y = hax.ones(B)

    with axis_mapping(resource_map), set_mesh(mesh):
        out = jax.jit(lambda a, b: hax.dot(B, a, b))(x, y)

    assert out.axes == (Data,)
    assert out.array.sharding.spec == sharding.spec


@skip_if_not_enough_devices(2)
def test_einsum_accepts_out_sharding():
    mesh = Mesh(np.array(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(
        mesh,
        PartitionSpec(
            "data",
        ),
    )
    resource_map = {"data": "data", "b": None}

    Data = Axis("data", len(jax.devices()))
    B = Axis("b", 2)
    x = hax.ones((Data, B))
    y = hax.ones(B)

    with axis_mapping(resource_map), set_mesh(mesh):
        out = jax.jit(lambda a, b: hax.einsum("x y,y->x", a, b, x=Data, y=B))(x, y)

    assert out.axes == (Data,)
    assert out.array.sharding.spec == sharding.spec


@skip_if_not_enough_devices(4)
def test_out_sharding_on_nontrivial_mesh():
    devices = jax.devices()
    if len(devices) % 2 != 0:
        import pytest

        pytest.skip("Need an even number of devices to form (n,2) mesh")

    mesh = Mesh(
        np.array(devices).reshape(-1, 2),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, PartitionSpec("data"))
    resource_map = {"data": "data", "b": None, "model": "model"}

    Data = Axis("data", mesh.devices.shape[0])
    B = Axis("b", 2)
    x = hax.ones((Data, B))
    y = hax.ones(B)

    with axis_mapping(resource_map), set_mesh(mesh):
        out = jax.jit(lambda a, b: hax.dot(B, a, b))(x, y)

    assert out.axes == (Data,)
    assert out.array.sharding.spec == sharding.spec
