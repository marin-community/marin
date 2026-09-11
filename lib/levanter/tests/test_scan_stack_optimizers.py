# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import numpy as np
from jax.sharding import Mesh

from haliax.nn import ArrayStacked
from haliax.partitioning import axis_mapping, set_mesh

from levanter.optim.kron import scale_by_kron
from levanter.optim.soap import scale_by_soap


class _Module(eqx.Module):
    weight: jax.Array

    @staticmethod
    def init(weight):
        return _Module(weight=weight)


def _stacked_params() -> ArrayStacked[_Module]:
    num_layers = 2
    width = 4
    return ArrayStacked.init(num_layers, _Module)(weight=jax.random.normal(jax.random.PRNGKey(0), (num_layers, width)))


def test_scale_by_soap_init_accepts_array_stacked_params():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with axis_mapping({}), set_mesh(mesh):
        optimizer = scale_by_soap(
            max_precond_dim=16,
            partition_grads_into_blocks=False,
            merge_small_dims=False,
        )
        state = optimizer.init(_stacked_params())

    assert "Q" in state


def test_scale_by_kron_init_accepts_array_stacked_params():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with axis_mapping({}), set_mesh(mesh):
        optimizer = scale_by_kron(
            b1=0.0,
            preconditioner_update_probability=1.0,
            max_size_triangular=16,
            partition_grads_into_blocks=False,
            merge_small_dims=False,
        )
        state = optimizer.init(_stacked_params())

    assert "Qs_preconditioners" in state
