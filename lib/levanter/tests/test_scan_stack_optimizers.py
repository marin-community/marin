# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from levanter.layers.stacked import ArrayStacked
from haliax.partitioning import axis_mapping, set_mesh

from levanter.optim.kron import scale_by_kron
from levanter.optim.soap import scale_by_soap
from levanter.optim.util import scan_aware_clip_by_block_rms
from levanter.utils.tree_utils import scan_aware_tree_map


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


def test_array_stacked_rms_clipping_treats_layers_independently():
    weights = jnp.array([[3.0, 4.0], [0.3, 0.4]])
    params = ArrayStacked.init(2, _Module)(weight=weights)
    transform = scan_aware_clip_by_block_rms(1.0)

    updates, _ = transform.update(params, transform.init(params))

    expected = jnp.stack([row / jnp.maximum(1.0, jnp.sqrt(jnp.mean(row * row))) for row in weights])
    np.testing.assert_allclose(updates.stacked.weight, expected, atol=1e-6, rtol=1e-6)


def test_scan_aware_tree_map_multiple_inputs_and_shared_leaves():
    class Layer(eqx.Module):
        weight: jax.Array
        bias: jax.Array

        @staticmethod
        def init(weight, bias):
            return Layer(weight, bias)

    weights = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    stack = ArrayStacked.init(3, Layer)(weight=weights, bias=jnp.zeros_like(weights))
    shared_bias = jnp.array([0.0, 2.0, 4.0, 6.0])
    stack = eqx.tree_at(lambda x: x.stacked.bias, stack, shared_bias)
    reference = ArrayStacked.init(3, Layer)(weight=weights * 2, bias=jnp.ones_like(weights))

    def center_and_add(layer, other):
        return Layer(layer.weight - jnp.mean(layer.weight) + other.weight, layer.bias + other.bias)

    mapped = scan_aware_tree_map(center_and_add, stack, reference, is_leaf=lambda node: isinstance(node, Layer))

    expected_weights = jnp.stack([row - jnp.mean(row) + row * 2 for row in weights])
    np.testing.assert_allclose(mapped.stacked.weight, expected_weights, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(mapped.stacked.bias, jnp.broadcast_to(shared_bias + 1, weights.shape))
