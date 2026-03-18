# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import haliax as hax
from haliax.nn import ArrayStacked


def test_array_stacked_fold_scan_and_layer_access_parity():
    class Module(eqx.Module):
        weight: jax.Array
        bias: jax.Array
        static: int = eqx.field(static=True)

        @staticmethod
        def init(weight, bias, static):
            return Module(weight=weight, bias=bias, static=static)

        def intermediate(self, carry: jax.Array, scale: float) -> jax.Array:
            return carry + self.weight * scale + self.bias + self.static

        def with_output(self, carry: jax.Array, scale: float) -> tuple[jax.Array, jax.Array]:
            carry = self.intermediate(carry, scale)
            return carry, carry * 2

    num_layers = 3
    width = 4
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    bias = jnp.arange(num_layers, dtype=jnp.float32)
    stacked = ArrayStacked.init(num_layers, Module)(weight=weights, bias=bias, static=2)

    assert stacked.num_layers == num_layers

    layer = stacked.get_layer(1)
    assert layer.static == 2
    assert jnp.allclose(layer.weight, weights[1])
    assert jnp.allclose(layer.bias, bias[1])

    unstacked = stacked.unstacked()
    assert len(unstacked) == num_layers
    assert jnp.allclose(unstacked[2].weight, weights[2])

    x = jax.random.normal(jax.random.PRNGKey(1), (width,))
    scale = 0.5

    fold_via = stacked.fold_via(Module.intermediate)(x, scale)
    carry = x
    for i in range(num_layers):
        carry = unstacked[i].intermediate(carry, scale)
    assert jnp.allclose(fold_via, carry)

    scan_carry, scan_out = stacked.scan_via(Module.with_output)(x, scale)
    expected_carry = x
    expected_out = []
    for i in range(num_layers):
        expected_carry, out = unstacked[i].with_output(expected_carry, scale)
        expected_out.append(out)
    assert jnp.allclose(scan_carry, expected_carry)
    assert jnp.allclose(scan_out, jnp.stack(expected_out))


def test_array_stacked_scans_layer_batched_args_only():
    class Module(eqx.Module):
        bias: jax.Array

        @staticmethod
        def init(bias):
            return Module(bias=bias)

        def step(self, carry: jax.Array, layer_scale: jax.Array, mask: jax.Array) -> jax.Array:
            return carry + self.bias + layer_scale + jnp.sum(mask)

    num_layers = 3
    width = 2
    bias = jnp.arange(num_layers, dtype=jnp.float32)
    stacked = ArrayStacked.init(num_layers, Module)(bias=bias)

    x = jnp.zeros((width,), dtype=jnp.float32)
    layer_scale = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    mask = jnp.ones((5, 5), dtype=jnp.float32)

    out = stacked.fold_via(Module.step, in_axes=(0, None))(x, layer_scale, mask)

    expected = x
    for i in range(num_layers):
        expected = expected + bias[i] + layer_scale[i] + jnp.sum(mask)

    assert jnp.allclose(out, expected)


def test_scan_aware_tree_map_updates_array_stacked_leaves():
    class Module(eqx.Module):
        weight: jax.Array
        bias: jax.Array

        @staticmethod
        def init(weight, bias):
            return Module(weight=weight, bias=bias)

    num_layers = 2
    width = 3
    weight = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    bias = jnp.arange(num_layers, dtype=jnp.float32)
    stacked = ArrayStacked.init(num_layers, Module)(weight=weight, bias=bias)

    mapped = hax.tree_util.scan_aware_tree_map(
        lambda x: x + 1 if isinstance(x, jax.Array) else x,
        stacked,
    )

    assert isinstance(mapped, ArrayStacked)
    assert jnp.allclose(mapped.get_layer(0).weight, stacked.get_layer(0).weight + 1)
    assert jnp.allclose(mapped.get_layer(1).bias, stacked.get_layer(1).bias + 1)


def test_array_stacked_rejects_named_array_module_leaves():
    class Module(eqx.Module):
        weight: hax.NamedArray

        @staticmethod
        def init(weight):
            return Module(weight=weight)

    Layers = hax.Axis("Layers", 2)
    Hidden = hax.Axis("Hidden", 3)
    weight = hax.random.normal(jax.random.PRNGKey(0), (Layers, Hidden))

    with pytest.raises(TypeError, match="does not support NamedArray leaves"):
        ArrayStacked.init(Layers.size, Module)(weight=weight)


def test_array_stacked_vmap_via_out_axes_none_matches_vmap_semantics():
    class Module(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Module(weight=weight)

    num_layers = 3
    width = 4
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    stacked = ArrayStacked.init(num_layers, Module)(weight=weights)
    x = jax.random.normal(jax.random.PRNGKey(1), (width,))

    with pytest.raises(ValueError):
        stacked.vmap_via(lambda layer, arg: layer.weight + arg, in_axes=(None,), out_axes=None)(x)


def test_array_stacked_vmap_via_maps_layers_with_out_axes_0():
    class Module(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Module(weight=weight)

    num_layers = 3
    width = 4
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    stacked = ArrayStacked.init(num_layers, Module)(weight=weights)
    x = jax.random.normal(jax.random.PRNGKey(1), (width,))

    out = stacked.vmap_via(lambda layer, arg: layer.weight + arg, in_axes=(None,), out_axes=0)(x)
    assert jnp.allclose(out, weights + x)
