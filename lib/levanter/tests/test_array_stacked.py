# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import pytest

from levanter.layers.stacked import ArrayStacked
from levanter.layers.scan_checkpoint import ScanCheckpointPolicy
from levanter.utils.tree_utils import scan_aware_tree_map


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

    mapped = scan_aware_tree_map(
        lambda x: x + 1 if isinstance(x, jax.Array) else x,
        stacked,
    )

    assert isinstance(mapped, ArrayStacked)
    assert jnp.allclose(mapped.get_layer(0).weight, stacked.get_layer(0).weight + 1)
    assert jnp.allclose(mapped.get_layer(1).bias, stacked.get_layer(1).bias + 1)


def test_array_stacked_init_keeps_non_array_module_leaves_unbatched():
    """A plain Python value on a non-static field stays that value, usable in layer control flow."""

    class Module(eqx.Module):
        weight: jax.Array
        repeats: int

        @staticmethod
        def init(weight):
            return Module(weight=weight, repeats=3)

        def __call__(self, carry: jax.Array) -> jax.Array:
            for _ in range(self.repeats):
                carry = carry + self.weight
            return carry

    num_layers = 2
    width = 4
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    stacked = ArrayStacked.init(num_layers, Module)(weight=weights)

    assert stacked.stacked.repeats == 3
    assert stacked.stacked.weight.shape == (num_layers, width)
    assert jnp.allclose(stacked.fold(jnp.zeros((width,))), 3 * weights.sum(axis=0))


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


def test_array_stacked_init_slices_layer_batched_inputs_under_eval_shape():
    class Module(eqx.Module):
        value: jax.Array

        @staticmethod
        def init(value):
            return Module(value=value)

    num_layers = 3
    width = 2
    value_shape = jax.ShapeDtypeStruct((num_layers, width), jnp.float32)

    stacked_shape = eqx.filter_eval_shape(ArrayStacked.init(num_layers, Module), value=value_shape)
    assert stacked_shape.stacked.value.shape == (num_layers, width)


def test_array_stacked_state_dict_is_devectorized_for_export():
    class Module(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Module(weight=weight)

    num_layers = 2
    width = 3
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    stacked = ArrayStacked.init(num_layers, Module)(weight=weights)

    state_dict = stacked.to_state_dict("layers")
    assert "layers.0.weight" in state_dict
    assert "layers.1.weight" in state_dict
    assert "layers.weight" not in state_dict

    reloaded = stacked.from_state_dict(state_dict, "layers")
    assert jnp.allclose(reloaded.stacked.weight, stacked.stacked.weight)
    template = eqx.filter_eval_shape(
        ArrayStacked.init(num_layers, Module), weight=jax.ShapeDtypeStruct(weights.shape, weights.dtype)
    )
    restored_template = template.from_state_dict(state_dict, "layers")
    assert jnp.allclose(restored_template.stacked.weight, weights)


@pytest.mark.parametrize("policy", [False, True, "full", "save_all", "nested", ScanCheckpointPolicy(nested=2)])
def test_array_stacked_checkpointing_matches_unrolled_values_and_gradients(policy):
    class Layer(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Layer(weight)

        def __call__(self, x, scale):
            y = jnp.tanh(x * self.weight + scale)
            return y, jnp.square(y)

    weights = jnp.linspace(0.2, 0.8, 12).reshape(4, 3)
    scales = jnp.arange(4, dtype=jnp.float32)[:, None] * 0.1
    x = jnp.ones(3)
    stack = ArrayStacked.init(4, Layer, gradient_checkpointing=policy)(weights)

    def scanned(weight):
        model = eqx.tree_at(lambda m: m.stacked.weight, stack, weight)
        carry, outputs = model.scan(x, scales, in_axes=0)
        return carry.sum() + outputs.sum()

    def unrolled(weight):
        carry = x
        loss = jnp.zeros(())
        for index in range(4):
            carry = jnp.tanh(carry * weight[index] + scales[index])
            loss = loss + jnp.square(carry).sum()
        return carry.sum() + loss

    actual, actual_grad = jax.jit(jax.value_and_grad(scanned))(weights)
    expected, expected_grad = jax.value_and_grad(unrolled)(weights)
    assert jnp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("policy", [True, "nested"])
def test_array_stacked_explicit_sharding_preserves_forward_and_gradients(policy):
    class Layer(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Layer(weight)

        def __call__(self, x, scale):
            return jnp.tanh(x * self.weight + scale)

    width = 3 * jax.device_count()
    mesh = Mesh(np.asarray(jax.devices()), ("model",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        weights = jax.device_put(
            jnp.linspace(0.2, 0.8, 4 * width).reshape(4, width), NamedSharding(mesh, P(None, "model"))
        )
        scales = jax.device_put(jnp.arange(4, dtype=jnp.float32)[:, None], NamedSharding(mesh, P(None, None)))
        x = jax.device_put(jnp.ones(width), NamedSharding(mesh, P("model")))
        stack = ArrayStacked.init(4, Layer, gradient_checkpointing=policy)(weights)

        def scanned(weight):
            model = eqx.tree_at(lambda m: m.stacked.weight, stack, weight)
            return model.fold(x, scales, in_axes=0).sum()

        def unrolled(weight):
            carry = x
            for index in range(4):
                carry = jnp.tanh(carry * weight[index] + scales[index])
            return carry.sum()

        actual, actual_grad = jax.jit(jax.value_and_grad(scanned))(weights)
        expected, expected_grad = jax.value_and_grad(unrolled)(weights)
        assert jnp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
        assert actual_grad.sharding == weights.sharding


def test_array_stacked_state_dict_round_trips_nested_modules_and_shared_leaves():
    class Layer(eqx.Module):
        linear: eqx.nn.Linear
        repeats: int

        @staticmethod
        def init(key):
            return Layer(eqx.nn.Linear(3, 2, key=key), repeats=3)

    stack = ArrayStacked.init(2, Layer)(key=jax.random.split(jax.random.key(0), 2))
    state = stack.to_state_dict("layers")
    assert state["layers.repeats"] == 3
    assert jnp.array_equal(state["layers.1.linear.weight"], stack.get_layer(1).linear.weight)
    modified = {**state, "layers.1.linear.weight": state["layers.1.linear.weight"] + 1}
    reloaded = stack.from_state_dict(modified, "layers")
    assert jnp.array_equal(reloaded.get_layer(0).linear.weight, stack.get_layer(0).linear.weight)
    assert jnp.array_equal(reloaded.get_layer(1).linear.weight, stack.get_layer(1).linear.weight + 1)
    assert reloaded.get_layer(1).repeats == 3
