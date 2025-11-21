# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from haliax._src.state_dict import flatten_modules_for_export, unflatten_modules_from_export
from haliax.nn import Linear
from haliax.nn.scan import Stacked, _stack_state_dict, _unstack_state_dict
from haliax.state_dict import from_state_dict, load_state_dict, save_state_dict, to_state_dict


@pytest.mark.parametrize("out_dims_first", [True, False])
def test_flatten_linear_layers(out_dims_first: bool):
    H = hax.Axis("H", 10)
    W = hax.Axis("W", 20)
    D = hax.Axis("D", 30)
    B = hax.Axis("B", 40)
    linear = hax.nn.Linear.init((H, W), (D, B), key=jax.random.PRNGKey(0), use_bias=True, out_first=out_dims_first)

    if out_dims_first:
        assert linear.weight.axes == (D, B, H, W)
    else:
        assert linear.weight.axes == (H, W, D, B)

    flat_linear = linear.flatten_for_export()

    flat_state_dict = to_state_dict(flat_linear)
    if out_dims_first:
        assert flat_state_dict["weight"].shape == (D.size * B.size, H.size * W.size)
    else:
        assert flat_state_dict["weight"].shape == (H.size * W.size, D.size * B.size)
    assert flat_state_dict["bias"].shape == (D.size * B.size,)
    assert flat_state_dict["weight"].dtype == flat_state_dict["bias"].dtype == linear.weight.dtype

    # now unflatten it
    linear2 = Linear.init((H, W), (D, B), key=jax.random.PRNGKey(1), use_bias=True, out_first=out_dims_first)
    new_linear = flat_linear.unflatten_from_export(linear2)

    if out_dims_first:
        assert new_linear.weight.axes == (D, B, H, W)
    else:
        assert new_linear.weight.axes == (H, W, D, B)
    assert new_linear.bias.axes == (D, B)  # type: ignore

    assert linear == new_linear


# Test cases for stack_state_dict
@pytest.mark.parametrize(
    "input_dict, prefix, expected_output",
    [
        # Single block stacking
        (
            {
                "block.0.weight": jnp.array([1, 2]),
                "block.0.bias": jnp.array([3]),
                "block.1.weight": jnp.array([4, 5]),
                "block.1.bias": jnp.array([6]),
            },
            "block",
            {
                "block.weight": jnp.array([[1, 2], [4, 5]]),
                "block.bias": jnp.array([[3], [6]]),
            },
        ),
        # Mixed data types and unmatched items remain unchanged
        (
            {
                "block.0.weight": jnp.array([1, 2]),
                "block.0.bias": jnp.array([3]),
                "block.1.weight": jnp.array([4, 5]),
                "block.1.bias": jnp.array([6.0]),
                "unrelated.item": jnp.array([7]),
            },
            "block",
            {
                "block.weight": jnp.array([[1, 2], [4, 5]]),
                "block.bias": jnp.array([[3.0], [6.0]]),
                "unrelated.item": jnp.array([7]),
            },
        ),
        # No items match prefix, all items should remain unchanged
        (
            {
                "module.0.param": jnp.array([1]),
                "module.1.param": jnp.array([2]),
            },
            "block",
            {
                "module.0.param": jnp.array([1]),
                "module.1.param": jnp.array([2]),
            },
        ),
    ],
)
def test_stack_state_dict(input_dict, prefix, expected_output):
    result = _stack_state_dict(input_dict, prefix)
    for key in expected_output:
        assert jnp.all(jnp.array_equal(result[key], expected_output[key])), f"Failed on key: {key}"

    # now unstack it
    unstacked = _unstack_state_dict(result, prefix)
    for key in input_dict:
        assert jnp.all(jnp.array_equal(unstacked[key], input_dict[key])), f"Failed on key: {key}"


class M(eqx.Module):
    a: Any
    b: Any

    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_to_from_state_dict():
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    m = M(a, b)

    state_dict = to_state_dict(m)
    assert state_dict == {"a": a, "b": b}

    m2 = M(jnp.array([0, 0]), jnp.array([0, 0]))
    m2 = from_state_dict(m2, state_dict)
    assert jnp.all(m2.a == a)
    assert jnp.all(m2.b == b)


def test_export_layer_norm():
    D = hax.Axis("D", 10)
    E = hax.Axis("E", 20)
    layer_norm = hax.nn.LayerNorm.init((D, E), eps=1e-5, use_weight=True, use_bias=True)

    flat_layer_norm = layer_norm.flatten_for_export()

    flat_state_dict = to_state_dict(flat_layer_norm)

    assert flat_state_dict["weight"].shape == (D.size * E.size,)
    assert flat_state_dict["bias"].shape == (D.size * E.size,)
    assert flat_state_dict["weight"].dtype == flat_state_dict["bias"].dtype == layer_norm.weight.dtype

    # now unflatten it
    layer_norm2 = hax.nn.LayerNorm.init((D, E), eps=1e-5, use_weight=True, use_bias=True)
    # ensure we have different weights
    layer_norm2 = dataclasses.replace(layer_norm2, weight=layer_norm2.weight + 1, bias=layer_norm2.bias + 1)

    new_layer_norm = flat_layer_norm.unflatten_from_export(layer_norm2)

    assert layer_norm == new_layer_norm


def test_save_state_dict_roundtrip(tmp_path):
    pytest.importorskip("safetensors.numpy")
    state_dict = {
        "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bias": np.linspace(0.0, 1.0, 3, dtype=np.float32),
    }
    path = tmp_path / "state.safetensors"

    save_state_dict(state_dict, path)

    assert path.exists()
    loaded = load_state_dict(path)

    assert set(loaded.keys()) == {"weight", "bias"}
    np.testing.assert_array_equal(loaded["weight"], state_dict["weight"])
    np.testing.assert_array_equal(loaded["bias"], state_dict["bias"])


def test_save_state_dict_filters_none_and_sets_metadata(tmp_path):
    safetensors = pytest.importorskip("safetensors")

    state_dict = {
        "weight": np.ones((4,), dtype=np.float32),
        "optional": None,
    }
    path = tmp_path / "state.safetensors"

    save_state_dict(state_dict, path)

    with safetensors.safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())

        assert "weight" in keys
        assert "optional" not in keys
        np.testing.assert_array_equal(f.get_tensor("weight"), state_dict["weight"])
        assert f.metadata().get("format") == "pt"


def test_stacked_layer_norm():
    L = hax.Axis("L", 4)
    D = hax.Axis("D", 10)
    E = hax.Axis("E", 20)

    norms = Stacked.init(L, hax.nn.LayerNorm)((D, E), eps=1e-5, use_weight=True, use_bias=True)

    norms_flat = flatten_modules_for_export(norms)

    flat_state_dict = to_state_dict(norms_flat)

    assert flat_state_dict["0.weight"].shape == (D.size * E.size,)
    assert flat_state_dict["0.bias"].shape == (D.size * E.size,)
    assert flat_state_dict["1.weight"].shape == (D.size * E.size,)

    # now unflatten it
    norms2 = Stacked.init(L, hax.nn.LayerNorm)((D, E), eps=1e-5, use_weight=True, use_bias=True)

    new_norms = unflatten_modules_from_export(norms_flat, norms2)

    assert norms == new_norms


def test_linear_doesnt_read_bias_if_it_didnt_have_bias():
    H = hax.Axis("H", 10)
    W = hax.Axis("W", 20)
    D = hax.Axis("D", 30)
    B = hax.Axis("B", 40)

    linear = hax.nn.Linear.init((H, W), (D, B), key=jax.random.PRNGKey(0), use_bias=False, out_first=True)

    flat_linear = linear.flatten_for_export()

    flat_state_dict = to_state_dict(flat_linear)

    assert "bias" not in flat_state_dict
    flat_state_dict["bias"] = jnp.zeros((D.size * B.size,))  # add a dummy bias

    # now unflatten it
    linear2 = Linear.init((H, W), (D, B), key=jax.random.PRNGKey(1), use_bias=False, out_first=True)
    flinear2 = linear2.flatten_for_export()
    flinear2 = from_state_dict(flinear2, flat_state_dict)
    new_linear = flinear2.unflatten_from_export(linear2)

    assert linear == new_linear
