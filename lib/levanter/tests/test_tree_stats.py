# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from haliax.nn import ArrayStacked

from levanter.analysis.tree_stats import summary_statistics_for_tree


class _Block(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    @staticmethod
    def init(weight, bias):
        return _Block(weight=weight, bias=bias)


class _Model(eqx.Module):
    blocks: ArrayStacked


def _make_model(num_layers: int, width: int) -> _Model:
    weights = jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
    biases = jax.random.normal(jax.random.PRNGKey(1), (num_layers, width))
    stack = ArrayStacked.init(num_layers, _Block)(weight=weights, bias=biases)
    return _Model(blocks=stack)


def test_array_stacked_split_emits_per_layer_norms():
    num_layers, width = 4, 3
    model = _make_model(num_layers, width)

    stats = summary_statistics_for_tree("grad", model, split_scan_layers=True)

    # One norm per (layer, parameter), keyed by a stable layer index, not a single collapsed norm.
    for i in range(num_layers):
        layer = model.blocks.get_layer(i)
        for name, expected in (("weight", layer.weight), ("bias", layer.bias)):
            key = f"grad/norm/blocks.{i}.{name}"
            assert key in stats, f"missing {key}; got {sorted(stats)}"
            assert jnp.allclose(stats[key], optax.global_norm(expected))


def test_array_stacked_split_also_keeps_whole_stack_norm():
    num_layers, width = 4, 3
    model = _make_model(num_layers, width)

    stats = summary_statistics_for_tree("grad", model, split_scan_layers=True)

    # Splitting still logs the whole-stack norm under the non-split key, so runs stay comparable to
    # those logged before per-layer splitting.
    for name, expected in (("weight", model.blocks.stacked.weight), ("bias", model.blocks.stacked.bias)):
        key = f"grad/norm/blocks.stacked.{name}"
        assert key in stats, f"missing {key}; got {sorted(stats)}"
        assert jnp.allclose(stats[key], optax.global_norm(expected))


def test_array_stacked_split_whole_stack_norm_handles_shared_leaves():
    num_layers, width = 4, 3
    model = _make_model(num_layers, width)
    # A leaf shared across layers has no leading num_layers axis, so its aggregate must be its own
    # norm, not sqrt(num_layers) times it.
    shared_bias = jax.random.normal(jax.random.PRNGKey(2), (width,))
    model = eqx.tree_at(lambda m: m.blocks.stacked.bias, model, shared_bias)

    stats = summary_statistics_for_tree("grad", model, split_scan_layers=True)

    assert jnp.allclose(stats["grad/norm/blocks.stacked.bias"], optax.global_norm(shared_bias))


def test_array_stacked_split_distinguishes_divergent_layers():
    # A single layer with a blown-up weight must be identifiable by its own key.
    num_layers, width = 4, 3
    model = _make_model(num_layers, width)
    model = eqx.tree_at(lambda m: m.blocks.stacked.weight, model, model.blocks.stacked.weight.at[2].set(1e3))

    stats = summary_statistics_for_tree("grad", model, split_scan_layers=True)

    per_layer = [float(stats[f"grad/norm/blocks.{i}.weight"]) for i in range(num_layers)]
    assert per_layer[2] == max(per_layer)
    assert per_layer[2] > 100 * max(per_layer[i] for i in range(num_layers) if i != 2)


def test_array_stacked_without_split_collapses_layer_axis():
    num_layers, width = 4, 3
    model = _make_model(num_layers, width)

    stats = summary_statistics_for_tree("grad", model, split_scan_layers=False)

    assert not any(".0." in key for key in stats)
    assert jnp.allclose(stats["grad/norm/blocks.stacked.weight"], optax.global_norm(model.blocks.stacked.weight))
