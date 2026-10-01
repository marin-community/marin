# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The train step runs on the state's flat leaves: same result as on the tree, structure checked."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.train import _flat_train_step


class _Inner(eqx.Module):
    w: jax.Array
    b: jax.Array | None


class _State(eqx.Module):
    step: jax.Array
    inner: tuple[_Inner, ...]


def _state():
    return _State(
        step=jnp.zeros([], jnp.int32),
        inner=(_Inner(jnp.ones((3, 2)), None), _Inner(jnp.full((2,), 2.0), jnp.ones(2))),
    )


def _step(state, batch, loop_active=None, router_tie_active=None):
    scale = 2.0 if loop_active else 1.0
    inner = tuple(eqx.tree_at(lambda m: m.w, m, m.w * batch * scale) for m in state.inner)
    return _State(step=state.step + 1, inner=inner), {"loss": jnp.sum(inner[0].w)}, None


def test_flat_step_matches_the_tree_step_over_several_steps():
    leaves, treedef = jax.tree.flatten(_state())
    flat = _flat_train_step(_step, treedef)
    expected = _state()
    for i in range(3):
        leaves, metrics, _ = flat(leaves, jnp.float32(1.5), loop_active=i == 1)
        expected, expected_metrics, _ = _step(expected, jnp.float32(1.5), loop_active=i == 1)
    got = jax.tree.unflatten(treedef, leaves)
    assert int(got.step) == 3 and got.inner[0].b is None
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
    np.testing.assert_allclose(float(metrics["loss"]), float(expected_metrics["loss"]))


def test_flat_step_rejects_a_changed_structure():
    leaves, treedef = jax.tree.flatten(_state())

    def grows(state, batch, loop_active=None, router_tie_active=None):
        extra = (*state.inner, _Inner(jnp.zeros(1), None))
        return _State(step=state.step, inner=extra), {}, None

    with pytest.raises(ValueError):
        _flat_train_step(grows, treedef)(leaves, jnp.float32(1.0))
