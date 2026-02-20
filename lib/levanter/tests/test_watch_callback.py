# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
from jax import numpy as jnp
import pytest

from levanter.callbacks.watch import WatchCallback, compute_watch_stats
from levanter.trainer_state import InsideJitInfo


class _TinyModel(eqx.Module):
    weight: jax.Array
    bias: jax.Array


@dataclass
class _DummyState:
    model: _TinyModel
    trainable_model: _TinyModel
    opt_state: dict[str, object]


def test_compute_watch_stats_expected_prefixes():
    params = {"w": jnp.array([1.0, -2.0])}
    grads = {"w": jnp.array([0.1, -0.2])}
    updates = {"w": jnp.array([-0.01, 0.02])}

    stats = compute_watch_stats(
        watch_targets=["grads", "params", "updates"],
        include_norms=True,
        include_per_parameter_norms=True,
        include_histogram=False,
        split_scan_layers=True,
        params=params,
        grads=grads,
        updates=updates,
    )

    assert stats
    assert any(key.startswith("grad/") for key in stats)
    assert any(key.startswith("params/") for key in stats)
    assert any(key.startswith("updates/") for key in stats)


def test_compute_watch_stats_opt_state_model_tree_filtering():
    tracked = _TinyModel(weight=jnp.array([1.0, 2.0]), bias=jnp.array([3.0]))
    opt_state = {
        "tracked": tracked,
        "ignored_array": jnp.array([5.0]),
        "ignored_tree": {"x": jnp.array([7.0])},
    }

    stats = compute_watch_stats(
        watch_targets=["opt_state"],
        include_norms=True,
        include_per_parameter_norms=True,
        include_histogram=False,
        split_scan_layers=True,
        opt_state=opt_state,
        model_tree_type=type(tracked),
    )

    assert stats
    assert all(key.startswith("opt_state/tracked/") for key in stats)
    assert not any("ignored" in key for key in stats)


def test_watch_callback_inside_step_matches_compute_helper():
    model = _TinyModel(weight=jnp.array([1.0, 2.0]), bias=jnp.array([0.5]))
    state = _DummyState(
        model=model,
        trainable_model=_TinyModel(weight=jnp.array([1.5, 2.5]), bias=jnp.array([0.25])),
        opt_state={"tracked": _TinyModel(weight=jnp.array([0.2, 0.4]), bias=jnp.array([0.1]))},
    )
    inside_info = InsideJitInfo(
        grads=_TinyModel(weight=jnp.array([0.01, -0.02]), bias=jnp.array([0.03])),
        updates=_TinyModel(weight=jnp.array([-0.001, 0.002]), bias=jnp.array([0.004])),
    )

    callback = WatchCallback(
        watch_targets=["grads", "params", "updates", "opt_state"],
        include_norms=True,
        include_per_parameter_norms=True,
        include_histogram=False,
        split_scan_layers=True,
    )
    callback_stats = callback.inside_step(cast(Any, state), inside_info)

    helper_stats = compute_watch_stats(
        watch_targets=["grads", "params", "updates", "opt_state"],
        include_norms=True,
        include_per_parameter_norms=True,
        include_histogram=False,
        split_scan_layers=True,
        params=state.trainable_model,
        grads=inside_info.grads,
        updates=inside_info.updates,
        opt_state=state.opt_state,
        model_tree_type=type(state.model),
    )

    assert callback_stats.keys() == helper_stats.keys()
    for key in callback_stats:
        assert jnp.allclose(jnp.asarray(callback_stats[key]), jnp.asarray(helper_stats[key]))


def test_invalid_watch_target_raises_value_error():
    with pytest.raises(ValueError, match="Invalid watch targets"):
        compute_watch_stats(
            watch_targets=cast(Any, ["bogus"]),
            include_norms=True,
            include_per_parameter_norms=True,
            include_histogram=False,
            split_scan_layers=True,
        )

    with pytest.raises(ValueError, match="Invalid watch targets"):
        WatchCallback(watch_targets=["bogus"])
