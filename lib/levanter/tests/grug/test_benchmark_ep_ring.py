# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.benchmark_ep_ring import (
    _parity_metrics,
    _parity_status,
    _parser,
    _routing_statistics,
    _selected_experts,
)
from experiments.grug.moe.repro_quack_grouped_mlp_numerics import _error_metrics
from levanter.grug._moe.common import resolve_moe_implementation


def test_ep_ring_benchmark_balanced_routing_has_exact_expert_counts():
    selected = _selected_experts(
        routing="balanced",
        tokens=128,
        top_k=4,
        num_experts=64,
        seed=0,
        skew_alpha=1.2,
    )

    counts = np.bincount(np.asarray(selected).reshape(-1), minlength=64)
    np.testing.assert_array_equal(counts, np.full(64, 8))


def test_ep_ring_benchmark_skew_routing_is_seeded_and_reports_padding():
    arguments = {
        "routing": "skew",
        "tokens": 128,
        "top_k": 4,
        "num_experts": 64,
        "skew_alpha": 1.2,
    }
    selected = _selected_experts(seed=17, **arguments)
    repeated = _selected_experts(seed=17, **arguments)
    different_seed = _selected_experts(seed=18, **arguments)

    np.testing.assert_array_equal(selected, repeated)
    assert not np.array_equal(selected, different_seed)

    statistics = _routing_statistics(selected, num_experts=64, capacity_factor=1.25)
    assert statistics["expert_count_max"] > statistics["expert_count_min"]
    assert statistics["padding_total"] == sum(statistics["padding_by_rank"])
    for group_sizes in statistics["quack_group_sizes_by_rank"]:
        assert sum(group_sizes) == statistics["local_capacity"]


def test_parity_metrics_reports_exact_errors_and_group_breakdown() -> None:
    reference = jnp.asarray([[1.0, 0.0], [2.0, -4.0]], dtype=jnp.float32)
    candidate = jnp.asarray([[1.05, 0.001], [2.5, -3.0]], dtype=jnp.float32)

    metrics = _parity_metrics(
        candidate,
        reference,
        group_ids=jnp.asarray([0, 1]),
        group_labels=("owner_rank=0", "owner_rank=1"),
    )

    expected_difference = np.asarray([[0.05, 0.001], [0.5, 1.0]], dtype=np.float32)
    assert metrics["allclose"] is False
    assert metrics["mismatch_count"] == 3
    assert metrics["mismatch_fraction"] == pytest.approx(0.75)
    assert metrics["reference_l2"] == pytest.approx(np.linalg.norm(reference))
    assert metrics["candidate_l2"] == pytest.approx(np.linalg.norm(candidate))
    assert metrics["relative_l2_error"] == pytest.approx(
        np.linalg.norm(expected_difference) / np.linalg.norm(reference)
    )
    assert metrics["worst_error"] == {
        "flat_index": 3,
        "index": [1, 1],
        "reference_magnitude": 4.0,
        "candidate_magnitude": 3.0,
    }
    assert metrics["abs_error_quantiles"]["exact"] is True
    assert metrics["abs_error_quantiles"]["sample_size"] == 4
    assert metrics["abs_error_quantiles"]["values"]["p100"] == pytest.approx(1.0)
    assert metrics["error_by_group"]["owner_rank=0"]["mismatch_count"] == 1
    assert metrics["error_by_group"]["owner_rank=1"]["mismatch_count"] == 2
    assert json.loads(json.dumps(metrics))["worst_error"]["index"] == [1, 1]


def test_parity_metrics_counts_nan_as_mismatch() -> None:
    metrics = _parity_metrics(jnp.asarray([jnp.nan]), jnp.asarray([0.0]))

    assert metrics["allclose"] is False
    assert metrics["mismatch_count"] == 1


def test_parity_mode_defaults_to_strict_and_accepts_diagnostic() -> None:
    assert _parser().parse_args([]).parity_mode == "strict"
    assert _parser().parse_args(["--parity-mode", "diagnostic"]).parity_mode == "diagnostic"


def test_diagnostic_parity_failure_is_recorded_as_non_promotable() -> None:
    parity = {
        "ring_quack": {
            "dropped_matches": True,
            "output": {"allclose": False},
            "gradients": {"x": {"allclose": True}},
        }
    }

    status = _parity_status(parity, mode="diagnostic")

    assert status == {
        "mode": "diagnostic",
        "passed": False,
        "failures": [{"implementation": "ring_quack", "tensor": "output"}],
        "promotable": False,
        "non_promotable_reason": "diagnostic parity mode",
    }


def test_strict_parity_failure_still_raises() -> None:
    parity = {
        "ring_quack": {
            "dropped_matches": False,
            "output": {"allclose": True},
            "gradients": {"w2": {"allclose": True}},
        }
    }

    with pytest.raises(AssertionError):
        _parity_status(parity, mode="strict")


def test_diagnostic_parity_pass_is_still_non_promotable() -> None:
    parity = {
        "ring_quack": {
            "dropped_matches": True,
            "output": {"allclose": True},
            "gradients": {"w2": {"allclose": True}},
        }
    }

    status = _parity_status(parity, mode="diagnostic")

    assert status["passed"] is True
    assert status["promotable"] is False
    assert status["non_promotable_reason"] == "diagnostic parity mode"


def test_quack_numerics_reproducer_reports_absolute_and_relative_error() -> None:
    expected = jnp.asarray([0.0, 2.0, -4.0], dtype=jnp.float32)
    actual = jnp.asarray([0.001, 2.5, -3.0], dtype=jnp.float32)

    metrics = _error_metrics(actual, expected)

    assert metrics["allclose"] is False
    assert metrics["mismatch_count"] == 3
    assert metrics["mismatch_fraction"] == pytest.approx(1.0)
    assert metrics["mean_abs_error"] == pytest.approx((0.001 + 0.5 + 1.0) / 3)
    assert metrics["max_abs_error"] == pytest.approx(1.0)
    assert metrics["relative_l2_error"] == pytest.approx(
        np.linalg.norm(np.asarray([0.001, 0.5, 1.0])) / np.linalg.norm(expected)
    )


def test_approximate_quack_ring_requires_explicit_backend_name() -> None:
    assert resolve_moe_implementation("ring_quack_approx") == "ring_quack_approx"
    assert resolve_moe_implementation(None) == "ring"
