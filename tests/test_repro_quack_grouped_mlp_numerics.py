# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import repro_quack_grouped_mlp_numerics as gate


def _metrics(relative_l2: float = 0.0, *, finite: bool = True) -> dict[str, float | bool]:
    return {
        "relative_l2_error": relative_l2,
        "actual_finite": finite,
        "reference_finite": finite,
    }


def test_balanced_route_table_preserves_exact_real_shape_routes():
    routes = gate.balanced_route_table()

    report = gate.route_report(routes)

    assert routes.shape == (16_384, 4)
    assert routes.dtype == np.int32
    assert np.array_equal(routes.reshape(-1)[:68], np.arange(68, dtype=np.int32) % 64)
    assert report["preserved_exactly"]
    assert report["expert_count_min"] == report["expert_count_max"] == 1_024


def test_tensor_metrics_reports_norm_cosine_finiteness_and_max_abs():
    reference = jnp.array([3.0, 4.0], dtype=jnp.float32)
    actual = jnp.array([3.0, 5.0], dtype=jnp.float32)

    metrics = gate.tensor_metrics(actual, reference)

    assert metrics["actual_finite"]
    assert metrics["reference_finite"]
    assert np.isclose(metrics["actual_l2_norm"], np.sqrt(34.0))
    assert metrics["reference_l2_norm"] == 5.0
    assert metrics["error_l2_norm"] == 1.0
    assert metrics["relative_l2_error"] == 0.2
    assert np.isclose(metrics["cosine_similarity"], 29.0 / (np.sqrt(34.0) * 5.0))
    assert metrics["max_abs_error"] == 1.0


def test_admission_report_requires_each_loss_and_gradient_leaf_but_not_output():
    tensors = {name: _metrics() for name in (*gate.REQUIRED_TENSORS, "output")}
    tensors["loss"] = _metrics(gate.RELATIVE_L2_LIMIT)
    tensors["output"] = _metrics(0.5)

    passing = gate.admission_report(tensors)
    tensors["gradient.routing_weights"] = _metrics(gate.RELATIVE_L2_LIMIT + 1e-6)
    failing = gate.admission_report(tensors)

    assert passing["passed"]
    assert passing["output_is_diagnostic_only"]
    assert not failing["passed"]
    assert failing["failures"] == ["gradient.routing_weights"]


def test_timing_summary_reports_reference_over_quack_speedup():
    summary = gate.summarize_benchmark_samples(
        {
            gate.ARM_QUACK: [0.001, 0.002, 0.003],
            gate.ARM_REFERENCE: [0.002, 0.004, 0.006],
        }
    )

    assert summary["sonic_quack_speedup_over_pallas_scatter"] == 2.0
