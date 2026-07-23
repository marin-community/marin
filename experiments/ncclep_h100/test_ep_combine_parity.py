# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.ncclep_h100.ep_combine_parity import (
    GLOBAL_TOKENS,
    RECV_CAPACITY_PER_RANK,
    REFERENCE_FORWARD,
    attribute_results,
    balanced_route_table,
    case_specs,
    distinct_route_contributions,
    expected_dispatch_fingerprints,
    expert_scales,
    route_capacity_report,
    route_weights,
    strict_metrics,
)

_SCRIPT = Path(__file__).with_name("run_combine_parity.sh")


def _metrics(passed: bool, relative_l2_error: float) -> dict:
    return {
        "allclose": passed,
        "relative_l2_error": relative_l2_error,
        "mismatch_count": 0 if passed else 1,
    }


def _case(*, parity: bool, dispatch: bool = True, relative_l2_error: float = 0.0) -> dict:
    return {
        "status": "completed",
        "output_finite": True,
        "dispatch": {"passed": dispatch},
        "references": {
            REFERENCE_FORWARD: _metrics(parity, relative_l2_error),
            "other": _metrics(False, relative_l2_error + 1.0),
        },
    }


def test_routes_and_fingerprints_are_exactly_balanced_for_both_topk_values() -> None:
    for top_k in (1, 4):
        routes = balanced_route_table(top_k)
        weights = route_weights(top_k)
        counts, token_sums, weighted_token_sums = expected_dispatch_fingerprints(routes, weights)
        capacity = route_capacity_report(routes)

        assert routes.shape == (GLOBAL_TOKENS, top_k)
        assert np.all(counts == GLOBAL_TOKENS * top_k // 64)
        assert token_sums.shape == (64, 17)
        assert np.isfinite(weighted_token_sums).all()
        assert capacity["aligned_destination_counts"] == [GLOBAL_TOKENS * top_k // 8] * 8
        assert max(capacity["aligned_destination_counts"]) <= RECV_CAPACITY_PER_RANK


def test_topk4_scaled_identity_has_distinct_route_contributions() -> None:
    routes = balanced_route_table(4)
    weights = route_weights(4)

    assert weights[0].tolist() == [0.5, 0.25, 0.125, 0.125]
    assert distinct_route_contributions(routes, weights, expert_scales())


def test_case_matrix_keeps_fp32_probe_separate_from_bf16_cases() -> None:
    specs = case_specs()

    assert [spec.name for spec in specs] == [
        "topk1_identity",
        "topk1_expert_scaled_identity",
        "topk4_identity",
        "topk4_expert_scaled_identity",
        "topk4_expert_scaled_identity_fp32_combine_input",
    ]
    assert [spec.combine_input_dtype for spec in specs].count("float32") == 1


def test_strict_metrics_reports_exact_bf16_ulp_buckets() -> None:
    reference = jnp.asarray([1.0, 1.0], dtype=jnp.bfloat16)
    candidate = jnp.asarray([1.0, 1.0078125], dtype=jnp.bfloat16)

    metrics = jax.device_get(strict_metrics(candidate, reference, jax, jnp))

    assert int(metrics["ulp"]["histogram"]["0"]) == 1
    assert int(metrics["ulp"]["histogram"]["1"]) == 1
    assert int(metrics["ulp"]["max"]) == 1
    assert float(metrics["max_abs"]) == 0.0078125
    assert int(metrics["absolute_error_histogram"]["0"]) == 1
    assert int(metrics["absolute_error_histogram"]["(0.00390625,0.0078125]"]) == 1


def test_attribution_localizes_topk_only_failure_after_exact_dispatch_and_topk1() -> None:
    cases = {
        "topk1_identity": _case(parity=True),
        "topk1_expert_scaled_identity": _case(parity=True),
        "topk4_identity": _case(parity=False, relative_l2_error=0.003),
        "topk4_expert_scaled_identity": _case(parity=False, relative_l2_error=0.002),
        "topk4_expert_scaled_identity_fp32_combine_input": {
            "status": "unsupported",
        },
    }

    attribution = attribute_results(cases)

    assert attribution["dispatch_fingerprints_exact"] is True
    assert attribution["topk1_identity_strict_parity"] is True
    assert attribution["topk4_identity_strict_parity"] is False
    assert attribution["most_specific_attribution"] == "multi_route_weight_application_or_combine_accumulation_order"
    assert attribution["topk4_scaled_closest_reference"] == REFERENCE_FORWARD
    assert attribution["fp32_combine_input"]["status"] == "unsupported"


def test_attribution_keeps_dispatch_in_scope_when_fingerprints_fail() -> None:
    cases = {
        "topk1_identity": _case(parity=False, dispatch=False),
        "topk1_expert_scaled_identity": _case(parity=False),
        "topk4_identity": _case(parity=False),
        "topk4_expert_scaled_identity": _case(parity=False),
        "topk4_expert_scaled_identity_fp32_combine_input": {
            "status": "unsupported",
        },
    }

    attribution = attribute_results(cases)

    assert attribution["dispatch_fingerprints_exact"] is False
    assert attribution["most_specific_attribution"] == "dispatch_membership_or_route_weight_transport"


def test_launcher_has_valid_bash_and_dry_run_contract() -> None:
    syntax = subprocess.run(["bash", "-n", _SCRIPT], check=False, capture_output=True, text=True)
    dry_run = subprocess.run(["bash", _SCRIPT, "--dry-run"], check=False, capture_output=True, text=True)

    assert syntax.returncode == 0, syntax.stderr
    assert dry_run.returncode == 0, dry_run.stderr
    assert "8 processes x 1 GPU" in dry_run.stdout
    assert "FP32 accumulation, forward-route BF16, reverse-route BF16" in dry_run.stdout
    assert "promotion decision: none" in dry_run.stdout
