# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.ncclep_h100.ep_full_mlp_ab import (
    ARM_RING,
    ARM_TE,
    PROMOTION_SPEEDUP,
    RECV_CAPACITY_PER_RANK,
    RELATIVE_L2_PROMOTION_LIMIT,
    RELATIVE_L2_PROMOTION_TENSORS,
    TimingSummary,
    _dispatch_with_token_gradient,
    balanced_route_table,
    build_summary,
    count_stablehlo_operations,
    parse_args,
    relative_l2_promotion_report,
    routing_capacity_report,
    summarize_times,
    timing_orders,
)

_SCRIPT = Path(__file__).with_name("run_full_mlp_ab.sh")


def _summary(
    *,
    ring_ms: float,
    te_ms: float,
    parity_passed: bool = True,
    relative_l2_passed: bool = True,
) -> dict:
    timings = {
        ARM_RING: TimingSummary(3, ring_ms, ring_ms, ring_ms),
        ARM_TE: TimingSummary(3, te_ms, te_ms, te_ms),
    }
    finite = {
        ARM_RING: {"output": True, "gradients": True},
        ARM_TE: {"output": True, "gradients": True},
    }
    return build_summary(
        timings=timings,
        parity={"passed": parity_passed, "relative_l2_criterion": {"passed": relative_l2_passed}},
        finite=finite,
        runtime={},
        routing=routing_capacity_report(balanced_route_table()),
        stablehlo={},
    )


def test_balanced_routes_fit_identical_ep8_capacity_without_drops() -> None:
    report = routing_capacity_report(balanced_route_table())

    assert report["destination_counts"] == [65_536] * 8
    assert report["aligned_destination_counts"] == [65_536] * 8
    assert report["capacity_padding_rows_per_rank"] == [RECV_CAPACITY_PER_RANK - 65_536] * 8
    assert report["validated_before_dispatch"] is True


def test_interleaved_schedule_balances_first_arm_bias() -> None:
    orders = timing_orders(6)

    assert orders == [
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
    ]


def test_timing_summary_reports_requested_percentiles() -> None:
    timing = summarize_times([0.001, 0.002, 0.003])

    assert timing.median_ms == pytest.approx(2.0)
    assert timing.p10_ms == pytest.approx(1.2)
    assert timing.p90_ms == pytest.approx(2.8)


def test_promotion_requires_twelve_percent_speedup_and_valid_numerics() -> None:
    passing = _summary(ring_ms=11.21, te_ms=10.0)
    parity_failure = _summary(ring_ms=11.21, te_ms=10.0, parity_passed=False)
    relative_l2_failure = _summary(ring_ms=11.21, te_ms=10.0, relative_l2_passed=False)
    too_slow = _summary(ring_ms=11.19, te_ms=10.0)

    assert passing["comparison"]["ring_over_te_speedup"] > PROMOTION_SPEEDUP
    assert passing["promotion_criterion"]["passed"] is True
    assert passing["status"] == "promote"
    assert parity_failure["promotion_criterion"]["passed"] is False
    assert relative_l2_failure["promotion_criterion"]["passed"] is False
    assert too_slow["promotion_criterion"]["passed"] is False


def test_stablehlo_report_counts_collectives_and_custom_call_targets() -> None:
    stablehlo = """
      %0 = stablehlo.all_gather %arg0
      %1 = stablehlo.reduce_scatter %0
      %2 = stablehlo.custom_call @foo(%1) {call_target_name = "te_dispatch"}
      %3 = stablehlo.custom_call @bar(%2) {call_target_name = "te_dispatch"}
    """

    report = count_stablehlo_operations(stablehlo)

    assert report["operations"]["all_gather"] == 1
    assert report["operations"]["reduce_scatter"] == 1
    assert report["operations"]["custom_call"] == 2
    assert report["custom_call_targets"] == {"te_dispatch": 2}


def test_launcher_has_valid_bash_and_dry_run_contract() -> None:
    syntax = subprocess.run(["bash", "-n", _SCRIPT], check=False, capture_output=True, text=True)
    dry_run = subprocess.run(["bash", _SCRIPT, "--dry-run"], check=False, capture_output=True, text=True)
    hybrid_dry_run = subprocess.run(
        ["bash", _SCRIPT, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NCCLEP_TOKEN_GRADIENT_IMPLEMENTATION": "hybrid_combine_forward"},
    )

    assert syntax.returncode == 0, syntax.stderr
    assert dry_run.returncode == 0, dry_run.stderr
    assert hybrid_dry_run.returncode == 0, hybrid_dry_run.stderr
    assert "8 processes x 1 GPU" in dry_run.stdout
    assert "token gradient implementation: native" in dry_run.stdout
    assert "token gradient implementation: hybrid_combine_forward" in hybrid_dry_run.stdout
    assert "TE value_and_grad p50 >= 1.12x ring" in dry_run.stdout


def test_diagnostic_parity_mode_is_explicit() -> None:
    assert parse_args(["--parity-mode", "strict"]).parity_mode == "strict"
    assert parse_args(["--parity-mode", "diagnostic"]).parity_mode == "diagnostic"


def test_overflow_policy_is_explicit() -> None:
    assert parse_args(["--overflow-policy", "trap"]).overflow_policy == "trap"
    assert parse_args(["--overflow-policy", "drop"]).overflow_policy == "drop"


def test_combine_dtype_is_explicit() -> None:
    assert parse_args(["--combine-dtype", "bf16"]).combine_dtype == "bf16"
    assert parse_args(["--combine-dtype", "fp32"]).combine_dtype == "fp32"
    assert parse_args(["--ring-combine-dtype", "bf16"]).ring_combine_dtype == "bf16"
    assert parse_args(["--ring-combine-dtype", "fp32"]).ring_combine_dtype == "fp32"
    assert parse_args(["--dispatch-dtype", "bf16"]).dispatch_dtype == "bf16"
    assert parse_args(["--dispatch-dtype", "fp32"]).dispatch_dtype == "fp32"


def test_token_gradient_implementation_defaults_to_native_and_accepts_hybrid() -> None:
    assert parse_args([]).token_gradient_implementation == "native"
    assert (
        parse_args(["--token-gradient-implementation", "hybrid_combine_forward"]).token_gradient_implementation
        == "hybrid_combine_forward"
    )


def test_relative_l2_promotion_gate_rejects_known_token_gradient_error() -> None:
    tensors = {name: {"relative_l2_error": 0.0} for name in (*RELATIVE_L2_PROMOTION_TENSORS, "output")}
    tensors["gradient.tokens"]["relative_l2_error"] = 0.002909

    report = relative_l2_promotion_report(tensors)

    assert report == {
        "maximum_relative_l2_error": RELATIVE_L2_PROMOTION_LIMIT,
        "required_tensors": list(RELATIVE_L2_PROMOTION_TENSORS),
        "observed": {
            "loss": 0.0,
            "gradient.tokens": 0.002909,
            "gradient.routing_weights": 0.0,
            "gradient.w13": 0.0,
            "gradient.w2": 0.0,
        },
        "failures": ["gradient.tokens"],
        "passed": False,
    }


def test_hybrid_dispatch_uses_combine_forward_token_gradient_and_dispatch_weight_gradient() -> None:
    def ep_dispatch(_config, _routes, tokens, weights, _recv_capacity):
        return tokens * 3, weights * 5, jnp.zeros((1,), dtype=jnp.uint8), jnp.ones((1,), dtype=jnp.int32)

    def ep_dispatch_bwd(
        _config,
        _handle,
        recv_token_cotangent,
        recv_weight_cotangent,
        _output_shape,
        *,
        out_partition_spec,
    ):
        del recv_token_cotangent, out_partition_spec
        return jnp.full((2, 2), -99.0), recv_weight_cotangent * 7

    def ep_combine_fwd(
        _config,
        _handle,
        recv_token_cotangent,
        _output_shape,
        *,
        out_partition_spec,
    ):
        del out_partition_spec
        return recv_token_cotangent * 11

    te_ep = SimpleNamespace(
        ep_dispatch=ep_dispatch,
        _default_out_partition_spec=lambda: (None,),
    )
    te_cpp_ep = SimpleNamespace(
        ep_dispatch_bwd=ep_dispatch_bwd,
        ep_combine_fwd=ep_combine_fwd,
    )
    te_sharding = SimpleNamespace(with_sharding_constraint=lambda value, _spec: value)
    dispatch = _dispatch_with_token_gradient(
        jax,
        jnp,
        te_ep,
        te_cpp_ep,
        te_sharding,
        "hybrid_combine_forward",
    )
    routes = jnp.zeros((2, 1), dtype=jnp.int32)
    tokens = jnp.ones((2, 2), dtype=jnp.float32)
    weights = jnp.ones((2, 1), dtype=jnp.float32)

    def loss(token_values, weight_values):
        recv_tokens, recv_weights, _, _ = dispatch(None, routes, token_values, weight_values, 2)
        return jnp.sum(recv_tokens) + jnp.sum(recv_weights)

    token_gradient, weight_gradient = jax.grad(loss, argnums=(0, 1))(tokens, weights)

    np.testing.assert_array_equal(token_gradient, np.full((2, 2), 11.0, dtype=np.float32))
    np.testing.assert_array_equal(weight_gradient, np.full((2, 1), 7.0, dtype=np.float32))
