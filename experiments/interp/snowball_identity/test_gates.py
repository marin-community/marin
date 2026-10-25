# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #310 pre-registered gates fail closed."""

import copy

from experiments.interp.snowball_identity.logit_check import GATE_GROUPS, parity_gate
from experiments.interp.snowball_identity.routing import bias_exception_required, dot_exception_required, route_gate

# The pinned 2026-10-24 baseline's values (logit-check-baseline-20261024-0252-chukar.json), which pass.
BASELINE = {
    "positions": {
        group: {
            "positions": positions,
            "top1_agree": positions - 3,
            "top1_near_tie": 1,
            "top1_disagree": 2,
            "max_abs_logprob_diff": 2.72,
            "p99_abs_logprob_diff": 0.59,
            "missing_from_harness_top50": 0,
        }
        for group, positions in GATE_GROUPS.items()
    },
    "contrastive": {
        "row_conditions": 64,
        "rows_won_agree": 64,
        "rows_top_candidate_agree": 63,
        "p99_abs_score_diff_vs_vllm": 1.18,
        "max_abs_score_diff_vs_vllm": 2.20,
    },
}


def failing(change) -> list[str]:
    report = copy.deepcopy(BASELINE)
    change(report)
    return parity_gate(report)


def test_parity_gate_passes_the_baseline_and_ignores_top_candidate():
    assert parity_gate(BASELINE) == []
    assert failing(lambda r: r["contrastive"].update(rows_top_candidate_agree=50)) == []


def test_parity_gate_fails_closed():
    assert failing(lambda r: r["positions"]["long_past_window"].update(p99_abs_logprob_diff=0.71))
    assert failing(lambda r: r["positions"]["check"].update(max_abs_logprob_diff=float("nan")))
    assert failing(lambda r: r["positions"]["check"].update(top1_agree=290, top1_disagree=15))
    assert failing(lambda r: r["positions"]["contrastive"].update(missing_from_harness_top50=1))
    assert failing(lambda r: r["positions"]["long"].update(positions=2998))
    assert failing(lambda r: r["positions"].pop("long_past_window"))
    assert failing(lambda r: r["contrastive"].update(rows_won_agree=63))
    assert failing(lambda r: r["contrastive"].pop("rows_won_agree"))
    assert failing(lambda r: r["contrastive"].update(p99_abs_score_diff_vs_vllm=1.51))
    assert failing(lambda r: r["contrastive"].update(max_abs_score_diff_vs_vllm=float("nan")))


def rounding(bias: int = 0, dot: int = 0, combined: int = 0, positions: int = 10) -> dict:
    return {
        "totals": {
            "bf16_bias": {"positions": positions, "positions_changed": bias},
            "bf16_logits": {"positions": positions, "positions_changed": dot},
            "bf16_logits_bias": {"positions": positions, "positions_changed": combined},
        }
    }


def cross(positions: int = 10, set_agree: int = 10, ordered_agree: int = 10) -> dict:
    return {"totals": {"positions": positions, "set_agree": set_agree, "ordered_agree": ordered_agree}}


FP32_BIAS = {"router_bias": "float32"}


def test_route_gate_matches_exceptions_to_their_triggers():
    assert not bias_exception_required(rounding()) and bias_exception_required(rounding(bias=1))
    assert dot_exception_required(rounding(dot=1))
    assert route_gate(cross(), rounding(), None, router_dot_fp32=False) == []
    # Bias-only trigger: FP32 bias, BF16 dot.
    assert route_gate(cross(), rounding(bias=2, combined=2), FP32_BIAS, router_dot_fp32=False) == []
    # Dot-only trigger: the FP32 dot needs the FP32 bias too; missing either fails.
    assert route_gate(cross(), rounding(dot=3, combined=3), FP32_BIAS, router_dot_fp32=True) == []
    assert route_gate(cross(), rounding(dot=3, combined=3), None, router_dot_fp32=True)
    assert route_gate(cross(), rounding(dot=3, combined=3), FP32_BIAS, router_dot_fp32=False)
    # Exceptions applied without a trigger fail.
    assert route_gate(cross(), rounding(), FP32_BIAS, router_dot_fp32=False)
    assert route_gate(cross(), rounding(), None, router_dot_fp32=True)
    # A combined-only change has no attributed cause.
    assert route_gate(cross(), rounding(combined=1), None, router_dot_fp32=False)


def test_route_gate_fails_on_any_cross_stack_difference_or_gap():
    assert route_gate(cross(set_agree=9, ordered_agree=9), rounding(), None, False)
    assert route_gate(cross(ordered_agree=9), rounding(), None, False)
    assert route_gate(cross(positions=9, set_agree=9, ordered_agree=9), rounding(), None, False)
    assert route_gate(cross(0, 0, 0), rounding(positions=0), None, False)
