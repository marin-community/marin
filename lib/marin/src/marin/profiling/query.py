# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query and comparison helpers for normalized profile summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from marin.profiling.schema import GapBeforeOp, GapRegionContext, HotOp, ProfileSummary, RegionAggregate


@dataclass(frozen=True)
class OpDelta:
    """Exclusive-duration delta for a hot op between two summaries."""

    name: str
    before: float
    after: float
    delta: float


def query_profile_summary(summary: ProfileSummary, question: str, *, top_k: int = 10) -> dict[str, Any]:
    """
    Answer common profile questions with deterministic, structured output.

    Supported query intents include:
    - top operations by exclusive time
    - compute vs communication dominance
    - collective breakdown
    - optimization next-steps
    """
    normalized = question.lower().strip()

    if "top" in normalized and "op" in normalized:
        return {
            "query_type": "top_ops",
            "results": [_hot_op_to_dict(op) for op in summary.hot_ops[:top_k]],
        }

    if any(token in normalized for token in ("comm", "collective")) and any(
        token in normalized for token in ("compute", "dominating", "dominant", "worst")
    ):
        compute_share = summary.time_breakdown.compute.share_of_total
        communication_share = summary.time_breakdown.communication.share_of_total
        dominant = "communication" if communication_share > compute_share else "compute"
        return {
            "query_type": "compute_vs_communication",
            "dominant": dominant,
            "compute_share": compute_share,
            "communication_share": communication_share,
            "communication_breakdown": [op.__dict__ for op in summary.communication_ops[:top_k]],
        }

    if "collective" in normalized or "communication" in normalized:
        return {
            "query_type": "collective_breakdown",
            "results": [op.__dict__ for op in summary.communication_ops[:top_k]],
        }

    if "gap" in normalized and "before" in normalized:
        target = _extract_target_after_keyword(question, "before")
        if target:
            match = _find_gap_match(summary.gap_before_ops, target)
            return {
                "query_type": "pre_op_gap",
                "target": target,
                "match": match.__dict__ if match is not None else None,
                "context": [
                    _gap_context_to_dict(context) for context in _find_gap_contexts(summary, target, top_k=top_k)
                ],
                "top_gaps": [gap.__dict__ for gap in summary.gap_before_ops[:top_k]],
            }
        return {
            "query_type": "pre_op_gap",
            "top_gaps": [gap.__dict__ for gap in summary.gap_before_ops[:top_k]],
        }

    if "context" in normalized and ("gap" in normalized or "op" in normalized or "copy" in normalized):
        target = _extract_target_after_keyword(question, "for")
        if target is None:
            target = _extract_target_after_keyword(question, "of")
        if target is None:
            target = _extract_target_after_keyword(question, "op")
        if target is None:
            target = _extract_op_like_token(question)
        return {
            "query_type": "gap_region_context",
            "target": target,
            "results": [_gap_context_to_dict(context) for context in _find_gap_contexts(summary, target, top_k=top_k)],
        }

    if "region" in normalized or "hierarch" in normalized:
        total_duration = summary.time_breakdown.total_duration
        return {
            "query_type": "hierarchical_regions",
            "results": [
                _region_to_dict(region, total_duration=total_duration) for region in summary.hierarchical_regions[:top_k]
            ],
        }

    if "memory-bound" in normalized or "compute-bound" in normalized:
        return {
            "query_type": "bound_analysis",
            "result": "unsupported_from_trace_only",
            "details": (
                "The current summary is derived from trace timings and does not include "
                "hardware-counter evidence needed for memory-vs-compute bound classification."
            ),
        }

    if "improve" in normalized or "regress" in normalized or "compare" in normalized:
        return {
            "query_type": "comparison_hint",
            "details": "Use compare_profile_summaries(before, after) to measure regressions/improvements.",
        }

    return {
        "query_type": "optimization_candidates",
        "results": [candidate.__dict__ for candidate in summary.optimization_candidates],
    }


def compare_profile_summaries(before: ProfileSummary, after: ProfileSummary, *, top_k: int = 10) -> dict[str, Any]:
    """
    Compare two profile summaries and report regressions/improvements.

    The function compares:
    - steady-state step-time median and p90
    - time breakdown shares
    - top op exclusive-duration deltas
    """
    before_ops = {op.name: op for op in before.hot_ops}
    after_ops = {op.name: op for op in after.hot_ops}

    all_op_names = sorted(set(before_ops) | set(after_ops))
    deltas: list[OpDelta] = []
    for name in all_op_names:
        before_value = before_ops.get(name).exclusive_duration if name in before_ops else 0.0
        after_value = after_ops.get(name).exclusive_duration if name in after_ops else 0.0
        deltas.append(OpDelta(name=name, before=before_value, after=after_value, delta=after_value - before_value))

    regressed = sorted(
        (delta for delta in deltas if delta.delta > 0),
        key=lambda delta: (-delta.delta, delta.name),
    )[:top_k]
    improved = sorted(
        (delta for delta in deltas if delta.delta < 0),
        key=lambda delta: (delta.delta, delta.name),
    )[:top_k]

    return {
        "before_source": before.source_path,
        "after_source": after.source_path,
        "step_time": {
            "steady_state_median_before": before.step_time.steady_state_steps.median,
            "steady_state_median_after": after.step_time.steady_state_steps.median,
            "steady_state_median_delta": _delta(
                before.step_time.steady_state_steps.median, after.step_time.steady_state_steps.median
            ),
            "steady_state_p90_before": before.step_time.steady_state_steps.p90,
            "steady_state_p90_after": after.step_time.steady_state_steps.p90,
            "steady_state_p90_delta": _delta(
                before.step_time.steady_state_steps.p90, after.step_time.steady_state_steps.p90
            ),
        },
        "time_breakdown_share_delta": {
            "compute": after.time_breakdown.compute.share_of_total - before.time_breakdown.compute.share_of_total,
            "communication": (
                after.time_breakdown.communication.share_of_total - before.time_breakdown.communication.share_of_total
            ),
            "host": after.time_breakdown.host.share_of_total - before.time_breakdown.host.share_of_total,
            "stall": after.time_breakdown.stall.share_of_total - before.time_breakdown.stall.share_of_total,
            "other": after.time_breakdown.other.share_of_total - before.time_breakdown.other.share_of_total,
        },
        "regressed_ops": [delta.__dict__ for delta in regressed],
        "improved_ops": [delta.__dict__ for delta in improved],
    }


def _hot_op_to_dict(op: HotOp) -> dict[str, Any]:
    return {
        "name": op.name,
        "category": op.category,
        "count": op.count,
        "exclusive_duration": op.exclusive_duration,
        "total_duration": op.total_duration,
        "avg_duration": op.avg_duration,
    }


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _extract_target_after_keyword(question: str, keyword: str) -> str | None:
    match = re.search(rf"\b{re.escape(keyword)}\b", question, flags=re.IGNORECASE)
    if match is None:
        return None
    tail = question[match.end() :].strip()
    tail = tail.strip("`\"'?.!, ")
    return _normalize_target(tail)


def _find_gap_match(gaps: list[GapBeforeOp], target: str) -> GapBeforeOp | None:
    normalized_target = target.lower()
    for gap in gaps:
        if gap.name.lower() == normalized_target:
            return gap
    for gap in gaps:
        if normalized_target in gap.name.lower():
            return gap
    return None


def _region_to_dict(region: RegionAggregate, *, total_duration: float) -> dict[str, Any]:
    inclusive_share = (region.inclusive_duration / total_duration) if total_duration > 0 else 0.0
    exclusive_share = (region.exclusive_duration / total_duration) if total_duration > 0 else 0.0
    return {
        "path": region.path,
        "depth": region.depth,
        "count": region.count,
        "inclusive_duration": region.inclusive_duration,
        "exclusive_duration": region.exclusive_duration,
        "inclusive_share_of_total": inclusive_share,
        "exclusive_share_of_total": exclusive_share,
    }


def _find_gap_contexts(summary: ProfileSummary, target: str | None, *, top_k: int) -> list[GapRegionContext]:
    if not summary.gap_region_contexts:
        return []
    if target is None:
        return summary.gap_region_contexts[:top_k]

    normalized_target = _normalize_target(target)
    if normalized_target is None:
        return summary.gap_region_contexts[:top_k]
    normalized_target = normalized_target.lower()
    exact = [context for context in summary.gap_region_contexts if context.op_name.lower() == normalized_target]
    if exact:
        return exact[:top_k]
    fuzzy = [context for context in summary.gap_region_contexts if normalized_target in context.op_name.lower()]
    return fuzzy[:top_k]


def _gap_context_to_dict(context: GapRegionContext) -> dict[str, Any]:
    return {
        "op_name": context.op_name,
        "region_path": context.region_path,
        "count": context.count,
        "total_gap_duration": context.total_gap_duration,
        "avg_gap_duration": context.avg_gap_duration,
        "total_overlap_duration": context.total_overlap_duration,
    }


def _normalize_target(target: str | None) -> str | None:
    if target is None:
        return None

    normalized = target.strip().strip("`\"'?.!, ")
    normalized = re.sub(r"^(op|operation)\s+", "", normalized, flags=re.IGNORECASE)
    if not normalized:
        return None
    return normalized


def _extract_op_like_token(question: str) -> str | None:
    match = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.\d+\b", question)
    if match is None:
        return None
    return _normalize_target(match.group(0))
