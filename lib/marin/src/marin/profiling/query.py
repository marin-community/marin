# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query and comparison helpers for normalized profile summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from marin.profiling.semantics import classify_semantic_family, estimate_flop_proxy
from marin.profiling.schema import (
    GapBeforeOp,
    GapRegionContext,
    HotOp,
    ProfileSummary,
    RegionAggregate,
    SemanticFamilyAggregate,
    StepClassSummary,
)


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

    before_step_classes = _step_class_rows(before.step_time.classes)
    after_step_classes = _step_class_rows(after.step_time.classes)
    step_class_delta = _compare_step_classes(before.step_time.classes, after.step_time.classes)
    semantic_family_delta = _compare_semantic_families(
        _semantic_families_for_summary(before),
        _semantic_families_for_summary(after),
    )
    provenance_checks = _compare_provenance(before, after)

    return {
        "before_source": before.source_path,
        "after_source": after.source_path,
        "provenance_checks": provenance_checks,
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
        "step_classes": {
            "before": before_step_classes,
            "after": after_step_classes,
            "delta": step_class_delta,
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
        "semantic_family_delta": semantic_family_delta,
        "regressed_ops": [delta.__dict__ for delta in regressed],
        "improved_ops": [delta.__dict__ for delta in improved],
    }


def _hot_op_to_dict(op: HotOp) -> dict[str, Any]:
    return {
        "name": op.name,
        "canonical_name": op.canonical_name,
        "category": op.category,
        "count": op.count,
        "exclusive_duration": op.exclusive_duration,
        "total_duration": op.total_duration,
        "avg_duration": op.avg_duration,
        "shape_signature": op.shape_signature,
        "source_file": op.source_file,
        "tf_op_path": op.tf_op_path,
        "flop_proxy_per_invocation": op.flop_proxy_per_invocation,
    }


def _step_class_rows(step_classes: list[StepClassSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_class in step_classes:
        rows.append(
            {
                "name": step_class.name,
                "count": step_class.count,
                "fraction_of_steady": step_class.fraction_of_steady,
                "median_duration": step_class.duration_stats.median,
                "p90_duration": step_class.duration_stats.p90,
                "representative_step": step_class.representative_step,
                "representative_duration": step_class.representative_duration,
                "periodicity": step_class.periodicity,
            }
        )
    return rows


def _compare_step_classes(before: list[StepClassSummary], after: list[StepClassSummary]) -> list[dict[str, Any]]:
    before_by_name = {row.name: row for row in before}
    after_by_name = {row.name: row for row in after}
    rows: list[dict[str, Any]] = []
    for name in sorted(set(before_by_name) | set(after_by_name)):
        before_row = before_by_name.get(name)
        after_row = after_by_name.get(name)
        before_median = before_row.duration_stats.median if before_row is not None else None
        after_median = after_row.duration_stats.median if after_row is not None else None
        rows.append(
            {
                "name": name,
                "before_count": before_row.count if before_row else 0,
                "after_count": after_row.count if after_row else 0,
                "before_fraction": before_row.fraction_of_steady if before_row else 0.0,
                "after_fraction": after_row.fraction_of_steady if after_row else 0.0,
                "median_before": before_median,
                "median_after": after_median,
                "median_delta": _delta(before_median, after_median),
                "median_regression_pct": _pct_delta(before_median, after_median),
            }
        )
    return rows


def _semantic_families_for_summary(summary: ProfileSummary) -> list[SemanticFamilyAggregate]:
    if summary.semantic_families:
        return summary.semantic_families

    # Backward-compatible fallback for older summaries.
    aggregate: dict[str, dict[str, float | int | str | None]] = {}
    total_duration = summary.time_breakdown.total_duration
    for op in summary.hot_ops:
        family = classify_semantic_family(op.name)
        bucket = aggregate.setdefault(
            family,
            {
                "count": 0,
                "total_duration": 0.0,
                "exclusive_duration": 0.0,
                "example_op": op.name,
                "flop_proxy_total": 0.0,
                "flop_proxy_count": 0,
            },
        )
        bucket["count"] = int(bucket["count"]) + op.count
        bucket["total_duration"] = float(bucket["total_duration"]) + op.total_duration
        bucket["exclusive_duration"] = float(bucket["exclusive_duration"]) + op.exclusive_duration
        flop_proxy = op.flop_proxy_per_invocation
        if flop_proxy is None and op.shape_signature:
            flop_proxy = estimate_flop_proxy(family, op.shape_signature)
        if flop_proxy is not None and op.count > 0:
            bucket["flop_proxy_total"] = float(bucket["flop_proxy_total"]) + (flop_proxy * op.count)
            bucket["flop_proxy_count"] = int(bucket["flop_proxy_count"]) + op.count

    rows: list[SemanticFamilyAggregate] = []
    for family, stats in sorted(aggregate.items(), key=lambda item: (-float(item[1]["exclusive_duration"]), item[0])):
        count = int(stats["count"])
        total = float(stats["total_duration"])
        exclusive = float(stats["exclusive_duration"])
        flop_proxy_total = float(stats["flop_proxy_total"])
        rows.append(
            SemanticFamilyAggregate(
                family=family,
                count=count,
                total_duration=total,
                exclusive_duration=exclusive,
                share_of_total=(exclusive / total_duration) if total_duration > 0 else 0.0,
                avg_duration=(total / count) if count else 0.0,
                avg_exclusive_duration=(exclusive / count) if count else 0.0,
                example_op=str(stats["example_op"]),
                dominant_shape_signature=None,
                flop_proxy_total=flop_proxy_total if flop_proxy_total > 0 else None,
                time_per_flop_proxy=(exclusive / flop_proxy_total) if flop_proxy_total > 0 else None,
            )
        )
    return rows


def _compare_semantic_families(
    before: list[SemanticFamilyAggregate],
    after: list[SemanticFamilyAggregate],
) -> list[dict[str, Any]]:
    before_by_family = {row.family: row for row in before}
    after_by_family = {row.family: row for row in after}
    rows: list[dict[str, Any]] = []
    for family in sorted(set(before_by_family) | set(after_by_family)):
        before_row = before_by_family.get(family)
        after_row = after_by_family.get(family)
        before_exclusive = before_row.exclusive_duration if before_row else 0.0
        after_exclusive = after_row.exclusive_duration if after_row else 0.0
        before_time_per_flop = before_row.time_per_flop_proxy if before_row else None
        after_time_per_flop = after_row.time_per_flop_proxy if after_row else None
        before_flops = before_row.flop_proxy_total if before_row else None
        after_flops = after_row.flop_proxy_total if after_row else None
        time_ratio = _ratio(before_exclusive, after_exclusive)
        flop_ratio = _ratio(before_flops, after_flops)
        rows.append(
            {
                "family": family,
                "before_exclusive_duration": before_exclusive,
                "after_exclusive_duration": after_exclusive,
                "exclusive_duration_delta": after_exclusive - before_exclusive,
                "exclusive_regression_pct": _pct_delta(before_exclusive, after_exclusive),
                "before_flop_proxy_total": before_flops,
                "after_flop_proxy_total": after_flops,
                "flop_proxy_ratio": flop_ratio,
                "before_time_per_flop_proxy": before_time_per_flop,
                "after_time_per_flop_proxy": after_time_per_flop,
                "time_per_flop_regression_pct": _pct_delta(before_time_per_flop, after_time_per_flop),
                "time_ratio": time_ratio,
                "work_ratio": flop_ratio,
                "efficiency_ratio": (time_ratio / flop_ratio) if time_ratio is not None and flop_ratio else None,
                "example_before": before_row.example_op if before_row else None,
                "example_after": after_row.example_op if after_row else None,
                "shape_signature_before": before_row.dominant_shape_signature if before_row else None,
                "shape_signature_after": after_row.dominant_shape_signature if after_row else None,
            }
        )
    return rows


def _compare_provenance(before: ProfileSummary, after: ProfileSummary) -> dict[str, Any]:
    messages: list[str] = []
    severity = "pass"

    before_hash = before.trace_provenance.trace_sha256
    after_hash = after.trace_provenance.trace_sha256
    if before_hash and after_hash and before_hash == after_hash:
        severity = "fail"
        messages.append("Before and after summaries point to identical trace content (matching SHA256).")

    if before.source_path == after.source_path:
        if severity != "fail":
            severity = "warn"
        messages.append("Before and after summaries have the same source_path.")

    if (
        before.run_metadata.run_id
        and after.run_metadata.run_id
        and before.run_metadata.run_id == after.run_metadata.run_id
    ):
        if severity != "fail":
            severity = "warn"
        messages.append("Before and after summaries have the same W&B run_id.")

    if before.run_metadata.artifact_ref and after.run_metadata.artifact_ref:
        if before.run_metadata.artifact_ref == after.run_metadata.artifact_ref:
            if severity != "fail":
                severity = "warn"
            messages.append("Before and after summaries reference the same W&B artifact.")

    if not before_hash or not after_hash:
        if severity == "pass":
            severity = "warn"
        messages.append("One or both summaries are missing trace SHA256 provenance.")

    return {
        "status": severity,
        "messages": messages,
        "before_trace_sha256": before_hash,
        "after_trace_sha256": after_hash,
        "before_run_id": before.run_metadata.run_id,
        "after_run_id": after.run_metadata.run_id,
    }


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _pct_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before <= 0:
        return None
    return ((after - before) / before) * 100.0


def _ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before <= 0:
        return None
    return after / before


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
