# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned schema for agent-consumable profile summaries."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

PROFILE_SUMMARY_SCHEMA_VERSION = "profile_summary.v1"


@dataclass(frozen=True)
class RunMetadata:
    """Metadata describing the run associated with a profile."""

    run_path: str | None = None
    run_id: str | None = None
    artifact_ref: str | None = None
    artifact_name: str | None = None
    hardware_type: str | None = None
    mesh_or_topology: str | None = None
    git_sha: str | None = None
    config_hash: str | None = None
    num_devices: int | None = None
    num_hosts: int | None = None


@dataclass(frozen=True)
class TraceOverview:
    """Trace-level metadata extracted from the input artifact."""

    display_time_unit: str | None
    num_events_total: int
    num_complete_events: int
    num_processes: int
    num_threads: int
    profile_start_ts: float | None
    profile_end_ts: float | None
    duration_basis: str


@dataclass(frozen=True)
class TraceProvenance:
    """Identity/provenance signals for deduplicating and validating traces."""

    trace_sha256: str | None
    run_ids: list[str] = field(default_factory=list)
    source_file_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DurationStats:
    """Summary stats for a set of durations."""

    count: int
    mean: float | None
    median: float | None
    p90: float | None
    min: float | None
    max: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> DurationStats:
        if not values:
            return cls(count=0, mean=None, median=None, p90=None, min=None, max=None)

        sorted_values = sorted(values)
        count = len(sorted_values)
        mean = sum(sorted_values) / count
        median = _quantile(sorted_values, 0.5)
        p90 = _quantile(sorted_values, 0.9)
        return cls(
            count=count,
            mean=mean,
            median=median,
            p90=p90,
            min=sorted_values[0],
            max=sorted_values[-1],
        )


@dataclass(frozen=True)
class StepTimeSummary:
    """Step-time estimates extracted from profile step markers."""

    warmup_steps_ignored: int
    all_steps: DurationStats
    steady_state_steps: DurationStats
    classes: list[StepClassSummary] = field(default_factory=list)


@dataclass(frozen=True)
class StepClassSummary:
    """One detected step-time class (for example, light/heavy)."""

    name: str
    count: int
    fraction_of_steady: float
    duration_stats: DurationStats
    representative_step: int | None
    representative_duration: float | None
    periodicity: int | None


@dataclass(frozen=True)
class BreakdownPart:
    """Single category in the time breakdown."""

    total_duration: float
    share_of_total: float


@dataclass(frozen=True)
class TimeBreakdown:
    """Aggregate time buckets used for bottleneck analysis."""

    duration_basis: str
    total_duration: float
    compute: BreakdownPart
    communication: BreakdownPart
    host: BreakdownPart
    stall: BreakdownPart
    other: BreakdownPart


@dataclass(frozen=True)
class HotOp:
    """Per-op aggregate useful for ranking hotspots."""

    name: str
    canonical_name: str
    category: str
    count: int
    total_duration: float
    exclusive_duration: float
    avg_duration: float
    shape_signature: str | None = None
    source_file: str | None = None
    tf_op_path: str | None = None
    flop_proxy_per_invocation: float | None = None


@dataclass(frozen=True)
class CommunicationOp:
    """Aggregate for communication collectives."""

    collective: str
    count: int
    total_duration: float
    avg_duration: float


@dataclass(frozen=True)
class GapBeforeOp:
    """Idle-gap statistics immediately preceding an op on the same track."""

    name: str
    count: int
    total_gap_duration: float
    max_gap_duration: float
    avg_gap_duration: float


@dataclass(frozen=True)
class RegionAggregate:
    """Aggregate durations for a hierarchical annotation region path."""

    path: str
    depth: int
    count: int
    inclusive_duration: float
    exclusive_duration: float


@dataclass(frozen=True)
class GapRegionContext:
    """Attribution of pre-op gaps to overlapping hierarchical regions."""

    op_name: str
    region_path: str
    count: int
    total_gap_duration: float
    total_overlap_duration: float
    avg_gap_duration: float


@dataclass(frozen=True)
class SemanticFamilyAggregate:
    """Aggregate metrics for an op semantic family (attention/loss/copy/etc)."""

    family: str
    count: int
    total_duration: float
    exclusive_duration: float
    share_of_total: float
    avg_duration: float
    avg_exclusive_duration: float
    example_op: str | None
    dominant_shape_signature: str | None
    flop_proxy_total: float | None
    time_per_flop_proxy: float | None


@dataclass(frozen=True)
class OptimizationCandidate:
    """Actionable hypothesis generated from the profile summary."""

    candidate_id: str
    title: str
    rationale: str
    evidence: list[str]
    suggestions: list[str]


@dataclass(frozen=True)
class ProfileSummary:
    """Normalized, versioned profile summary for agents and automation."""

    schema_version: str
    generated_at_utc: str
    source_format: str
    source_path: str
    run_metadata: RunMetadata
    trace_overview: TraceOverview
    trace_provenance: TraceProvenance
    step_time: StepTimeSummary
    time_breakdown: TimeBreakdown
    hot_ops: list[HotOp]
    semantic_families: list[SemanticFamilyAggregate]
    communication_ops: list[CommunicationOp]
    gap_before_ops: list[GapBeforeOp]
    hierarchical_regions: list[RegionAggregate]
    gap_region_contexts: list[GapRegionContext]
    optimization_candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a JSON-serializable dictionary."""
        return cast(dict[str, Any], dataclasses.asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        """Render the summary as deterministic JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def create(
        cls,
        *,
        source_format: str,
        source_path: str,
        run_metadata: RunMetadata,
        trace_overview: TraceOverview,
        trace_provenance: TraceProvenance,
        step_time: StepTimeSummary,
        time_breakdown: TimeBreakdown,
        hot_ops: list[HotOp],
        semantic_families: list[SemanticFamilyAggregate],
        communication_ops: list[CommunicationOp],
        gap_before_ops: list[GapBeforeOp],
        hierarchical_regions: list[RegionAggregate],
        gap_region_contexts: list[GapRegionContext],
        optimization_candidates: list[OptimizationCandidate],
    ) -> ProfileSummary:
        """Create a summary with default schema version and timestamp."""
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return cls(
            schema_version=PROFILE_SUMMARY_SCHEMA_VERSION,
            generated_at_utc=generated_at,
            source_format=source_format,
            source_path=source_path,
            run_metadata=run_metadata,
            trace_overview=trace_overview,
            trace_provenance=trace_provenance,
            step_time=step_time,
            time_breakdown=time_breakdown,
            hot_ops=hot_ops,
            semantic_families=semantic_families,
            communication_ops=communication_ops,
            gap_before_ops=gap_before_ops,
            hierarchical_regions=hierarchical_regions,
            gap_region_contexts=gap_region_contexts,
            optimization_candidates=optimization_candidates,
        )


def profile_summary_from_dict(data: Mapping[str, Any]) -> ProfileSummary:
    """Parse a dictionary into a typed profile summary."""

    run_metadata = _parse_run_metadata(cast(Mapping[str, Any], data["run_metadata"]))
    trace_overview = _parse_trace_overview(cast(Mapping[str, Any], data["trace_overview"]))
    trace_provenance = _parse_trace_provenance(cast(Mapping[str, Any], data.get("trace_provenance", {})))
    step_time = _parse_step_time(cast(Mapping[str, Any], data["step_time"]))
    time_breakdown = _parse_time_breakdown(cast(Mapping[str, Any], data["time_breakdown"]))
    hot_ops = [_parse_hot_op(cast(Mapping[str, Any], op)) for op in cast(list[Mapping[str, Any]], data["hot_ops"])]
    semantic_families = [
        _parse_semantic_family_aggregate(cast(Mapping[str, Any], family))
        for family in cast(list[Mapping[str, Any]], data.get("semantic_families", []))
    ]
    communication_ops = [
        _parse_communication_op(cast(Mapping[str, Any], op))
        for op in cast(list[Mapping[str, Any]], data["communication_ops"])
    ]
    gap_before_ops = [
        _parse_gap_before_op(cast(Mapping[str, Any], op)) for op in cast(list[Mapping[str, Any]], data["gap_before_ops"])
    ]
    hierarchical_regions = [
        _parse_region_aggregate(cast(Mapping[str, Any], region))
        for region in cast(list[Mapping[str, Any]], data["hierarchical_regions"])
    ]
    gap_region_contexts = [
        _parse_gap_region_context(cast(Mapping[str, Any], context))
        for context in cast(list[Mapping[str, Any]], data.get("gap_region_contexts", []))
    ]
    optimization_candidates = [
        _parse_optimization_candidate(cast(Mapping[str, Any], candidate))
        for candidate in cast(list[Mapping[str, Any]], data["optimization_candidates"])
    ]

    return ProfileSummary(
        schema_version=cast(str, data["schema_version"]),
        generated_at_utc=cast(str, data["generated_at_utc"]),
        source_format=cast(str, data["source_format"]),
        source_path=cast(str, data["source_path"]),
        run_metadata=run_metadata,
        trace_overview=trace_overview,
        trace_provenance=trace_provenance,
        step_time=step_time,
        time_breakdown=time_breakdown,
        hot_ops=hot_ops,
        semantic_families=semantic_families,
        communication_ops=communication_ops,
        gap_before_ops=gap_before_ops,
        hierarchical_regions=hierarchical_regions,
        gap_region_contexts=gap_region_contexts,
        optimization_candidates=optimization_candidates,
    )


def _parse_run_metadata(data: Mapping[str, Any]) -> RunMetadata:
    return RunMetadata(
        run_path=cast(str | None, data.get("run_path")),
        run_id=cast(str | None, data.get("run_id")),
        artifact_ref=cast(str | None, data.get("artifact_ref")),
        artifact_name=cast(str | None, data.get("artifact_name")),
        hardware_type=cast(str | None, data.get("hardware_type")),
        mesh_or_topology=cast(str | None, data.get("mesh_or_topology")),
        git_sha=cast(str | None, data.get("git_sha")),
        config_hash=cast(str | None, data.get("config_hash")),
        num_devices=cast(int | None, data.get("num_devices")),
        num_hosts=cast(int | None, data.get("num_hosts")),
    )


def _parse_trace_overview(data: Mapping[str, Any]) -> TraceOverview:
    return TraceOverview(
        display_time_unit=cast(str | None, data.get("display_time_unit")),
        num_events_total=cast(int, data["num_events_total"]),
        num_complete_events=cast(int, data["num_complete_events"]),
        num_processes=cast(int, data["num_processes"]),
        num_threads=cast(int, data["num_threads"]),
        profile_start_ts=cast(float | None, data.get("profile_start_ts")),
        profile_end_ts=cast(float | None, data.get("profile_end_ts")),
        duration_basis=cast(str, data["duration_basis"]),
    )


def _parse_trace_provenance(data: Mapping[str, Any]) -> TraceProvenance:
    return TraceProvenance(
        trace_sha256=cast(str | None, data.get("trace_sha256")),
        run_ids=list(cast(list[str], data.get("run_ids", []))),
        source_file_hints=list(cast(list[str], data.get("source_file_hints", []))),
    )


def _parse_duration_stats(data: Mapping[str, Any]) -> DurationStats:
    return DurationStats(
        count=cast(int, data["count"]),
        mean=cast(float | None, data.get("mean")),
        median=cast(float | None, data.get("median")),
        p90=cast(float | None, data.get("p90")),
        min=cast(float | None, data.get("min")),
        max=cast(float | None, data.get("max")),
    )


def _parse_step_time(data: Mapping[str, Any]) -> StepTimeSummary:
    return StepTimeSummary(
        warmup_steps_ignored=cast(int, data["warmup_steps_ignored"]),
        all_steps=_parse_duration_stats(cast(Mapping[str, Any], data["all_steps"])),
        steady_state_steps=_parse_duration_stats(cast(Mapping[str, Any], data["steady_state_steps"])),
        classes=[
            _parse_step_class_summary(cast(Mapping[str, Any], step_class))
            for step_class in cast(list[Mapping[str, Any]], data.get("classes", []))
        ],
    )


def _parse_step_class_summary(data: Mapping[str, Any]) -> StepClassSummary:
    return StepClassSummary(
        name=cast(str, data["name"]),
        count=cast(int, data["count"]),
        fraction_of_steady=cast(float, data["fraction_of_steady"]),
        duration_stats=_parse_duration_stats(cast(Mapping[str, Any], data["duration_stats"])),
        representative_step=cast(int | None, data.get("representative_step")),
        representative_duration=cast(float | None, data.get("representative_duration")),
        periodicity=cast(int | None, data.get("periodicity")),
    )


def _parse_breakdown_part(data: Mapping[str, Any]) -> BreakdownPart:
    return BreakdownPart(
        total_duration=cast(float, data["total_duration"]),
        share_of_total=cast(float, data["share_of_total"]),
    )


def _parse_time_breakdown(data: Mapping[str, Any]) -> TimeBreakdown:
    return TimeBreakdown(
        duration_basis=cast(str, data["duration_basis"]),
        total_duration=cast(float, data["total_duration"]),
        compute=_parse_breakdown_part(cast(Mapping[str, Any], data["compute"])),
        communication=_parse_breakdown_part(cast(Mapping[str, Any], data["communication"])),
        host=_parse_breakdown_part(cast(Mapping[str, Any], data["host"])),
        stall=_parse_breakdown_part(cast(Mapping[str, Any], data["stall"])),
        other=_parse_breakdown_part(cast(Mapping[str, Any], data["other"])),
    )


def _parse_hot_op(data: Mapping[str, Any]) -> HotOp:
    name = cast(str, data["name"])
    return HotOp(
        name=name,
        canonical_name=cast(str, data.get("canonical_name") or _canonical_name(name)),
        category=cast(str, data["category"]),
        count=cast(int, data["count"]),
        total_duration=cast(float, data["total_duration"]),
        exclusive_duration=cast(float, data["exclusive_duration"]),
        avg_duration=cast(float, data["avg_duration"]),
        shape_signature=cast(str | None, data.get("shape_signature")),
        source_file=cast(str | None, data.get("source_file")),
        tf_op_path=cast(str | None, data.get("tf_op_path")),
        flop_proxy_per_invocation=cast(
            float | None,
            data.get("flop_proxy_per_invocation", data.get("work_proxy_per_invocation")),
        ),
    )


def _parse_semantic_family_aggregate(data: Mapping[str, Any]) -> SemanticFamilyAggregate:
    return SemanticFamilyAggregate(
        family=cast(str, data["family"]),
        count=cast(int, data["count"]),
        total_duration=cast(float, data["total_duration"]),
        exclusive_duration=cast(float, data["exclusive_duration"]),
        share_of_total=cast(float, data["share_of_total"]),
        avg_duration=cast(float, data["avg_duration"]),
        avg_exclusive_duration=cast(float, data["avg_exclusive_duration"]),
        example_op=cast(str | None, data.get("example_op")),
        dominant_shape_signature=cast(str | None, data.get("dominant_shape_signature")),
        flop_proxy_total=cast(float | None, data.get("flop_proxy_total", data.get("work_proxy_total"))),
        time_per_flop_proxy=cast(
            float | None,
            data.get("time_per_flop_proxy", data.get("normalized_exclusive_per_work")),
        ),
    )


def _parse_communication_op(data: Mapping[str, Any]) -> CommunicationOp:
    return CommunicationOp(
        collective=cast(str, data["collective"]),
        count=cast(int, data["count"]),
        total_duration=cast(float, data["total_duration"]),
        avg_duration=cast(float, data["avg_duration"]),
    )


def _parse_gap_before_op(data: Mapping[str, Any]) -> GapBeforeOp:
    return GapBeforeOp(
        name=cast(str, data["name"]),
        count=cast(int, data["count"]),
        total_gap_duration=cast(float, data["total_gap_duration"]),
        max_gap_duration=cast(float, data["max_gap_duration"]),
        avg_gap_duration=cast(float, data["avg_gap_duration"]),
    )


def _parse_region_aggregate(data: Mapping[str, Any]) -> RegionAggregate:
    return RegionAggregate(
        path=cast(str, data["path"]),
        depth=cast(int, data["depth"]),
        count=cast(int, data["count"]),
        inclusive_duration=cast(float, data["inclusive_duration"]),
        exclusive_duration=cast(float, data["exclusive_duration"]),
    )


def _parse_gap_region_context(data: Mapping[str, Any]) -> GapRegionContext:
    return GapRegionContext(
        op_name=cast(str, data["op_name"]),
        region_path=cast(str, data["region_path"]),
        count=cast(int, data["count"]),
        total_gap_duration=cast(float, data["total_gap_duration"]),
        total_overlap_duration=cast(float, data["total_overlap_duration"]),
        avg_gap_duration=cast(float, data["avg_gap_duration"]),
    )


def _parse_optimization_candidate(data: Mapping[str, Any]) -> OptimizationCandidate:
    return OptimizationCandidate(
        candidate_id=cast(str, data["candidate_id"]),
        title=cast(str, data["title"]),
        rationale=cast(str, data["rationale"]),
        evidence=list(cast(list[str], data["evidence"])),
        suggestions=list(cast(list[str], data["suggestions"])),
    )


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Expected non-empty list of values.")
    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _canonical_name(name: str) -> str:
    return re.sub(r"\.\d+$", "", name.strip().lstrip("%"))
