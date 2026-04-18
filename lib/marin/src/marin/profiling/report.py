# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-readable report generation from profile summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from marin.profiling.schema import ProfileSummary


@dataclass
class _RegionGapStats:
    count: int = 0
    total_gap_duration: float = 0.0
    payload_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _RegionFirstGapRow:
    region_path: str
    count: int
    total_gap_duration: float
    avg_gap_duration: float
    top_payload_ops: str


def build_markdown_report(summary: ProfileSummary, *, top_k: int = 10) -> str:
    """Build a deterministic markdown report from a normalized profile summary."""
    metadata = summary.run_metadata
    step = summary.step_time.steady_state_steps
    breakdown = summary.time_breakdown

    lines: list[str] = []
    lines.append(f"# Profile Report ({summary.schema_version})")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Run: `{metadata.run_path or 'unknown'}`")
    lines.append(f"- Artifact: `{metadata.artifact_ref or 'unknown'}`")
    lines.append(f"- Hardware: `{metadata.hardware_type or 'unknown'}`")
    lines.append(f"- Topology: `{metadata.mesh_or_topology or 'unknown'}`")
    lines.append(f"- Git SHA: `{metadata.git_sha or 'unknown'}`")
    lines.append(f"- Generated At (UTC): `{summary.generated_at_utc}`")
    lines.append("")
    lines.append("## Trace Provenance")
    lines.append(f"- Trace SHA256: `{summary.trace_provenance.trace_sha256 or 'unknown'}`")
    lines.append(f"- Observed run_ids: `{', '.join(summary.trace_provenance.run_ids[:8]) or 'unknown'}`")
    lines.append("")
    lines.append("## Trace Overview")
    lines.append(f"- Complete events: `{summary.trace_overview.num_complete_events}`")
    lines.append(f"- Total events: `{summary.trace_overview.num_events_total}`")
    lines.append(f"- Processes: `{summary.trace_overview.num_processes}`")
    lines.append(f"- Threads: `{summary.trace_overview.num_threads}`")
    lines.append(f"- Suspected truncation: `{'yes' if summary.trace_overview.suspected_truncation else 'no'}`")
    if summary.trace_overview.quality_warnings:
        lines.append("- Quality warnings:")
        for warning in summary.trace_overview.quality_warnings:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append("## Step Time (Steady State)")
    lines.append(f"- Steps counted: `{step.count}`")
    lines.append(f"- Median: `{_fmt(step.median)}`")
    lines.append(f"- P90: `{_fmt(step.p90)}`")
    lines.append(f"- Mean: `{_fmt(step.mean)}`")
    lines.append("")
    if summary.step_time.classes:
        lines.append("## Step Classes")
        lines.append("| Class | Count | Fraction | Median | P90 | Representative Step | Periodicity |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for step_class in summary.step_time.classes:
            lines.append(
                f"| `{step_class.name}` | {step_class.count} | {_pct(step_class.fraction_of_steady)} | "
                f"{_fmt(step_class.duration_stats.median)} | {_fmt(step_class.duration_stats.p90)} | "
                f"{step_class.representative_step if step_class.representative_step is not None else 'n/a'} | "
                f"{step_class.periodicity if step_class.periodicity is not None else 'n/a'} |"
            )
        lines.append("")
    lines.append(f"## Time Breakdown (`{summary.time_breakdown.duration_basis}`)")
    lines.append("| Category | Duration | Share |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Compute | {_fmt(breakdown.compute.total_duration)} | {_pct(breakdown.compute.share_of_total)} |")
    communication_duration = _fmt(breakdown.communication.total_duration)
    communication_share = _pct(breakdown.communication.share_of_total)
    lines.append(f"| Communication | {communication_duration} | {communication_share} |")
    lines.append(f"| Host | {_fmt(breakdown.host.total_duration)} | {_pct(breakdown.host.share_of_total)} |")
    lines.append(f"| Stall | {_fmt(breakdown.stall.total_duration)} | {_pct(breakdown.stall.share_of_total)} |")
    lines.append(f"| Other | {_fmt(breakdown.other.total_duration)} | {_pct(breakdown.other.share_of_total)} |")
    lines.append("")
    lines.append("## Hierarchical Regions")
    lines.append("| Region Path | Depth | Count | Inclusive | Inclusive % | Exclusive | Exclusive % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    root_totals = _hierarchical_root_totals(summary)
    fallback_total = max(root_totals.values(), default=0.0)
    for region in summary.hierarchical_regions[:top_k]:
        inclusive = _fmt(region.inclusive_duration)
        exclusive = _fmt(region.exclusive_duration)
        root = region.path.split("=>", 1)[0]
        region_total = root_totals.get(root, fallback_total)
        inclusive_share = _pct((region.inclusive_duration / region_total) if region_total > 0 else 0.0)
        exclusive_share = _pct((region.exclusive_duration / region_total) if region_total > 0 else 0.0)
        lines.append(
            f"| `{region.path}` | {region.depth} | {region.count} | {inclusive} | "
            f"{inclusive_share} | {exclusive} | {exclusive_share} |"
        )
    lines.append("")
    lines.append("## Top Ops")
    lines.append("| Op | Canonical | Category | Count | Exclusive | Avg | Shape Signature |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for op in summary.hot_ops[:top_k]:
        lines.append(
            f"| {_md_code(op.name)} | {_md_code(op.canonical_name)} | {op.category} | {op.count} | "
            f"{_fmt(op.exclusive_duration)} | {_fmt(op.avg_duration)} | {_md_code(op.shape_signature or 'n/a')} |"
        )
    lines.append("")
    lines.append("## Semantic Families")
    lines.append("_Note: FLOP proxy metrics are relative scaling heuristics from trace shapes, not hardware MFU._")
    lines.append(
        "| Family | Count | Exclusive | Share | Avg Exclusive | FLOP Proxy Total | " "FLOP Proxy/s | Example Op |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for family in summary.semantic_families[:top_k]:
        lines.append(
            f"| `{family.family}` | {family.count} | {_fmt(family.exclusive_duration)} | "
            f"{_pct(family.share_of_total)} | "
            f"{_fmt(family.avg_exclusive_duration)} | {_fmt(family.flop_proxy_total)} | "
            f"{_fmt_sci(_inverse_positive(family.time_per_flop_proxy))} | `{family.example_op or 'n/a'}` |"
        )
    lines.append("")
    lines.append("## Communication Collectives")
    lines.append("| Collective | Count | Exclusive | Avg |")
    lines.append("|---|---:|---:|---:|")
    for op in summary.communication_ops[:top_k]:
        lines.append(f"| {op.collective} | {op.count} | {_fmt(op.total_duration)} | {_fmt(op.avg_duration)} |")
    lines.append("")
    lines.append("## Pre-Op Gaps")
    lines.append("| Payload Op | Marker Op | Count | Total Gap | Max Gap | Avg Gap |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for gap in summary.gap_before_ops[:top_k]:
        total_gap = _fmt(gap.total_gap_duration)
        max_gap = _fmt(gap.max_gap_duration)
        avg_gap = _fmt(gap.avg_gap_duration)
        payload = gap.payload_op or gap.name
        marker = gap.marker_op or payload
        lines.append(f"| {_md_code(payload)} | {_md_code(marker)} | {gap.count} | {total_gap} | {max_gap} | {avg_gap} |")
    lines.append("")
    lines.append("## Gap Context (Region-First)")
    lines.append("| Region Path | Count | Total Gap | Avg Gap | Top Payload Ops |")
    lines.append("|---|---:|---:|---:|---|")
    for row in _region_first_gap_rows(summary, top_k=top_k):
        lines.append(
            f"| {_md_code(row.region_path)} | {row.count} | {_fmt(row.total_gap_duration)} | "
            f"{_fmt(row.avg_gap_duration)} | {_md_code(row.top_payload_ops)} |"
        )
    lines.append("")
    lines.append("## Optimization Candidates")
    for candidate in summary.optimization_candidates:
        lines.append(f"### {candidate.title}")
        lines.append(candidate.rationale)
        lines.append("")
        lines.append("Evidence:")
        for evidence in candidate.evidence:
            lines.append(f"- {evidence}")
        lines.append("")
        lines.append("Suggestions:")
        for suggestion in candidate.suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def _inverse_positive(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return 1.0 / value


def _hierarchical_root_totals(summary: ProfileSummary) -> dict[str, float]:
    totals: dict[str, float] = {}
    for region in summary.hierarchical_regions:
        if region.depth != 1:
            continue
        totals[region.path] = max(0.0, float(region.inclusive_duration))
    return totals


def _md_code(value: str) -> str:
    escaped = value.replace("`", "\\`").replace("|", "\\|")
    return f"`{escaped}`"


def _region_first_gap_rows(summary: ProfileSummary, *, top_k: int) -> list[_RegionFirstGapRow]:
    aggregate: dict[str, _RegionGapStats] = {}
    for context in summary.gap_region_contexts:
        bucket = aggregate.setdefault(context.region_path, _RegionGapStats())
        bucket.count += context.count
        bucket.total_gap_duration += context.total_gap_duration
        bucket.payload_counts[context.op_name] += context.count

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -item[1].total_gap_duration,
            item[0],
        ),
    )
    rows: list[_RegionFirstGapRow] = []
    for region_path, stats in ranked[:top_k]:
        count = stats.count
        total = stats.total_gap_duration
        top_payloads = ", ".join(name for name, _ in stats.payload_counts.most_common(3))
        rows.append(
            _RegionFirstGapRow(
                region_path=region_path,
                count=count,
                total_gap_duration=total,
                avg_gap_duration=(total / count) if count else 0.0,
                top_payload_ops=top_payloads or "n/a",
            )
        )
    return rows
