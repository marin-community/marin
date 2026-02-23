# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-readable report generation from profile summaries."""

from __future__ import annotations

from marin.profiling.schema import ProfileSummary


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
    lines.append("## Step Time (Steady State)")
    lines.append(f"- Steps counted: `{step.count}`")
    lines.append(f"- Median: `{_fmt(step.median)}`")
    lines.append(f"- P90: `{_fmt(step.p90)}`")
    lines.append(f"- Mean: `{_fmt(step.mean)}`")
    lines.append("")
    lines.append("## Time Breakdown (Exclusive)")
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
    lines.append("## Top Ops")
    lines.append("| Op | Category | Count | Exclusive | Avg |")
    lines.append("|---|---|---:|---:|---:|")
    for op in summary.hot_ops[:top_k]:
        lines.append(
            f"| `{op.name}` | {op.category} | {op.count} | {_fmt(op.exclusive_duration)} | {_fmt(op.avg_duration)} |"
        )
    lines.append("")
    lines.append("## Communication Collectives")
    lines.append("| Collective | Count | Exclusive | Avg |")
    lines.append("|---|---:|---:|---:|")
    for op in summary.communication_ops[:top_k]:
        lines.append(f"| {op.collective} | {op.count} | {_fmt(op.total_duration)} | {_fmt(op.avg_duration)} |")
    lines.append("")
    lines.append("## Pre-Op Gaps")
    lines.append("| Op | Count | Total Gap | Max Gap | Avg Gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for gap in summary.gap_before_ops[:top_k]:
        total_gap = _fmt(gap.total_gap_duration)
        max_gap = _fmt(gap.max_gap_duration)
        avg_gap = _fmt(gap.avg_gap_duration)
        lines.append(f"| `{gap.name}` | {gap.count} | {total_gap} | {max_gap} | {avg_gap} |")
    lines.append("")
    lines.append("## Gap Context (By Region)")
    lines.append("| Op | Region Path | Count | Total Gap | Avg Gap |")
    lines.append("|---|---|---:|---:|---:|")
    for context in summary.gap_region_contexts[:top_k]:
        total_gap = _fmt(context.total_gap_duration)
        avg_gap = _fmt(context.avg_gap_duration)
        lines.append(f"| `{context.op_name}` | `{context.region_path}` | {context.count} | {total_gap} | {avg_gap} |")
    lines.append("")
    lines.append("## Hierarchical Regions")
    lines.append("| Region Path | Depth | Count | Inclusive | Inclusive % | Exclusive | Exclusive % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    total_duration = summary.time_breakdown.total_duration
    for region in summary.hierarchical_regions[:top_k]:
        inclusive = _fmt(region.inclusive_duration)
        exclusive = _fmt(region.exclusive_duration)
        inclusive_share = _pct((region.inclusive_duration / total_duration) if total_duration > 0 else 0.0)
        exclusive_share = _pct((region.exclusive_duration / total_duration) if total_duration > 0 else 0.0)
        lines.append(
            f"| `{region.path}` | {region.depth} | {region.count} | {inclusive} | "
            f"{inclusive_share} | {exclusive} | {exclusive_share} |"
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
