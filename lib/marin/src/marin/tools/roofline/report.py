# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roofline report construction and rendering."""

from __future__ import annotations

import json
from pathlib import Path

from marin.tools.roofline.attribution import load_attribution_rules
from marin.tools.roofline.hardware import hardware_by_name
from marin.tools.roofline.model_spec import model_preset
from marin.tools.roofline.profile_ingest import ingest_profile
from marin.tools.roofline.scenario import build_rows, build_totals
from marin.tools.roofline.types import ImportState, RooflineReport
from marin.tools.roofline.wandb_ingest import normalize_wandb_run

ROOFLINE_SCHEMA_VERSION = "roofline_dashboard.v1"


def build_report(
    *,
    hardware_name: str,
    model_preset_name: str,
    wandb_run: str | None = None,
    profile: str | None = None,
) -> RooflineReport:
    hardware = hardware_by_name(hardware_name)
    model = model_preset(model_preset_name)
    wandb_ref = normalize_wandb_run(wandb_run)
    profile_result = ingest_profile(profile)
    rules = load_attribution_rules()
    rows, unattributed = build_rows(model, hardware, profile_result.rows, rules)
    warnings = list(profile_result.warnings)
    if profile_result.rows:
        warnings.append(
            "Observed profile times are track-summed unless critical_path_observed_time is populated; "
            "do not read them as exposed wall-clock time."
        )
    return RooflineReport(
        schema_version=ROOFLINE_SCHEMA_VERSION,
        model=model.to_dict(),
        hardware=hardware.to_dict(),
        rows=rows,
        totals=build_totals(rows),
        imports=ImportState(
            wandb_run=wandb_ref.path if wandb_ref else None,
            wandb_run_url=wandb_ref.url if wandb_ref else None,
            profile_path=profile_result.source_path,
            profile_devices=profile_result.profile_devices,
            profile_steps=profile_result.profile_steps,
            warnings=warnings,
        ),
        unattributed=unattributed,
        attribution_rules=[rule.to_dict() for rule in rules],
    )


def write_json_report(report: RooflineReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(report: RooflineReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: RooflineReport) -> str:
    lines = [
        "# Roofline Dashboard Summary",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Model: `{report.model['model_preset']}`",
        f"- Hardware: `{report.hardware['name']}`",
    ]
    if report.imports.wandb_run_url:
        lines.append(f"- W&B: {report.imports.wandb_run_url}")
    if report.imports.profile_path:
        lines.append(f"- Profile: `{report.imports.profile_path}`")
    if report.imports.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report.imports.warnings)

    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Compute roofline floor: `{report.totals.compute_roofline_time:.6f}s`",
            f"- Communication roofline floor: `{report.totals.comm_roofline_time:.6f}s`",
            f"- Estimated scenario total: `{report.totals.estimated_scenario_time:.6f}s`",
        ]
    )
    if report.totals.observed_track_summed_time is not None:
        lines.append(f"- Observed track-summed profile time: `{report.totals.observed_track_summed_time:.6f}s`")
    if report.totals.observed_exposed_time is not None:
        lines.append(f"- Observed exposed wall-clock time: `{report.totals.observed_exposed_time:.6f}s`")

    lines.extend(["", "## Rows", ""])
    headers = [
        "Semantic op",
        "Kind",
        "Ideal (ms)",
        "Efficiency",
        "Estimate (ms)",
        "Observed basis",
        "Observed (ms)",
        "Achieved",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for row in report.rows:
        observed = "" if row.profile_observed_time is None else f"{row.profile_observed_time * 1000.0:.3f}"
        achieved = "" if row.profile_achieved_pct is None else f"{row.profile_achieved_pct * 100.0:.1f}%"
        lines.append(
            "| "
            f"{row.semantic_op} | {row.kind.value} | {row.ideal_time * 1000.0:.3f} | "
            f"{row.user_efficiency * 100.0:.1f}% | {row.estimated_time * 1000.0:.3f} | "
            f"{row.observed_time_basis.value} | {observed} | {achieved} |"
        )

    if report.unattributed:
        lines.extend(["", "## Top Unattributed", ""])
        lines.append("| Name | Duration (ms) | Suggested regex |")
        lines.append("|---|---:|---|")
        for row in report.unattributed[:10]:
            duration_ms = float(row["duration"]) / 1000.0
            suggested = row["suggested_regex"]
            lines.append(f"| {row['name']} | {duration_ms:.3f} | `{suggested}` |")
    return "\n".join(lines) + "\n"
