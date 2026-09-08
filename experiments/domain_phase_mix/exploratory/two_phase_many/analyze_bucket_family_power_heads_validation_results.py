# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Join the 3e18 bucket-family GRP validation panel to final heldout metrics."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs/bucket_family_power_heads_validation_panel_20260714"
HELDOUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_3e18_append_only_heldouts_20260714"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/bucket_family_power_heads_validation_results_20260715"

MANIFEST = PANEL_DIR / "candidate_manifest.csv"
HELDOUTS = HELDOUT_DIR / "heldout_current.csv"
EXPECTED_CANDIDATES = 22
PANEL_PREFIX = "bfgrp_"
TARGET_COLUMN = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
POLICY_LABEL = {
    "single_phase_tied": "one phase",
    "two_phase": "two phase",
}
RESULT_FIELDS = (
    "candidate",
    "objective",
    "model",
    "policy",
    "selected_l2",
    "aggregate_kl_coefficient",
    "phase_information_budget",
    "predicted_bpb",
    "observed_target_bpb",
    "optimism_bpb",
    "uncheatable_bpb",
    "table9_macro_bpb",
    "wandb_run_id",
    "wandb_run_name",
    "wandb_url",
    "policy_class",
    "fit_panel_overlap",
    "max_simulated_epoch",
    "aggregate_tv_to_proportional",
    "phase_information_kl",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Missing finite {field!r} for {row.get('wandb_run_name') or row.get('candidate')}")
    return value


def complete(row: Mapping[str, str]) -> bool:
    return row["training_state"] == "finished" and row["checkpoint_declared_complete"] == "1"


def prior_frontiers(
    heldouts: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    frontiers: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for objective, target in TARGET_COLUMN.items():
        for policy in POLICY_LABEL:
            eligible = [
                row
                for row in heldouts
                if complete(row)
                and row["fit_panel_overlap"] == "coordinate_disjoint"
                and row["policy_class"] == policy
                and not row["wandb_run_base"].startswith(PANEL_PREFIX)
                and row[target] != ""
            ]
            frontiers[objective][policy] = min(eligible, key=lambda row: finite(row, target))
    return dict(frontiers)


def noise_summary(heldouts: Sequence[Mapping[str, str]]) -> dict[str, dict[str, float | int]]:
    repeats = [
        row
        for row in heldouts
        if complete(row) and row["training_series"] == "delphi_3e18_baseline_noise_panel_20260703"
    ]
    if len(repeats) != 10:
        raise ValueError(f"Expected 10 proportional repeats, found {len(repeats)}")
    output: dict[str, dict[str, float | int]] = {}
    for objective, target in TARGET_COLUMN.items():
        values = [finite(row, target) for row in repeats]
        standard_deviation = statistics.stdev(values)
        output[objective] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "standard_deviation": standard_deviation,
            "difference_standard_deviation": math.sqrt(2.0) * standard_deviation,
        }
    return output


def markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    rendered = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    rendered.extend(f"| {' | '.join(str(value) for value in row)} |" for row in rows)
    return "\n".join(rendered)


def main() -> None:
    manifest = read_csv(MANIFEST)
    heldouts = read_csv(HELDOUTS)
    if len(manifest) != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} manifest rows, found {len(manifest)}")
    by_base = {row["wandb_run_base"]: row for row in heldouts}
    results: list[dict[str, object]] = []
    for candidate in manifest:
        run_base = f"{candidate['candidate']}_3e18"
        if run_base not in by_base:
            raise ValueError(f"Missing heldout registry row for {run_base}")
        heldout = by_base[run_base]
        if not complete(heldout):
            raise ValueError(f"Incomplete validation row {heldout['wandb_run_name']}")
        objective = candidate["objective"]
        observed = finite(heldout, TARGET_COLUMN[objective])
        predicted = finite(candidate, "predicted_bpb")
        results.append(
            {
                **{field: candidate[field] for field in candidate if field in RESULT_FIELDS},
                "observed_target_bpb": observed,
                "optimism_bpb": observed - predicted,
                "uncheatable_bpb": finite(heldout, "uncheatable_bpb"),
                "table9_macro_bpb": finite(heldout, "table9_macro_bpb"),
                "wandb_run_id": heldout["wandb_run_id"],
                "wandb_run_name": heldout["wandb_run_name"],
                "wandb_url": heldout["wandb_url"],
                "policy_class": heldout["policy_class"],
                "fit_panel_overlap": heldout["fit_panel_overlap"],
            }
        )
    frontiers = prior_frontiers(heldouts)
    noise = noise_summary(heldouts)
    ranking: list[dict[str, object]] = []
    for objective in TARGET_COLUMN:
        objective_rows = sorted(
            (row for row in results if row["objective"] == objective),
            key=lambda row: float(row["observed_target_bpb"]),
        )
        for rank, row in enumerate(objective_rows, start=1):
            ranking.append({**row, "objective_rank": rank})

    summary_rows: list[dict[str, object]] = []
    for objective in TARGET_COLUMN:
        for policy in ("single_phase_tied", "two_phase"):
            panel_rows = [row for row in results if row["objective"] == objective and row["policy_class"] == policy]
            winner = min(panel_rows, key=lambda row: float(row["observed_target_bpb"]))
            prior = frontiers[objective][policy]
            prior_value = finite(prior, TARGET_COLUMN[objective])
            observed = float(winner["observed_target_bpb"])
            summary_rows.append(
                {
                    "objective": objective,
                    "policy_class": policy,
                    "panel_winner": winner["candidate"],
                    "panel_winner_bpb": observed,
                    "prior_frontier_run": prior["wandb_run_name"],
                    "prior_frontier_bpb": prior_value,
                    "delta_vs_prior_frontier": observed - prior_value,
                    "difference_sd_units": (
                        (observed - prior_value) / float(noise[objective]["difference_standard_deviation"])
                    ),
                    "new_frontier": observed < prior_value,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "results.csv", RESULT_FIELDS, results)
    write_csv(
        OUTPUT_DIR / "ranking.csv",
        (*RESULT_FIELDS, "objective_rank"),
        ranking,
    )
    write_csv(
        OUTPUT_DIR / "frontier_comparison.csv",
        (
            "objective",
            "policy_class",
            "panel_winner",
            "panel_winner_bpb",
            "prior_frontier_run",
            "prior_frontier_bpb",
            "delta_vs_prior_frontier",
            "difference_sd_units",
            "new_frontier",
        ),
        summary_rows,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(results),
        "heldout_registry_count": len(heldouts),
        "noise": noise,
        "frontier_comparison": summary_rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_rows = []
    for row in summary_rows:
        report_rows.append(
            (
                row["objective"],
                POLICY_LABEL[str(row["policy_class"])],
                row["panel_winner"],
                f"{float(row['panel_winner_bpb']):.6f}",
                f"{float(row['prior_frontier_bpb']):.6f}",
                f"{float(row['delta_vs_prior_frontier']):+.6f}",
                "yes" if row["new_frontier"] else "no",
            )
        )
    report = [
        "# Bucket-family GRP 3e18 validation",
        "",
        f"All {len(results)} materialized candidates completed training and native Table-9 evaluation.",
        "Positive optimism means the surrogate predicted a BPB lower than was observed.",
        "",
        markdown_table(
            report_rows,
            ("Objective", "Policy", "Panel winner", "Observed", "Prior frontier", "Delta", "New frontier"),
        ),
        "",
        "## Objective rankings",
    ]
    for objective in TARGET_COLUMN:
        report.extend(["", f"### {objective}", ""])
        top = sorted(
            (row for row in results if row["objective"] == objective),
            key=lambda row: float(row["observed_target_bpb"]),
        )
        report.append(
            markdown_table(
                [
                    (
                        index,
                        row["candidate"],
                        row["policy"],
                        f"{float(row['observed_target_bpb']):.6f}",
                        f"{float(row['predicted_bpb']):.6f}",
                        f"{float(row['optimism_bpb']):+.6f}",
                    )
                    for index, row in enumerate(top, start=1)
                ],
                ("Rank", "Candidate", "Policy", "Observed", "Predicted", "Optimism"),
            )
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
