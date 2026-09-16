# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Visualize bucket-level exposure of the worst frozen heldout predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "worst_heldout_policy_visualizations"
EPSILON = 1e-6


def relative_log10(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(numerator, EPSILON) / np.maximum(denominator, EPSILON))


def main() -> None:
    gate.assert_sealed_absent(FAILURE_ATLAS)
    gate.assert_sealed_absent(DASHBOARD)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(FAILURE_ATLAS)
    worst = (
        atlas.loc[atlas["mechanism"].eq("baseline")]
        .sort_values(["dataset", "optimism"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(10)
        .copy()
    )
    bundle = json.loads(DASHBOARD.read_text())
    swarm = bundle["swarms"]["delphi_3e18"]
    domains = swarm["domains"]
    domain_ids = [domain["id"] for domain in domains]
    domain_labels = [domain["label"] for domain in domains]
    proportional = np.asarray([domain["proportionalWeight"] for domain in domains], dtype=float)
    reference_phase0 = proportional * np.asarray([domain["phase0EpochFactor"] for domain in domains])
    reference_phase1 = proportional * np.asarray([domain["phase1EpochFactor"] for domain in domains])
    reference_total = reference_phase0 + reference_phase1
    row_by_name = {row["name"]: row for row in swarm["rows"]}

    exposure_rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for record in worst.itertuples(index=False):
        row = row_by_name[record.row_id]
        phase0 = np.asarray(row["phase0Epochs"], dtype=float)
        phase1 = np.asarray(row["phase1Epochs"], dtype=float)
        total = phase0 + phase1
        for index, domain in enumerate(domain_ids):
            exposure_rows.append(
                {
                    "dataset": record.dataset,
                    "row_id": record.row_id,
                    "domain": domain,
                    "phase0_epochs": phase0[index],
                    "phase1_epochs": phase1[index],
                    "total_epochs": total[index],
                    "relative_total_exposure": total[index] / reference_total[index],
                }
            )
        summaries.append(
            {
                "dataset": record.dataset,
                "row_id": record.row_id,
                "observed": record.observed,
                "predicted": record.predicted,
                "optimism": record.optimism,
                "phase_tv": record.phase_tv,
                "max_epoch": float(np.max(total)),
                "buckets_below_quarter_proportional": int(np.sum(total / reference_total < 0.25)),
                "buckets_below_tenth_proportional": int(np.sum(total / reference_total < 0.10)),
            }
        )

    exposures = pd.DataFrame(exposure_rows)
    summary = pd.DataFrame(summaries)
    exposures.to_csv(DEFAULT_OUTPUT / "worst_policy_exposures.csv", index=False)
    summary.to_csv(DEFAULT_OUTPUT / "worst_policy_summary.csv", index=False)

    for dataset, selected in worst.groupby("dataset", sort=True):
        names = selected["row_id"].tolist()
        rows = [row_by_name[name] for name in names]
        phase0 = np.asarray([row["phase0Epochs"] for row in rows], dtype=float)
        phase1 = np.asarray([row["phase1Epochs"] for row in rows], dtype=float)
        total = phase0 + phase1
        panels = (
            ("Aggregate exposure / proportional", relative_log10(total, reference_total[None, :])),
            ("Phase 0 exposure / proportional", relative_log10(phase0, reference_phase0[None, :])),
            ("Phase 1 exposure / proportional", relative_log10(phase1, reference_phase1[None, :])),
        )
        figure = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in panels])
        for column, (_title, values) in enumerate(panels, start=1):
            custom = np.empty((len(names), len(domain_ids), 3), dtype=object)
            custom[:, :, 0] = np.asarray(
                [row["observed"]["table9" if "table9" in dataset else "uncheatable"] for row in rows]
            )[:, None]
            custom[:, :, 1] = selected["predicted"].to_numpy()[:, None]
            custom[:, :, 2] = selected["optimism"].to_numpy()[:, None]
            figure.add_trace(
                go.Heatmap(
                    z=values,
                    x=domain_labels,
                    y=names,
                    customdata=custom,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "%{y}<br>%{x}<br>log10 relative exposure=%{z:.2f}"
                        "<br>observed=%{customdata[0]:.4f}<br>predicted=%{customdata[1]:.4f}"
                        "<br>optimism=%{customdata[2]:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_layout(
            title=f"Worst frozen predictions: {dataset}",
            template="plotly_white",
            coloraxis={
                "colorscale": "RdYlGn_r",
                "cmin": -4.0,
                "cmax": 2.0,
                "colorbar": {"title": "log10 exposure<br>relative to proportional"},
            },
            width=2100,
            height=760,
            margin={"l": 260, "r": 80, "t": 95, "b": 270},
        )
        figure.update_xaxes(tickangle=-55)
        figure.write_html(
            DEFAULT_OUTPUT / f"{dataset}_worst_policy_exposures.html",
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {"format": "png", "scale": 4}},
        )

    aggregate = summary.groupby("dataset", as_index=False).agg(
        median_buckets_below_quarter=("buckets_below_quarter_proportional", "median"),
        max_buckets_below_quarter=("buckets_below_quarter_proportional", "max"),
        median_max_epoch=("max_epoch", "median"),
        max_max_epoch=("max_epoch", "max"),
        median_phase_tv=("phase_tv", "median"),
        worst_optimism=("optimism", "max"),
    )
    report = f"""# Worst heldout policy exposure audit

The ten most optimistic frozen-baseline predictions per target are rendered at bucket resolution. Colors are log10 realized exposure relative to the proportional policy in aggregate and in each phase.

{aggregate.to_markdown(index=False, floatfmt=".4f")}

{summary.to_markdown(index=False, floatfmt=".4f")}

The extreme failures are not one uniform corner: they combine severe multi-bucket starvation, one or more heavily repeated small buckets, and varying phase divergence. This is consistent with bundled unsupported interventions and explains why a single concentration or coverage scalar did not transfer. It does not identify a unique response law.
"""
    (DEFAULT_OUTPUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
