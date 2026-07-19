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
"""Test whether fit-panel OOF residual uncertainty covers frozen heldouts.

This is an uncertainty-transfer falsification, not a surrogate or a post-hoc
calibration method. One-sided upper prediction bounds are estimated entirely
from the strongest baseline's fit-panel OOF residuals, then frozen before they
are evaluated on policy-matched development heldouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "oof_uncertainty_transfer_audit"
SOURCE_PREDICTIONS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/predictions.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
VARIANT = "inverse_power_deficit_early_family_asymmetric_surplus"
TARGET_CONFIGS = {
    "delphi_3e18_uncheatable": ("identity_raw_bpb", 0.0, 0.001),
    "delphi_3e18_table9": ("log_reducible_bpb", 0.75, 0.01),
}
COVERAGE_LEVELS = (0.8, 0.9, 0.95, 0.975, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    """Return the finite-sample split-conformal quantile."""

    rank = min(int(np.ceil((len(values) + 1) * coverage)), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def selected_source_rows(source: pd.DataFrame, dataset: str) -> pd.DataFrame:
    link, floor_fraction, l2 = TARGET_CONFIGS[dataset]
    selected = source.loc[
        source["dataset"].eq(dataset)
        & source["deficit_variant"].eq(VARIANT)
        & source["link"].eq(link)
        & np.isclose(source["floor_fraction"], floor_fraction)
        & np.isclose(source["l2"], l2)
    ].copy()
    if set(selected["split"]) != {"fit_oof", "heldout"}:
        raise ValueError(f"Incomplete source predictions for {dataset}")
    return selected


def audit_target(
    source: pd.DataFrame,
    atlas: pd.DataFrame,
    dataset: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = selected_source_rows(source, dataset)
    fit = selected.loc[selected["split"].eq("fit_oof")].copy()
    heldout_ids = set(atlas.loc[atlas["dataset"].eq(dataset) & atlas["mechanism"].eq("baseline"), "row_id"].astype(str))
    heldout = selected.loc[selected["split"].eq("heldout") & selected["row_id"].astype(str).isin(heldout_ids)].copy()
    if len(fit) != 280 or len(heldout) != 259:
        raise ValueError(f"Unexpected fit/heldout counts for {dataset}: {len(fit)}/{len(heldout)}")

    fit_signed = (fit["observed"] - fit["predicted"]).to_numpy(dtype=float)
    fit_absolute = np.abs(fit_signed)
    heldout_signed = (heldout["observed"] - heldout["predicted"]).to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    row_level: list[dict[str, object]] = []
    for coverage in COVERAGE_LEVELS:
        one_sided_radius = conformal_quantile(fit_signed, coverage)
        symmetric_radius = conformal_quantile(fit_absolute, coverage)
        one_sided_covered = heldout_signed <= one_sided_radius
        symmetric_covered = np.abs(heldout_signed) <= symmetric_radius
        rows.append(
            {
                "dataset": dataset,
                "nominal_coverage": coverage,
                "fit_one_sided_radius": one_sided_radius,
                "fit_symmetric_radius": symmetric_radius,
                "heldout_one_sided_coverage": float(np.mean(one_sided_covered)),
                "heldout_symmetric_coverage": float(np.mean(symmetric_covered)),
                "one_sided_coverage_gap": float(np.mean(one_sided_covered) - coverage),
                "symmetric_coverage_gap": float(np.mean(symmetric_covered) - coverage),
                "one_sided_miss_count": int(np.sum(~one_sided_covered)),
                "symmetric_miss_count": int(np.sum(~symmetric_covered)),
                "one_sided_excess_miss_p_value": float(
                    binom.sf(int(np.sum(~one_sided_covered)) - 1, len(heldout), 1.0 - coverage)
                ),
                "max_one_sided_exceedance": float(np.max(heldout_signed - one_sided_radius)),
            }
        )
        for row_id, observed, predicted, residual, covered in zip(
            heldout["row_id"],
            heldout["observed"],
            heldout["predicted"],
            heldout_signed,
            one_sided_covered,
            strict=True,
        ):
            row_level.append(
                {
                    "dataset": dataset,
                    "nominal_coverage": coverage,
                    "row_id": row_id,
                    "observed": observed,
                    "predicted": predicted,
                    "optimism": residual,
                    "one_sided_upper_bound": predicted + one_sided_radius,
                    "one_sided_covered": bool(covered),
                }
            )
    return rows, row_level


def render(summary: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("One-sided upper coverage", "Fit-derived uncertainty radius"),
    )
    colors = {
        "delphi_3e18_uncheatable": "#1a9850",
        "delphi_3e18_table9": "#d73027",
    }
    for dataset, local in summary.groupby("dataset", sort=True):
        figure.add_trace(
            go.Scatter(
                x=local["nominal_coverage"],
                y=local["heldout_one_sided_coverage"],
                mode="lines+markers",
                name=dataset,
                line={"color": colors[dataset]},
                customdata=local[["one_sided_miss_count", "max_one_sided_exceedance"]],
                hovertemplate=(
                    "nominal=%{x:.3f}<br>heldout=%{y:.3f}<br>misses=%{customdata[0]}"
                    "<br>max exceedance=%{customdata[1]:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=local["nominal_coverage"],
                y=local["fit_one_sided_radius"],
                mode="lines+markers",
                name=dataset,
                legendgroup=dataset,
                showlegend=False,
                line={"color": colors[dataset]},
            ),
            row=1,
            col=2,
        )
    figure.add_trace(
        go.Scatter(
            x=[min(COVERAGE_LEVELS), max(COVERAGE_LEVELS)],
            y=[min(COVERAGE_LEVELS), max(COVERAGE_LEVELS)],
            mode="lines",
            line={"dash": "dash", "color": "#64748b"},
            name="nominal",
        ),
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Nominal fit-OOF coverage")
    figure.update_yaxes(title_text="Frozen-heldout coverage", row=1, col=1)
    figure.update_yaxes(title_text="One-sided BPB radius", row=1, col=2)
    figure.update_layout(
        title="Fit-panel residual uncertainty does not certify extreme optimism",
        template="plotly_white",
        width=1350,
        height=620,
        legend={"orientation": "h", "y": -0.18},
    )
    figure.write_html(output, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    args = parse_args()
    for path in (SOURCE_PREDICTIONS, FAILURE_ATLAS):
        gate.assert_sealed_absent(path)
    source = pd.read_csv(SOURCE_PREDICTIONS)
    atlas = pd.read_csv(FAILURE_ATLAS)
    summary_rows: list[dict[str, object]] = []
    row_level: list[dict[str, object]] = []
    for dataset in TARGET_CONFIGS:
        summary, rows = audit_target(source, atlas, dataset)
        summary_rows.extend(summary)
        row_level.extend(rows)
    summary = pd.DataFrame(summary_rows)
    rows = pd.DataFrame(row_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "uncertainty_transfer_summary.csv", index=False)
    rows.to_csv(args.output_dir / "uncertainty_transfer_rows.csv", index=False)
    render(summary, args.output_dir / "oof_uncertainty_transfer.html")
    report = [
        "# Fit-OOF uncertainty transfer audit",
        "",
        "One-sided upper prediction bounds are frozen from fit-panel OOF residuals. Heldouts do not select a radius or alter predictions.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Central coverage is often preserved or conservative. The relevant failure is tail certification: high-confidence bounds remain wide yet can still miss the extreme optimistic policies by much more than their fit-derived radius.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
