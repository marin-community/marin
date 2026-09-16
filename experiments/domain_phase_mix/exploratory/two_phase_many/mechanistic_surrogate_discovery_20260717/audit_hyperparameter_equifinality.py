# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Audit nonlinear hyperparameter identification across the portfolio screens.

This is a profile-likelihood-style diagnostic over prespecified CV grids. It
does not use heldout outcomes. A mechanism is weakly identified when many
distinct nonlinear settings lie near the best OOF RMSE or when their values
span most of the searched grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "hyperparameter_equifinality_audit"
NEAR_THRESHOLDS = (0.01, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def standardized_screens() -> list[tuple[Path, pd.DataFrame]]:
    screens: list[tuple[Path, pd.DataFrame]] = []
    for path in sorted(ARTIFACT_ROOT.rglob("hyperparameter_screen.csv")):
        gate.assert_sealed_absent(path)
        frame = pd.read_csv(path)
        required = {"panel", "family", "config", "parameters", "l2", "rmse"}
        if required.issubset(frame.columns):
            screens.append((path, frame))
    if not screens:
        raise ValueError("No standardized portfolio screens found")
    return screens


def parameter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        parameters = json.loads(row.parameters)
        parameters["l2"] = float(row.l2)
        records.append(parameters)
    return pd.DataFrame(records, index=frame.index)


def normalized_span(values: pd.Series, full_values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    full_values = pd.to_numeric(full_values, errors="coerce").dropna()
    if full_values.nunique() <= 1:
        return 0.0
    denominator = float(full_values.max() - full_values.min())
    if denominator == 0:
        return 0.0
    return float((values.max() - values.min()) / denominator)


def entropy_fraction(values: pd.Series, full_values: pd.Series) -> float:
    full_levels = max(int(full_values.nunique()), 1)
    if full_levels <= 1 or values.empty:
        return 0.0
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy / np.log(full_levels)


def audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for path, frame in standardized_screens():
        artifact = str(path.parent.relative_to(ARTIFACT_ROOT))
        for (panel, family), group in frame.groupby(["panel", "family"], sort=True):
            group = group.reset_index(drop=True)
            parameters = parameter_frame(group)
            best_index = int(group["rmse"].idxmin())
            best_rmse = float(group.loc[best_index, "rmse"])
            best_parameters = parameters.loc[best_index]
            row: dict[str, object] = {
                "artifact": artifact,
                "panel": panel,
                "family": family,
                "n_grid_rows": len(group),
                "n_shapes": int(group["config"].nunique()),
                "best_rmse": best_rmse,
                "best_config": group.loc[best_index, "config"],
                "best_l2": float(group.loc[best_index, "l2"]),
                "n_nonlinear_parameters": max(len(parameters.columns) - 1, 0),
            }
            for threshold in NEAR_THRESHOLDS:
                suffix = f"{int(threshold * 100)}pct"
                near_mask = group["rmse"].le(best_rmse * (1 + threshold) + 1e-12)
                near = parameters.loc[near_mask]
                row[f"near_{suffix}_rows"] = int(near_mask.sum())
                row[f"near_{suffix}_fraction"] = float(near_mask.mean())
                row[f"near_{suffix}_shape_fraction"] = float(
                    group.loc[near_mask, "config"].nunique() / max(group["config"].nunique(), 1)
                )
                nonlinear_spans: list[float] = []
                nonlinear_entropies: list[float] = []
                for parameter in parameters.columns:
                    values = parameters[parameter]
                    if values.nunique(dropna=True) <= 1:
                        continue
                    near_values = near[parameter].dropna()
                    span = normalized_span(near_values, values)
                    entropy = entropy_fraction(near_values, values)
                    if parameter != "l2":
                        nonlinear_spans.append(span)
                        nonlinear_entropies.append(entropy)
                    parameter_rows.append(
                        {
                            "artifact": artifact,
                            "panel": panel,
                            "family": family,
                            "threshold": threshold,
                            "parameter": parameter,
                            "n_grid_levels": int(values.nunique(dropna=True)),
                            "n_near_levels": int(near_values.nunique(dropna=True)),
                            "grid_min": float(pd.to_numeric(values, errors="coerce").min()),
                            "grid_max": float(pd.to_numeric(values, errors="coerce").max()),
                            "near_level_fraction": float(near_values.nunique(dropna=True) / values.nunique(dropna=True)),
                            "normalized_span": span,
                            "normalized_entropy": entropy,
                            "best_value": float(best_parameters[parameter]),
                            "best_at_lower_boundary": bool(
                                np.isclose(float(best_parameters[parameter]), float(values.min()))
                            ),
                            "best_at_upper_boundary": bool(
                                np.isclose(float(best_parameters[parameter]), float(values.max()))
                            ),
                        }
                    )
                row[f"near_{suffix}_median_nonlinear_span"] = (
                    float(np.median(nonlinear_spans)) if nonlinear_spans else np.nan
                )
                row[f"near_{suffix}_max_nonlinear_span"] = float(np.max(nonlinear_spans)) if nonlinear_spans else np.nan
                row[f"near_{suffix}_median_nonlinear_entropy"] = (
                    float(np.median(nonlinear_entropies)) if nonlinear_entropies else np.nan
                )
            group_rows.append(row)
    groups = pd.DataFrame(group_rows)
    parameters = pd.DataFrame(parameter_rows)
    if groups.empty or parameters.empty:
        raise ValueError("Equifinality audit produced no rows")
    return groups, parameters


def family_summary(groups: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    nonlinear = parameters.loc[(parameters["parameter"] != "l2") & parameters["threshold"].eq(0.01)].copy()
    parameter_summary = nonlinear.groupby("family", as_index=False).agg(
        median_parameter_span=("normalized_span", "median"),
        max_parameter_span=("normalized_span", "max"),
        median_parameter_entropy=("normalized_entropy", "median"),
        boundary_selection_fraction=(
            "best_at_lower_boundary",
            lambda values: float(
                np.mean(
                    values.to_numpy(dtype=bool)
                    | nonlinear.loc[values.index, "best_at_upper_boundary"].to_numpy(dtype=bool)
                )
            ),
        ),
    )
    summary = groups.groupby("family", as_index=False).agg(
        n_panels=("panel", "nunique"),
        n_screen_groups=("panel", "size"),
        median_nonlinear_parameters=("n_nonlinear_parameters", "median"),
        median_near_1pct_fraction=("near_1pct_fraction", "median"),
        max_near_1pct_fraction=("near_1pct_fraction", "max"),
        median_near_1pct_shape_fraction=("near_1pct_shape_fraction", "median"),
    )
    summary = summary.merge(parameter_summary, on="family", how="left")
    has_nonlinear_parameters = summary["median_nonlinear_parameters"].gt(0)
    summary["weakly_identified"] = has_nonlinear_parameters & (
        summary["median_near_1pct_shape_fraction"].ge(0.25)
        | summary["median_parameter_span"].ge(0.5)
        | summary["boundary_selection_fraction"].ge(0.75)
    )
    summary["identifiability_status"] = np.select(
        [~has_nonlinear_parameters, summary["weakly_identified"]],
        ["not_applicable", "weak"],
        default="apparently_localized",
    )
    return summary.sort_values(
        ["weakly_identified", "median_near_1pct_shape_fraction", "median_parameter_span"],
        ascending=[False, False, False],
    )


def cross_panel_stability(parameters: pd.DataFrame) -> pd.DataFrame:
    selected = parameters.loc[parameters["threshold"].eq(0.01) & parameters["parameter"].ne("l2")].drop_duplicates(
        ["artifact", "panel", "family", "parameter"]
    )
    rows: list[dict[str, object]] = []
    for (family, parameter), group in selected.groupby(["family", "parameter"], sort=True):
        grid_min = float(group["grid_min"].min())
        grid_max = float(group["grid_max"].max())
        denominator = grid_max - grid_min
        best_values = group["best_value"].to_numpy(dtype=float)
        modal_fraction = float(group["best_value"].value_counts(normalize=True).max())
        rows.append(
            {
                "family": family,
                "parameter": parameter,
                "n_panels": int(group["panel"].nunique()),
                "n_selected_values": int(group["best_value"].nunique()),
                "selected_min": float(np.min(best_values)),
                "selected_max": float(np.max(best_values)),
                "selected_span_fraction": (
                    float((np.max(best_values) - np.min(best_values)) / denominator) if denominator > 0 else 0.0
                ),
                "modal_selected_fraction": modal_fraction,
                "boundary_selection_fraction": float(
                    np.mean(
                        group["best_at_lower_boundary"].to_numpy(dtype=bool)
                        | group["best_at_upper_boundary"].to_numpy(dtype=bool)
                    )
                ),
                "cross_panel_stable": (
                    bool(
                        (len(group) == 1)
                        or ((np.max(best_values) - np.min(best_values)) / denominator <= 0.25 and modal_fraction >= 0.5)
                    )
                    if denominator > 0
                    else True
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    groups: pd.DataFrame,
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
    cross_panel: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output_dir / "screen_group_equifinality.csv", index=False)
    parameters.to_csv(output_dir / "parameter_profiles.csv", index=False)
    summary.to_csv(output_dir / "family_equifinality_summary.csv", index=False)
    cross_panel.to_csv(output_dir / "cross_panel_selection_stability.csv", index=False)

    plotted = summary.dropna(subset=["median_parameter_span"]).copy()
    figure = px.scatter(
        plotted,
        x="median_near_1pct_shape_fraction",
        y="median_parameter_span",
        color="boundary_selection_fraction",
        size="n_screen_groups",
        hover_name="family",
        hover_data=[
            "n_panels",
            "median_nonlinear_parameters",
            "max_near_1pct_fraction",
            "max_parameter_span",
            "weakly_identified",
        ],
        color_continuous_scale="RdYlGn_r",
        labels={
            "median_near_1pct_shape_fraction": "Median fraction of shapes within 1% of best OOF RMSE",
            "median_parameter_span": "Median near-optimal nonlinear-parameter grid span",
            "boundary_selection_fraction": "Best-setting boundary fraction",
        },
        title="Mechanistic hyperparameter equifinality across CV screens",
    )
    figure.add_vline(x=0.25, line_dash="dash", line_color="#7d8790")
    figure.add_hline(y=0.5, line_dash="dash", line_color="#7d8790")
    figure.update_layout(template="plotly_white", width=1180, height=760)
    figure.write_html(
        output_dir / "hyperparameter_equifinality.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    nonlinear = parameters.loc[(parameters["parameter"] != "l2") & parameters["threshold"].eq(0.01)]
    nonlinear_families = summary.loc[summary["median_nonlinear_parameters"].gt(0)]
    weak = nonlinear_families.loc[nonlinear_families["weakly_identified"]]
    transferable = cross_panel.loc[cross_panel["n_panels"].ge(2)]
    unstable = transferable.loc[~transferable["cross_panel_stable"]]
    report = [
        "# Hyperparameter-equifinality audit",
        "",
        (
            f"This audit covers {len(groups):,} panel-family screen groups, "
            f"{groups['n_grid_rows'].sum():,} cross-validated grid rows, and "
            f"{nonlinear['parameter'].count():,} nonlinear parameter profiles."
        ),
        "",
        (
            f"{len(weak)}/{len(nonlinear_families)} families with nonlinear hyperparameters satisfy at least one "
            "prespecified weak-identification flag: "
            "at least 25% of tested shapes remain within 1% of the best OOF RMSE, the median near-optimal "
            "parameter span covers at least half of its tested grid, or at least 75% of selected settings "
            "land on a grid boundary."
        ),
        "",
        "The diagnostic is deliberately based only on fit-panel CV landscapes. It cannot promote a model and "
        "does not use the frozen heldouts. A broad near-optimal profile means the fitted transition law should "
        "not be interpreted as an identified training dynamic.",
        "",
        (
            f"Across panels, {len(unstable)}/{len(transferable)} nonlinear family-parameter pairs tested on at "
            "least two panels fail the prespecified stability rule: selected values span more than 25% of the "
            "tested grid or no value is selected on at least half of panels."
        ),
        "",
        "## Most weakly identified families",
        "",
        weak.head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Strongest apparent identifiability",
        "",
        nonlinear_families.loc[~nonlinear_families["weakly_identified"]]
        .tail(20)
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Cross-panel unstable parameter selections",
        "",
        unstable.sort_values(["selected_span_fraction", "modal_selected_fraction"], ascending=[False, True])
        .head(30)
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Scope",
        "",
        "The result is grid-relative: it proves equifinality over the predeclared tested settings, not over the "
        "full continuous parameter space. Boundary selections are therefore evidence for an unresolved search "
        "direction, not evidence that the boundary value is physically correct.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    groups, parameters = audit()
    summary = family_summary(groups, parameters)
    cross_panel = cross_panel_stability(parameters)
    write_report(args.output_dir, groups, parameters, summary, cross_panel)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
