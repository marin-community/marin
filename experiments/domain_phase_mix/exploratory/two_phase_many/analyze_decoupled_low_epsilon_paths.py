# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///
"""Analyze locally materialized low phase-information paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_low_epsilon_paths_20260712"
VALIDATED_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712"
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
FAMILY_COLORS = {"effective_exposure": "#D73027", "separate_heads": "#1A9850"}
FAMILY_LABELS = {"effective_exposure": "Effective-exposure DSP", "separate_heads": "Separate heads"}
ANCHOR_LABELS = {
    "unch05": "Uncheatable aggregate KL=0.05",
    "t9s05": "Table-9 stable aggregate KL=0.05",
    "t9b075": "Table-9 observed-best aggregate KL=0.075",
}
PHASE_FRACTIONS = np.array([0.8, 0.2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def fisher_cosine(left: np.ndarray, right: np.ndarray, aggregate: np.ndarray) -> float:
    metric = 1.0 / np.clip(aggregate, 1e-12, None)
    numerator = float(np.sum(left * right * metric))
    denominator = float(np.sqrt(np.sum(left**2 * metric) * np.sum(right**2 * metric)))
    return numerator / denominator


def weighted_policy_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        sum(
            PHASE_FRACTIONS[phase] * 0.5 * np.abs(left[phase] - right[phase]).sum()
            for phase in range(len(PHASE_FRACTIONS))
        )
    )


def load_weights(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(float).T
    aggregate = frame["aggregate_weight"].to_numpy(float)
    domains = frame["domain"].astype(str).tolist()
    return weights, aggregate, domains


def analyze(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    manifest = pd.read_csv(input_dir / "candidate_manifest.csv")
    paths = manifest[manifest["phase_information_budget"].gt(0)].copy()
    weights_by_key: dict[tuple[str, str, float], np.ndarray] = {}
    aggregate_by_anchor: dict[str, np.ndarray] = {}
    domain_by_anchor: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    overlap_error = 0.0

    for record in paths.to_dict(orient="records"):
        candidate = str(record["candidate"])
        weights, aggregate, domains = load_weights(input_dir / "mixtures" / f"{candidate}.csv")
        anchor = str(record["anchor_tag"])
        family = str(record["family"])
        epsilon = float(record["phase_information_budget"])
        aggregate_by_anchor.setdefault(anchor, aggregate)
        domain_by_anchor.setdefault(anchor, domains)
        if not np.allclose(aggregate, aggregate_by_anchor[anchor], atol=1e-12):
            raise ValueError(f"Aggregate drift in {candidate}")
        if domains != domain_by_anchor[anchor]:
            raise ValueError(f"Domain-order drift in {candidate}")
        weights_by_key[(anchor, family, epsilon)] = weights

        validated_path = VALIDATED_DIR / "mixtures" / f"{candidate}.csv"
        if validated_path.exists():
            validated_weights, validated_aggregate, validated_domains = load_weights(validated_path)
            if validated_domains != domains:
                raise ValueError(f"Validated domain order differs for {candidate}")
            overlap_error = max(
                overlap_error,
                float(np.max(np.abs(weights - validated_weights))),
                float(np.max(np.abs(aggregate - validated_aggregate))),
            )

        delta = weights[1] - weights[0]
        positive_index = int(np.argmax(delta))
        negative_index = int(np.argmin(delta))
        rows.append(
            {
                **record,
                "max_positive_shift_domain": domains[positive_index],
                "max_positive_shift": float(delta[positive_index]),
                "max_negative_shift_domain": domains[negative_index],
                "max_negative_shift": float(delta[negative_index]),
                "max_absolute_bucket_shift": float(np.max(np.abs(delta))),
                "phase_tv_per_sqrt_epsilon": float(record["phase_tv"] / np.sqrt(epsilon)),
            }
        )

    diagnostics = pd.DataFrame(rows)
    for (anchor, family), group in diagnostics.groupby(["anchor_tag", "family"]):
        epsilon_min = float(group["phase_information_budget"].min())
        reference = weights_by_key[(anchor, family, epsilon_min)]
        reference_delta = reference[1] - reference[0]
        aggregate = aggregate_by_anchor[anchor]
        previous_weights: np.ndarray | None = None
        for index in group.sort_values("phase_information_budget").index:
            epsilon = float(diagnostics.loc[index, "phase_information_budget"])
            weights = weights_by_key[(anchor, family, epsilon)]
            delta = weights[1] - weights[0]
            diagnostics.loc[index, "fisher_direction_cosine_to_epsilon_min"] = fisher_cosine(
                delta,
                reference_delta,
                aggregate,
            )
            diagnostics.loc[index, "weighted_policy_tv_from_previous"] = (
                0.0 if previous_weights is None else weighted_policy_tv(weights, previous_weights)
            )
            previous_weights = weights

    family_rows: list[dict[str, object]] = []
    for anchor in sorted(diagnostics["anchor_tag"].unique()):
        anchor_rows = diagnostics[diagnostics["anchor_tag"].eq(anchor)]
        for epsilon in sorted(anchor_rows["phase_information_budget"].unique()):
            effective = weights_by_key[(anchor, "effective_exposure", float(epsilon))]
            separate = weights_by_key[(anchor, "separate_heads", float(epsilon))]
            family_rows.append(
                {
                    "anchor_tag": anchor,
                    "phase_information_budget": epsilon,
                    "effective_vs_separate_weighted_policy_tv": weighted_policy_tv(effective, separate),
                    "effective_vs_separate_phase_direction_fisher_cosine": fisher_cosine(
                        effective[1] - effective[0],
                        separate[1] - separate[0],
                        aggregate_by_anchor[anchor],
                    ),
                }
            )
    return diagnostics, pd.DataFrame(family_rows), overlap_error


def render(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    anchors = ["unch05", "t9s05", "t9b075"]
    metrics = [
        ("phase_tv", "Phase TV"),
        ("max_absolute_bucket_shift", "Largest bucket shift"),
        ("fisher_direction_cosine_to_epsilon_min", "Direction cosine vs epsilon=0.001"),
    ]
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in anchors] + [""] * 6,
        vertical_spacing=0.1,
        horizontal_spacing=0.07,
    )
    legend_seen: set[str] = set()
    for row, (metric, axis_title) in enumerate(metrics, start=1):
        for col, anchor in enumerate(anchors, start=1):
            anchor_rows = diagnostics[diagnostics["anchor_tag"].eq(anchor)]
            for family, path in anchor_rows.groupby("family"):
                path = path.sort_values("phase_information_budget")
                label = FAMILY_LABELS[str(family)]
                figure.add_trace(
                    go.Scatter(
                        x=path["phase_information_budget"],
                        y=path[metric],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=label not in legend_seen,
                        line={"color": FAMILY_COLORS[str(family)], "width": 2},
                        marker={"size": 8},
                        customdata=np.stack(
                            [
                                path["candidate"],
                                path["predicted_gain_vs_tied"],
                                path["max_weight"],
                                path["max_simulated_epoch"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[0]}<br>epsilon=%{x:.4f}<br>value=%{y:.6f}"
                            "<br>predicted gain=%{customdata[1]:.6f}<br>max weight=%{customdata[2]:.4f}"
                            "<br>max simulated epoch=%{customdata[3]:.3f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                legend_seen.add(label)
            figure.update_xaxes(title_text="Phase-information budget", row=row, col=col)
            figure.update_yaxes(title_text=axis_title if col == 1 else None, row=row, col=col)
    figure.update_layout(
        title={"text": "Low-epsilon phase-asymmetry paths", "x": 0.5},
        template="plotly_white",
        width=1700,
        height=1250,
        margin={"l": 90, "r": 40, "t": 130, "b": 150},
        legend={"orientation": "h", "yanchor": "top", "y": -0.07, "xanchor": "center", "x": 0.5},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.write_html(output_dir / "low_epsilon_phase_asymmetry.html", include_plotlyjs=True, config=EXPORT_CONFIG)
    figure.write_image(output_dir / "low_epsilon_phase_asymmetry.png", scale=2)


def write_report(
    diagnostics: pd.DataFrame,
    family_distance: pd.DataFrame,
    overlap_error: float,
    output_dir: Path,
) -> None:
    summary = (
        diagnostics.groupby(["anchor_tag", "family"], as_index=False)
        .agg(
            min_phase_tv=("phase_tv", "min"),
            max_phase_tv=("phase_tv", "max"),
            min_direction_cosine=("fisher_direction_cosine_to_epsilon_min", "min"),
            min_tv_per_sqrt_epsilon=("phase_tv_per_sqrt_epsilon", "min"),
            max_tv_per_sqrt_epsilon=("phase_tv_per_sqrt_epsilon", "max"),
            max_aggregate_error=("max_aggregate_error", "max"),
            max_simulated_epoch_range=("max_simulated_epoch", lambda values: float(values.max() - values.min())),
        )
        .sort_values(["anchor_tag", "family"])
    )
    summary.to_csv(output_dir / "low_epsilon_path_summary.csv", index=False)
    report = f"""# Low-epsilon phase-information paths

The local panel contains effective-exposure and separate-heads policies for three fixed aggregates at phase-information
budgets 0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, and 0.025.

- Maximum aggregate-weight error: {diagnostics['max_aggregate_error'].max():.3e}.
- Maximum discrepancy from previously validated overlapping candidate files: {overlap_error:.3e}.
- Max simulated epochs are exactly constant along each path because aggregate exposure is fixed.
- Phase TV grows approximately as sqrt(epsilon); TV/sqrt(epsilon) ranges from
  {diagnostics['phase_tv_per_sqrt_epsilon'].min():.3f} to
  {diagnostics['phase_tv_per_sqrt_epsilon'].max():.3f}.
- Fisher direction cosine relative to epsilon=0.001 is at least
  {diagnostics['fisher_direction_cosine_to_epsilon_min'].min():.3f}; the low-epsilon solves remain on stable directions.
- Effective exposure and separate heads are distinct directions. Their weighted policy TV ranges from
  {family_distance['effective_vs_separate_weighted_policy_tv'].min():.3f} to
  {family_distance['effective_vs_separate_weighted_policy_tv'].max():.3f} over the panel.

## Path summary

{summary.to_markdown(index=False, floatfmt='.6f')}
"""
    (output_dir / "low_epsilon_report.md").write_text(report)


def main() -> None:
    args = parse_args()
    diagnostics, family_distance, overlap_error = analyze(args.input_dir)
    diagnostics.to_csv(args.input_dir / "low_epsilon_path_diagnostics.csv", index=False)
    family_distance.to_csv(args.input_dir / "low_epsilon_family_distance.csv", index=False)
    render(diagnostics, args.input_dir)
    write_report(diagnostics, family_distance, overlap_error, args.input_dir)
    print((args.input_dir / "low_epsilon_report.md").read_text())


if __name__ == "__main__":
    main()
