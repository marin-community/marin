# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Audit matched-policy and round-one mechanism transfer from 300M to Delphi 3e18."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DISCOVERY_OUTPUT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = DISCOVERY_OUTPUT / "round1_cross_scale_matched_policy"
FREEZE_PATH = DISCOVERY_OUTPUT / "round1_candidate_freeze/candidate_freeze.json"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COLORS = {"one_phase": "#e66b2e", "two_phase": "#17384a", "phase_delta": "#2f855a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def matched_indices(source: models.PairedPanel, target: models.PairedPanel) -> tuple[np.ndarray, np.ndarray]:
    if source.domain_names != target.domain_names:
        raise ValueError("Cross-scale panels do not share domain ordering")
    distance = np.max(np.abs(source.weights[:, None, :, :] - target.weights[None, :, :, :]), axis=(2, 3))
    nearest = np.argmin(distance, axis=1)
    minimum = np.min(distance, axis=1)
    matched = minimum < 1e-10
    if len(set(nearest[matched].tolist())) != int(matched.sum()):
        raise ValueError("Cross-scale policy alignment is not one-to-one")
    return np.flatnonzero(matched), nearest[matched]


def relation_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(source) & np.isfinite(target)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        raise ValueError("Cross-scale metric needs at least three matched observations")
    slope, intercept = np.polyfit(source, target, 1)
    return {
        "n": len(source),
        "pearson": float(pearsonr(source, target).statistic),
        "spearman": float(spearmanr(source, target).statistic),
        "delphi_on_300m_slope": float(slope),
        "delphi_on_300m_intercept": float(intercept),
        "mean_300m": float(np.mean(source)),
        "mean_delphi": float(np.mean(target)),
        "standard_deviation_300m": float(np.std(source, ddof=1)),
        "standard_deviation_delphi": float(np.std(target, ddof=1)),
    }


def candidate_configs(
    freeze: dict[str, Any],
    target: str,
) -> list[tuple[str, models.PhaseFamily, models.PhaseConfig, models.AggregateConfig]]:
    result: list[tuple[str, models.PhaseFamily, models.PhaseConfig, models.AggregateConfig]] = []
    for candidate, payload in freeze["candidate_definitions"].items():
        aggregate = models.AggregateConfig(**payload["aggregate_config"])
        config = payload["target_configs"][target]
        if candidate == "paired_marginal_value_transport":
            family = models.PhaseFamily.PMVT
            phase_config: models.PhaseConfig = models.PMVTConfig(**config)
        elif candidate == "terminal_equilibrium_adaptation":
            family = models.PhaseFamily.TERMINAL_EQUILIBRIUM
            phase_config = models.TerminalEquilibriumConfig(**config)
        else:
            raise ValueError(f"Unknown frozen candidate {candidate}")
        result.append((candidate, family, phase_config, aggregate))
    return result


def phase_transfer_metrics(
    source: models.PairedPanel,
    target: models.PairedPanel,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    family: models.PhaseFamily,
    config: models.PhaseConfig,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    source_train = np.flatnonzero(source.paired_mask)
    target_valid = target.paired_mask[target_indices]
    source_indices = source_indices[target_valid]
    target_indices = target_indices[target_valid]
    model = models.fit_phase(source, source_train, family, config)
    prediction = model.predict_delta(target.weights[target_indices])
    observed = target.two_phase_target[target_indices] - target.one_phase_target[target_indices]
    summary = screen.scalar_metrics(observed, prediction)
    summary["source_scale"] = source.name.split("_", 1)[0]
    summary["target_scale"] = target.name.split("_", 1)[0]
    coefficients = pd.DataFrame(
        {
            "feature": model.head.feature_names,
            "coefficient": model.head.coefficients_in_natural_units,
        }
    )
    predictions = pd.DataFrame(
        {
            "source_index": source_indices,
            "target_index": target_indices,
            "observed_delta": observed,
            "predicted_delta": prediction,
        }
    )
    return summary, coefficients, predictions


def plot_matched(frame: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("One-phase policies", "Two-phase policies", "Two-minus-one phase effect")
    )
    for column, policy_class in enumerate(("one_phase", "two_phase", "phase_delta"), start=1):
        local = frame.loc[frame["policy_class"].eq(policy_class)]
        minimum = float(min(local["value_300m"].min(), local["value_delphi"].min()))
        maximum = float(max(local["value_300m"].max(), local["value_delphi"].max()))
        padding = 0.04 * max(maximum - minimum, 1e-3)
        fig.add_trace(
            go.Scatter(
                x=[minimum - padding, maximum + padding],
                y=[minimum - padding, maximum + padding],
                mode="lines",
                line={"color": "#8c979e", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        for target, symbol in (("uncheatable", "circle"), ("table9", "diamond")):
            subset = local.loc[local["target"].eq(target)]
            fig.add_trace(
                go.Scatter(
                    x=subset["value_300m"],
                    y=subset["value_delphi"],
                    mode="markers",
                    name=f"{target} · {policy_class}",
                    marker={"color": COLORS[policy_class], "symbol": symbol, "size": 7, "opacity": 0.7},
                    customdata=np.column_stack([subset["source_index"], subset["target_index"]]),
                    hovertemplate="300M=%{x:.5f}<br>Delphi=%{y:.5f}<br>indices=%{customdata[0]}→%{customdata[1]}<extra></extra>",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(title_text="300M BPB or phase effect", row=1, col=column)
        if column == 1:
            fig.update_yaxes(title_text="Delphi 3e18 BPB or phase effect", row=1, col=column)
    fig.update_layout(
        title="Matched-policy scale transfer: identical mixture coordinates",
        template="plotly_white",
        width=1600,
        height=580,
        legend={"orientation": "h", "y": 1.12},
    )
    fig.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE_PATH.read_text())
    correlation_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for target in ("uncheatable", "table9"):
        panel_300m = screen.load_panel("300m", target)
        panel_delphi = screen.load_panel("delphi_3e18", target)
        indices_300m, indices_delphi = matched_indices(panel_300m, panel_delphi)
        if len(indices_300m) != 280:
            raise ValueError(f"Expected 280 matched two-phase coordinates, found {len(indices_300m)}")

        pairs = {
            "two_phase": (panel_300m.two_phase_target[indices_300m], panel_delphi.two_phase_target[indices_delphi]),
        }
        paired = panel_300m.paired_mask[indices_300m] & panel_delphi.paired_mask[indices_delphi]
        if int(paired.sum()) != 238:
            raise ValueError(f"Expected 238 matched one-phase outcomes, found {int(paired.sum())}")
        pairs["one_phase"] = (
            panel_300m.one_phase_target[indices_300m[paired]],
            panel_delphi.one_phase_target[indices_delphi[paired]],
        )
        pairs["phase_delta"] = (
            panel_300m.two_phase_target[indices_300m[paired]] - panel_300m.one_phase_target[indices_300m[paired]],
            panel_delphi.two_phase_target[indices_delphi[paired]]
            - panel_delphi.one_phase_target[indices_delphi[paired]],
        )

        for policy_class, (values_300m, values_delphi) in pairs.items():
            correlation_rows.append(
                {"target": target, "policy_class": policy_class, **relation_metrics(values_300m, values_delphi)}
            )
            source_local = indices_300m if policy_class == "two_phase" else indices_300m[paired]
            target_local = indices_delphi if policy_class == "two_phase" else indices_delphi[paired]
            for source_index, target_index, value_300m, value_delphi in zip(
                source_local, target_local, values_300m, values_delphi, strict=True
            ):
                matched_rows.append(
                    {
                        "target": target,
                        "policy_class": policy_class,
                        "source_index": int(source_index),
                        "target_index": int(target_index),
                        "value_300m": float(value_300m),
                        "value_delphi": float(value_delphi),
                    }
                )

        for candidate, family, phase_config, _aggregate_config in candidate_configs(freeze, target):
            for source, destination, source_indices, destination_indices in (
                (panel_300m, panel_delphi, indices_300m, indices_delphi),
                (panel_delphi, panel_300m, indices_delphi, indices_300m),
            ):
                summary, coefficients, predictions = phase_transfer_metrics(
                    source, destination, source_indices, destination_indices, family, phase_config
                )
                direction = f"{summary['source_scale']}_to_{summary['target_scale']}"
                transfer_rows.append({"target": target, "candidate": candidate, "direction": direction, **summary})
                for row in coefficients.to_dict("records"):
                    coefficient_rows.append(
                        {"target": target, "candidate": candidate, "fit_scale": summary["source_scale"], **row}
                    )
                for row in predictions.to_dict("records"):
                    prediction_rows.append({"target": target, "candidate": candidate, "direction": direction, **row})

    correlations = pd.DataFrame(correlation_rows)
    matched = pd.DataFrame(matched_rows)
    transfers = pd.DataFrame(transfer_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.DataFrame(prediction_rows)
    correlations.to_csv(args.output_dir / "matched_target_correlations.csv", index=False)
    matched.to_csv(args.output_dir / "matched_targets.csv", index=False)
    transfers.to_csv(args.output_dir / "phase_mechanism_transfer_metrics.csv", index=False)
    coefficients.to_csv(args.output_dir / "phase_coefficients_by_scale.csv", index=False)
    predictions.to_csv(args.output_dir / "phase_transfer_predictions.csv", index=False)
    plot_matched(matched, args.output_dir / "matched_policy_scale_transfer.html")

    report = [
        "# Round-one matched-policy cross-scale audit",
        "",
        "All rows compare exactly identical mixture coordinates. Two-phase targets use all 280 coordinates; one-phase targets and two-minus-one effects use the 238 coordinates with a newly trained Delphi one-phase counterpart.",
        "",
        "## Observed target transfer",
        "",
        correlations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Frozen phase-law transfer",
        "",
        transfers.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The transfer predictions use coefficients fit at one scale without fitting an output calibration on the destination scale. They therefore test the phase law in BPB units rather than merely its ranking.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
