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
"""Evaluate frozen round-one candidates on historical 3e18 heldouts only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    freeze_pareto_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DISCOVERY_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = DISCOVERY_OUTPUT / "round1_historical_heldouts"
FREEZE_PATH = DISCOVERY_OUTPUT / "round1_candidate_freeze/candidate_freeze.json"
HELDOUT_PATH = RESEARCH_DIR / "reference_outputs/delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
BASELINE_PATH = DISCOVERY_OUTPUT / "frozen_gate/baseline_metrics.csv"
ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
ADVERSARIAL_SERIES = "delphi_3e18_adversarial_stress_panel_20260716"
TARGET_COLUMN = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COLORS = {"paired_marginal_value_transport": "#2f6f8f", "terminal_equilibrium_adaptation": "#d65f35"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze["historical_outcomes_used_for_selection"]:
        raise ValueError("Candidate freeze already records historical selection")
    if freeze["adversarial_outcomes_used_for_selection"]:
        raise ValueError("Candidate freeze records adversarial selection")
    return freeze


def load_historical_rows(domain_names: tuple[str, ...]) -> pd.DataFrame:
    """Skip excluded series before parsing any target or mixture payload."""

    rows: list[dict[str, Any]] = []
    with HELDOUT_PATH.open(newline="") as source:
        for raw in csv.DictReader(source):
            if raw["training_series"] in {ONE_PHASE_SERIES, ADVERSARIAL_SERIES}:
                continue
            if raw["fit_panel_overlap"] != "coordinate_disjoint":
                continue
            if raw["training_state"] != "finished" or raw["checkpoint_declared_complete"] != "1":
                continue
            phase0 = json.loads(raw["phase_0_weights_json"])
            phase1 = json.loads(raw["phase_1_weights_json"])
            row = {
                "heldout_id": raw["heldout_id"],
                "run_name": raw["wandb_run_name"],
                "training_series": raw["training_series"],
                "policy_class": "single_phase" if raw["policy_class"] == "single_phase_tied" else "two_phase",
                "phase_0_fraction": float(raw["phase_0_fraction"]),
                "uncheatable": float(raw[TARGET_COLUMN["uncheatable"]]),
                "table9": float(raw[TARGET_COLUMN["table9"]]),
                "weights": np.asarray(
                    [
                        [float(phase0[domain]) for domain in domain_names],
                        [float(phase1[domain]) for domain in domain_names],
                    ],
                    dtype=float,
                ),
            }
            rows.append(row)
    if len(rows) != 352:
        raise ValueError(f"Expected 352 historical coordinate-disjoint heldouts, found {len(rows)}")
    if any(not np.allclose(row["weights"].sum(axis=1), 1.0, atol=1e-8) for row in rows):
        raise ValueError("Historical heldout contains a non-normalized mixture")
    return pd.DataFrame(rows)


def aggregate_config(config: dict[str, Any]) -> models.AggregateConfig:
    return models.AggregateConfig(**config)


def phase_config(candidate: str, config: dict[str, Any]) -> models.PhaseConfig:
    if candidate == "paired_marginal_value_transport":
        return models.PMVTConfig(**config)
    if candidate == "terminal_equilibrium_adaptation":
        return models.TerminalEquilibriumConfig(**config)
    raise ValueError(f"Unknown candidate {candidate}")


def phase_family(candidate: str) -> models.PhaseFamily:
    if candidate == "paired_marginal_value_transport":
        return models.PhaseFamily.PMVT
    if candidate == "terminal_equilibrium_adaptation":
        return models.PhaseFamily.TERMINAL_EQUILIBRIUM
    raise ValueError(f"Unknown candidate {candidate}")


def nearest_support_metrics(panel: models.PairedPanel, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aggregate_fit = panel.aggregate_weights
    aggregate_query = panel.alpha0 * weights[:, 0, :] + panel.alpha1 * weights[:, 1, :]
    aggregate_tv = 0.5 * np.abs(aggregate_query[:, None, :] - aggregate_fit[None, :, :]).sum(axis=2)
    policy_tv = 0.5 * (
        panel.alpha0 * np.abs(weights[:, None, 0, :] - panel.weights[None, :, 0, :]).sum(axis=2)
        + panel.alpha1 * np.abs(weights[:, None, 1, :] - panel.weights[None, :, 1, :]).sum(axis=2)
    )
    return np.min(aggregate_tv, axis=1), np.min(policy_tv, axis=1)


def exposure_diagnostics(panel: models.PairedPanel, weights: np.ndarray) -> dict[str, np.ndarray]:
    exposure = weights[:, 0, :] * panel.c0[None, :] + weights[:, 1, :] * panel.c1[None, :]
    relative = exposure / np.maximum(panel.proportional_exposure[None, :], 1e-12)
    phase_divergence = 0.5 * np.abs(weights[:, 1, :] - weights[:, 0, :]).sum(axis=1)
    return {
        "minimum_relative_exposure": np.min(relative, axis=1),
        "underquarter_bucket_count": np.sum(relative < 0.25, axis=1),
        "maximum_simulated_epochs": np.max(exposure, axis=1),
        "maximum_bucket_weight": np.max(weights, axis=(1, 2)),
        "phase_total_variation": phase_divergence,
    }


def support_split(distance: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(distance, [1 / 3, 2 / 3])
    return np.asarray(
        ["near" if value <= quantiles[0] else "middle" if value <= quantiles[1] else "far" for value in distance]
    )


def metric_record(
    target: str,
    candidate: str,
    split: str,
    observed: Iterable[float],
    predicted: Iterable[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, bins = gate.metrics(observed, predicted)
    return (
        {"target": target, "candidate": candidate, "split": split, **summary},
        [{"target": target, "candidate": candidate, "split": split, **row} for row in bins],
    )


def plot_predictions(frame: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable prediction",
            "Table-9 prediction",
            "Uncheatable residual",
            "Table-9 residual",
        ),
    )
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        subset = frame.loc[frame["target"].eq(target)]
        minimum = float(min(subset["observed"].min(), subset["predicted"].min()))
        maximum = float(max(subset["observed"].max(), subset["predicted"].max()))
        fig.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                line={"dash": "dash", "color": "#83909a"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=[minimum, maximum], y=[0, 0], mode="lines", line={"dash": "dash", "color": "#83909a"}, showlegend=False
            ),
            row=2,
            col=column,
        )
        for candidate in COLORS:
            local = subset.loc[subset["candidate"].eq(candidate)]
            custom = np.column_stack([local["run_name"], local["training_series"], local["policy_class"]])
            label = "PMVT" if candidate.startswith("paired") else "Terminal equilibrium"
            fig.add_trace(
                go.Scatter(
                    x=local["predicted"],
                    y=local["observed"],
                    mode="markers",
                    name=label,
                    legendgroup=candidate,
                    marker={"color": COLORS[candidate], "size": 7, "opacity": 0.75},
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{customdata[1]}<br>policy=%{customdata[2]}"
                        "<br>pred=%{x:.5f}<br>obs=%{y:.5f}<extra></extra>"
                    ),
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"] - local["observed"],
                    mode="markers",
                    name=label,
                    legendgroup=candidate,
                    marker={"color": COLORS[candidate], "size": 7, "opacity": 0.75},
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{customdata[1]}<br>policy=%{customdata[2]}"
                        "<br>obs=%{x:.5f}<br>residual=%{y:.5f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    fig.update_xaxes(title_text="Predicted BPB", row=1)
    fig.update_yaxes(title_text="Observed BPB", row=1, col=1)
    fig.update_xaxes(title_text="Observed BPB", row=2)
    fig.update_yaxes(title_text="Predicted - observed", row=2, col=1)
    fig.update_layout(
        title="Frozen round-one candidates on 352 historical Delphi 3e18 heldouts",
        template="plotly_white",
        height=960,
        width=1500,
        legend={"orientation": "h", "y": 1.06},
    )
    fig.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def baseline_reference() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_PATH)
    return baseline.loc[
        baseline["swarm"].eq("delphi_3e18")
        & baseline["policy"].eq("two_phase")
        & baseline["split"].isin(("historical_352", "historical_352__single_phase", "historical_352__two_phase"))
    ].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = load_freeze()
    candidate_defs = freeze["candidate_definitions"]
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for target in ("uncheatable", "table9"):
        panel = paired_screen.load_panel("delphi_3e18", target)
        heldout = load_historical_rows(panel.domain_names)
        weights = np.stack(heldout["weights"].to_list())
        aggregate_distance, policy_distance = nearest_support_metrics(panel, weights)
        support = support_split(policy_distance)
        diagnostics = exposure_diagnostics(panel, weights)
        for candidate, definition in candidate_defs.items():
            aggregate_model = models.fit_aggregate(
                panel,
                np.arange(panel.n),
                aggregate_config(definition["aggregate_config"]),
            )
            phase_model = models.fit_phase(
                panel,
                np.arange(panel.n),
                phase_family(candidate),
                phase_config(candidate, definition["target_configs"][target]),
            )
            predicted = models.predict_combined(aggregate_model, phase_model, weights)
            observed = heldout[target].to_numpy(dtype=float)
            local = heldout.drop(columns=["weights"]).copy()
            local["target"] = target
            local["candidate"] = candidate
            local["observed"] = observed
            local["predicted"] = predicted
            local["residual"] = predicted - observed
            local["optimism"] = observed - predicted
            local["nearest_aggregate_tv"] = aggregate_distance
            local["nearest_policy_tv"] = policy_distance
            local["support_tertile"] = support
            for name, values in diagnostics.items():
                local[name] = values
            prediction_rows.extend(local.to_dict(orient="records"))
            specs = [("historical_352", np.ones(len(local), dtype=bool))]
            specs.extend(
                (f"historical_352__{policy}", local["policy_class"].eq(policy).to_numpy())
                for policy in ("single_phase", "two_phase")
            )
            specs.extend(
                (f"historical_352__support_{level}", local["support_tertile"].eq(level).to_numpy())
                for level in ("near", "middle", "far")
            )
            for split, mask in specs:
                record, bins = metric_record(target, candidate, split, observed[mask], predicted[mask])
                metric_rows.append(record)
                bin_rows.extend(bins)
            worst = local.nlargest(10, "optimism").copy()
            worst["rank"] = np.arange(1, len(worst) + 1)
            worst_rows.extend(worst.to_dict(orient="records"))
            for rank, index in enumerate(np.argsort(observed - predicted)[-10:][::-1], start=1):
                relative_exposure = (weights[index, 0] * panel.c0 + weights[index, 1] * panel.c1) / np.maximum(
                    panel.proportional_exposure, 1e-12
                )
                top = np.argsort(relative_exposure)[:8]
                exposure_rows.extend(
                    {
                        "target": target,
                        "candidate": candidate,
                        "optimism_rank": rank,
                        "heldout_id": heldout.iloc[index]["heldout_id"],
                        "domain": panel.domain_names[domain],
                        "relative_exposure": float(relative_exposure[domain]),
                        "phase_0_weight": float(weights[index, 0, domain]),
                        "phase_1_weight": float(weights[index, 1, domain]),
                    }
                    for domain in top
                )
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    worst = pd.DataFrame(worst_rows)
    exposures = pd.DataFrame(exposure_rows)
    predictions.to_csv(args.output_dir / "historical_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "historical_metrics.csv", index=False)
    bins.to_csv(args.output_dir / "calibration_bins.csv", index=False)
    worst.to_csv(args.output_dir / "worst_predictions.csv", index=False)
    exposures.to_csv(args.output_dir / "worst_underexposed_buckets.csv", index=False)
    baseline_reference().to_csv(args.output_dir / "frozen_baseline_reference.csv", index=False)
    plot_predictions(predictions, args.output_dir / "historical_predictions_and_residuals.html")

    headline = metrics.loc[metrics["split"].eq("historical_352")]
    report_lines = [
        "# Round-one historical heldout audit",
        "",
        f"Candidate freeze: `{freeze['freeze_sha256']}`.",
        "",
        "The 120-row adversarial development panel was skipped before parsing target values. The 238 matched one-phase rows remain fitting data, and the untouched frontier phase-fiber panel is absent.",
        "",
        "## Headline metrics",
        "",
        headline[
            [
                "target",
                "candidate",
                "rmse",
                "spearman",
                "bias_predicted_minus_observed",
                "calibration_slope_observed_on_predicted",
                "regret_at_1",
                "optimism_gt_0p05_count",
                "worst_optimism",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Interpretation",
        "",
    ]
    for target in ("uncheatable", "table9"):
        local = headline.loc[headline["target"].eq(target)].sort_values("rmse")
        best = local.iloc[0]
        report_lines.append(
            f"- **{target}:** lowest candidate RMSE is {best['rmse']:.5f} from `{best['candidate']}`; "
            f"calibration slope is {best['calibration_slope_observed_on_predicted']:.3f}, "
            f"with {int(best['optimism_gt_0p05_count'])} optimism errors above 0.05 BPB."
        )
    report_lines.extend(
        [
            "",
            "The candidate form and hyperparameters remain unchanged regardless of this result. Any further mechanism must be proposed in a new preregistered batch; coefficient retuning against these rows is disallowed.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
