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
"""Screen convex-potential phase laws using only paired and fit-panel outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    potential_phase_models as potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round2_potential_phase"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COLORS = {
    potential.PhaseLaw.WORK_DISSIPATION.value: "#2f6f8f",
    potential.PhaseLaw.RELAXATION.value: "#d65f35",
    "tied_potential": "#5c6770",
}


@dataclass(frozen=True)
class OOFPrediction:
    aggregate: np.ndarray
    delta: np.ndarray
    combined: np.ndarray
    coefficient_rows: tuple[dict[str, Any], ...]
    minimum_bregman: float
    maximum_tied_correction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def geometry(panel: Any) -> potential.PotentialGeometry:
    return potential.PotentialGeometry(
        domain_names=panel.domain_names,
        family_names=panel.family_names,
        family_members=panel.family_members,
        proportional_weights=panel.proportional_weights,
        total_epoch_coefficients=panel.c0 + panel.c1,
    )


def potential_configs() -> list[potential.PotentialConfig]:
    configs = [
        potential.PotentialConfig(response, curvature, offset, l2)
        for response in (potential.DebtResponse.INVERSE_POWER, potential.DebtResponse.LOGARITHMIC)
        for curvature in (
            (0.1, 0.2, 0.3, 0.5, 0.75, 1.0) if response is potential.DebtResponse.INVERSE_POWER else (1.0,)
        )
        for offset in (0.01, 0.03, 0.1, 0.3, 1.0)
        for l2 in (0.001, 0.01, 0.1, 1.0, 10.0)
    ]
    return configs


def phase_configs(law: potential.PhaseLaw) -> list[potential.PhaseConfig]:
    if law is potential.PhaseLaw.WORK_DISSIPATION:
        return [potential.WorkDissipationConfig(l2) for l2 in (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)]
    if law is potential.PhaseLaw.RELAXATION:
        return [
            potential.RelaxationConfig(rate, l2)
            for rate in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
            for l2 in (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
        ]
    raise ValueError(f"No phase grid for {law}")


def potential_oof(panel: Any, config: potential.PotentialConfig) -> np.ndarray:
    prediction = np.full(panel.n, np.nan, dtype=float)
    aggregate = panel.aggregate_weights
    geom = geometry(panel)
    for train, test in paired_screen.folds(panel):
        train = train[panel.paired_mask[train]]
        test = test[panel.paired_mask[test]]
        if len(test) == 0:
            continue
        model = potential.fit_potential(geom, aggregate, panel.one_phase_target, train, config)
        prediction[test] = model.predict(aggregate[test])
    if not np.isfinite(prediction[panel.paired_mask]).all():
        raise RuntimeError(f"Incomplete potential OOF prediction for {panel.name}")
    return prediction


def select_potential(panel: Any) -> tuple[potential.PotentialConfig, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    observed = panel.one_phase_target[panel.paired_mask]
    for config in potential_configs():
        prediction = potential_oof(panel, config)
        predictions[config.key] = prediction
        rows.append(
            {
                "panel": panel.name,
                "config": config.key,
                "config_json": json.dumps({**asdict(config), "response": config.response.value}, sort_keys=True),
                **paired_screen.scalar_metrics(observed, prediction[panel.paired_mask]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    selected = table.iloc[0]
    values = json.loads(selected["config_json"])
    values["response"] = potential.DebtResponse(values["response"])
    config = potential.PotentialConfig(**values)
    return config, predictions[config.key], table


def phase_oof(
    panel: Any,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
    phase_config: potential.PhaseConfig,
) -> OOFPrediction:
    aggregate_prediction = np.full(panel.n, np.nan, dtype=float)
    delta_prediction = np.full(panel.n, np.nan, dtype=float)
    combined_prediction = np.full(panel.n, np.nan, dtype=float)
    coefficient_rows: list[dict[str, Any]] = []
    minimum_bregman = float("inf")
    maximum_tied_correction = 0.0
    aggregate = panel.aggregate_weights
    phase_delta = panel.two_phase_target - panel.one_phase_target
    geom = geometry(panel)
    for fold, (train, test) in enumerate(paired_screen.folds(panel)):
        paired_train = train[panel.paired_mask[train]]
        model = potential.fit_potential(geom, aggregate, panel.one_phase_target, paired_train, potential_config)
        phase_model = potential.fit_phase_potential(
            model,
            panel.weights,
            phase_delta,
            paired_train,
            panel.alpha0,
            law,
            phase_config,
        )
        aggregate_prediction[test] = model.predict(aggregate[test])
        delta_prediction[test] = phase_model.predict_delta(panel.weights[test])
        combined_prediction[test] = aggregate_prediction[test] + delta_prediction[test]
        for feature, coefficient in zip(
            phase_model.head.feature_names,
            phase_model.head.coefficients_in_natural_units,
            strict=True,
        ):
            coefficient_rows.append({"fold": fold, "feature": feature, "coefficient": coefficient})
        if law is potential.PhaseLaw.WORK_DISSIPATION:
            aggregate_test, endpoint_test, _displacement = potential.transported_endpoint(
                panel.weights[test], panel.alpha0
            )
            minimum_bregman = min(minimum_bregman, float(np.min(model.bregman(endpoint_test, aggregate_test))))
        tied = np.stack([aggregate[test], aggregate[test]], axis=1)
        maximum_tied_correction = max(maximum_tied_correction, float(np.max(np.abs(phase_model.predict_delta(tied)))))
    if not np.isfinite(combined_prediction).all():
        raise RuntimeError(f"Incomplete phase OOF prediction for {panel.name} {law.value}")
    if law is not potential.PhaseLaw.WORK_DISSIPATION:
        minimum_bregman = np.nan
    return OOFPrediction(
        aggregate_prediction,
        delta_prediction,
        combined_prediction,
        tuple(coefficient_rows),
        minimum_bregman,
        maximum_tied_correction,
    )


def select_phase(
    panel: Any,
    potential_config: potential.PotentialConfig,
    law: potential.PhaseLaw,
) -> tuple[potential.PhaseConfig, OOFPrediction, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, OOFPrediction] = {}
    paired = panel.paired_mask
    observed_delta = panel.two_phase_target[paired] - panel.one_phase_target[paired]
    for config in phase_configs(law):
        prediction = phase_oof(panel, potential_config, law, config)
        predictions[config.key] = prediction
        rows.append(
            {
                "panel": panel.name,
                "law": law.value,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "minimum_bregman": prediction.minimum_bregman,
                "maximum_tied_correction": prediction.maximum_tied_correction,
                **{
                    f"delta_{key}": value
                    for key, value in paired_screen.scalar_metrics(observed_delta, prediction.delta[paired]).items()
                },
                **{
                    f"combined_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, prediction.combined).items()
                },
            }
        )
    table = pd.DataFrame(rows).sort_values(["delta_rmse", "combined_rmse", "delta_worst_optimism"])
    selected = table.iloc[0]
    values = json.loads(selected["config_json"])
    config_class = {
        potential.PhaseLaw.WORK_DISSIPATION: potential.WorkDissipationConfig,
        potential.PhaseLaw.RELAXATION: potential.RelaxationConfig,
    }[law]
    config = config_class(**values)
    return config, predictions[config.key], table


def coefficient_stability(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    result = []
    for feature, local in frame.groupby("feature", sort=False):
        values = local["coefficient"].to_numpy(float)
        result.append(
            {
                "feature": feature,
                "fold_mean": float(np.mean(values)),
                "fold_standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "sign_consistency": float(max(np.mean(values >= 0.0), np.mean(values <= 0.0))),
            }
        )
    return result


def render_predictions(frame: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("300M Uncheatable", "300M Table-9", "Delphi Uncheatable", "Delphi Table-9")
    )
    panels = ("300m_uncheatable", "300m_table9", "delphi_3e18_uncheatable", "delphi_3e18_table9")
    for index, panel in enumerate(panels):
        row, column = index // 2 + 1, index % 2 + 1
        local = frame.loc[frame["panel"].eq(panel)]
        minimum = float(min(local["observed"].min(), local["predicted"].min()))
        maximum = float(max(local["observed"].max(), local["predicted"].max()))
        fig.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                line={"dash": "dash", "color": "#8d989f"},
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        for law in potential.PhaseLaw:
            subset = local.loc[local["law"].eq(law.value)]
            fig.add_trace(
                go.Scatter(
                    x=subset["predicted"],
                    y=subset["observed"],
                    mode="markers",
                    name=law.value.replace("_", " "),
                    legendgroup=law.value,
                    marker={"color": COLORS[law.value], "size": 7, "opacity": 0.72},
                    showlegend=index == 0,
                    hovertemplate="pred=%{x:.5f}<br>obs=%{y:.5f}<extra></extra>",
                ),
                row=row,
                col=column,
            )
        fig.update_xaxes(title_text="OOF predicted BPB", row=row, col=column)
        if column == 1:
            fig.update_yaxes(title_text="Observed BPB", row=row, col=column)
    fig.update_layout(
        title="Round-two convex-potential phase laws: grouped OOF",
        template="plotly_white",
        width=1450,
        height=980,
        legend={"orientation": "h", "y": 1.06},
    )
    fig.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    potential_grid: list[pd.DataFrame] = []
    phase_grid: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for scale in ("300m", "delphi_3e18"):
        for target in ("uncheatable", "table9"):
            panel = paired_screen.load_panel(scale, target)
            selected_potential, potential_prediction, potential_table = select_potential(panel)
            potential_grid.append(potential_table)
            selected_rows.append(
                {
                    "panel": panel.name,
                    "component": "tied_potential",
                    "selected_config": selected_potential.key,
                    **{
                        f"one_phase_{key}": value
                        for key, value in paired_screen.scalar_metrics(
                            panel.one_phase_target[panel.paired_mask], potential_prediction[panel.paired_mask]
                        ).items()
                    },
                }
            )
            for law in potential.PhaseLaw:
                selected_phase, oof, phase_table = select_phase(panel, selected_potential, law)
                phase_grid.append(phase_table)
                selected_rows.append(
                    {
                        "panel": panel.name,
                        "component": law.value,
                        "selected_potential_config": selected_potential.key,
                        "selected_config": selected_phase.key,
                        "minimum_bregman": oof.minimum_bregman,
                        "maximum_tied_correction": oof.maximum_tied_correction,
                        **{
                            f"combined_{key}": value
                            for key, value in paired_screen.scalar_metrics(panel.two_phase_target, oof.combined).items()
                        },
                        **{
                            f"delta_{key}": value
                            for key, value in paired_screen.scalar_metrics(
                                panel.two_phase_target[panel.paired_mask] - panel.one_phase_target[panel.paired_mask],
                                oof.delta[panel.paired_mask],
                            ).items()
                        },
                    }
                )
                for index, (observed, predicted, aggregate_prediction, delta_prediction) in enumerate(
                    zip(panel.two_phase_target, oof.combined, oof.aggregate, oof.delta, strict=True)
                ):
                    prediction_rows.append(
                        {
                            "panel": panel.name,
                            "law": law.value,
                            "row_index": index,
                            "paired": bool(panel.paired_mask[index]),
                            "observed": observed,
                            "predicted": predicted,
                            "aggregate_prediction": aggregate_prediction,
                            "delta_prediction": delta_prediction,
                        }
                    )
                for record in coefficient_stability(oof.coefficient_rows):
                    coefficient_rows.append({"panel": panel.name, "law": law.value, **record})

    potential_table = pd.concat(potential_grid, ignore_index=True)
    phase_table = pd.concat(phase_grid, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    potential_table.to_csv(args.output_dir / "potential_hyperparameter_grid.csv", index=False)
    phase_table.to_csv(args.output_dir / "phase_hyperparameter_grid.csv", index=False)
    selected.to_csv(args.output_dir / "selected_configs_and_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "selected_oof_predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "phase_coefficient_stability.csv", index=False)
    render_predictions(predictions, args.output_dir / "selected_oof_predictions.html")
    report = [
        "# Round-two convex-potential phase screen",
        "",
        "Hyperparameters were selected only on grouped OOF predictions from the one-phase and paired fit panels. Historical and adversarial outcomes were not read.",
        "",
        selected.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold coefficient stability",
        "",
        coefficients.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
