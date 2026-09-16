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
"""Screen preregistered multi-rate component relaxation on paired fit panels."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
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
    component_relaxation_models as relaxation,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    potential_phase_models as potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_potential_phase_round2 as potential_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round3_component_relaxation"
)
RATE_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FoldPotential:
    fold: int
    train: np.ndarray
    test: np.ndarray
    model: potential.ConvexPotential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def geometry(panel: Any) -> potential.PotentialGeometry:
    return potential_screen.geometry(panel)


def fold_potentials(panel: Any, config: potential.PotentialConfig) -> list[FoldPotential]:
    result = []
    for fold, (train, test) in enumerate(paired_screen.folds(panel)):
        paired_train = train[panel.paired_mask[train]]
        model = potential.fit_potential(
            geometry(panel),
            panel.aggregate_weights,
            panel.one_phase_target,
            paired_train,
            config,
        )
        result.append(FoldPotential(fold, train, test, model))
    return result


def rate_configs(family_count: int) -> list[relaxation.ComponentRelaxationConfig]:
    return [
        relaxation.ComponentRelaxationConfig(tuple(rates)) for rates in itertools.product(RATE_GRID, repeat=family_count)
    ]


def oof_prediction(
    panel: Any,
    folds: list[FoldPotential],
    config: relaxation.ComponentRelaxationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.full(panel.n, np.nan, dtype=float)
    tied = np.full(panel.n, np.nan, dtype=float)
    fold_index = np.full(panel.n, -1, dtype=int)
    for fold in folds:
        model = relaxation.fit_component_relaxation(fold.model, panel.alpha0, config)
        combined[fold.test] = model.predict(panel.weights[fold.test])
        tied[fold.test] = fold.model.predict(panel.aggregate_weights[fold.test])
        fold_index[fold.test] = fold.fold
    if not np.isfinite(combined).all() or not np.isfinite(tied).all() or np.any(fold_index < 0):
        raise RuntimeError(f"Incomplete component-relaxation OOF prediction for {panel.name}")
    return combined, combined - tied, fold_index


def select_rates(
    panel: Any,
    potential_config: potential.PotentialConfig,
) -> tuple[
    relaxation.ComponentRelaxationConfig,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    folds = fold_potentials(panel, potential_config)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    observed_delta = panel.two_phase_target[panel.paired_mask] - panel.one_phase_target[panel.paired_mask]
    for config in rate_configs(len(panel.family_names)):
        combined, delta, fold_index = oof_prediction(panel, folds, config)
        predictions[config.key] = (combined, delta, fold_index)
        rows.append(
            {
                "panel": panel.name,
                "config": config.key,
                **{
                    f"combined_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, combined).items()
                },
                **{
                    f"delta_{key}": value
                    for key, value in paired_screen.scalar_metrics(observed_delta, delta[panel.paired_mask]).items()
                },
            }
        )
    table = pd.DataFrame(rows).sort_values(["delta_rmse", "combined_rmse", "delta_regret_at_1"])
    best = table.iloc[0]
    rates = tuple(float(value) for value in str(best["config"]).removeprefix("rates=").split(","))
    selected = relaxation.ComponentRelaxationConfig(rates)
    combined, delta, fold_index = predictions[selected.key]

    fold_rows = []
    for fold in range(len(folds)):
        local_rows = []
        test = fold_index == fold
        paired_test = test & panel.paired_mask
        for config in rate_configs(len(panel.family_names)):
            candidate_combined, candidate_delta, _ = predictions[config.key]
            local_rows.append(
                {
                    "fold": fold,
                    "config": config.key,
                    "delta_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    candidate_delta[paired_test]
                                    - (panel.two_phase_target[paired_test] - panel.one_phase_target[paired_test])
                                )
                                ** 2
                            )
                        )
                    ),
                    "combined_rmse": float(
                        np.sqrt(np.mean((candidate_combined[test] - panel.two_phase_target[test]) ** 2))
                    ),
                }
            )
        fold_rows.append(pd.DataFrame(local_rows).sort_values(["delta_rmse", "combined_rmse"]).iloc[0].to_dict())
    return selected, combined, delta, fold_index, table, pd.DataFrame(fold_rows)


def render_predictions(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("300M Uncheatable", "300M Table-9", "Delphi Uncheatable", "Delphi Table-9"),
    )
    panels = ("300m_uncheatable", "300m_table9", "delphi_3e18_uncheatable", "delphi_3e18_table9")
    for index, panel in enumerate(panels):
        row, column = index // 2 + 1, index % 2 + 1
        local = frame.loc[frame["panel"].eq(panel)]
        minimum = float(min(local["observed"].min(), local["predicted"].min()))
        maximum = float(max(local["observed"].max(), local["predicted"].max()))
        figure.add_trace(
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
        figure.add_trace(
            go.Scatter(
                x=local["predicted"],
                y=local["observed"],
                mode="markers",
                marker={"size": 7, "color": "#2f6f8f", "opacity": 0.72},
                name="MCR OOF",
                showlegend=index == 0,
                hovertemplate="pred=%{x:.5f}<br>obs=%{y:.5f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.update_xaxes(title_text="OOF predicted BPB", row=row, col=column)
        if column == 1:
            figure.update_yaxes(title_text="Observed BPB", row=row, col=column)
    figure.update_layout(
        title="Multi-rate component relaxation: paired fit-panel OOF",
        template="plotly_white",
        width=1450,
        height=980,
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    prediction_rows = []
    grid_tables = []
    fold_rows = []
    for scale in ("300m", "delphi_3e18"):
        for target in ("uncheatable", "table9"):
            panel = paired_screen.load_panel(scale, target)
            potential_config, _potential_prediction, _potential_grid = potential_screen.select_potential(panel)
            selected, combined, delta, fold_index, table, fold_selection = select_rates(panel, potential_config)
            grid_tables.append(table)
            selected_rows.append(
                {
                    "panel": panel.name,
                    "family_names": ",".join(panel.family_names),
                    "selected_potential": potential_config.key,
                    "selected_rates": selected.key,
                    **{
                        f"combined_{key}": value
                        for key, value in paired_screen.scalar_metrics(panel.two_phase_target, combined).items()
                    },
                    **{
                        f"delta_{key}": value
                        for key, value in paired_screen.scalar_metrics(
                            panel.two_phase_target[panel.paired_mask] - panel.one_phase_target[panel.paired_mask],
                            delta[panel.paired_mask],
                        ).items()
                    },
                }
            )
            for index in range(panel.n):
                prediction_rows.append(
                    {
                        "panel": panel.name,
                        "row_index": index,
                        "fold": int(fold_index[index]),
                        "paired": bool(panel.paired_mask[index]),
                        "observed": panel.two_phase_target[index],
                        "predicted": combined[index],
                        "predicted_phase_delta": delta[index],
                    }
                )
            fold_selection.insert(0, "panel", panel.name)
            fold_rows.append(fold_selection)

    selected = pd.DataFrame(selected_rows)
    predictions = pd.DataFrame(prediction_rows)
    grid = pd.concat(grid_tables, ignore_index=True)
    fold_selection = pd.concat(fold_rows, ignore_index=True)
    selected.to_csv(args.output_dir / "selected_configs_and_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "selected_oof_predictions.csv", index=False)
    grid.to_csv(args.output_dir / "rate_grid.csv", index=False)
    fold_selection.to_csv(args.output_dir / "fold_rate_selection.csv", index=False)
    render_predictions(predictions, args.output_dir / "selected_oof_predictions.html")
    report = [
        "# Multi-rate component relaxation: fit-panel screen",
        "",
        "The tied component amplitudes and nonlinear response were selected using only one-phase grouped OOF. Family relaxation rates were then selected using paired phase-delta OOF. No historical or adversarial outcome was read.",
        "",
        selected.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Foldwise best rate configurations",
        "",
        fold_selection.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(selected.to_string(index=False))
    print("\nFoldwise selections")
    print(fold_selection.to_string(index=False))


if __name__ == "__main__":
    main()
