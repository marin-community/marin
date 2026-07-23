# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Diagnose the minimal aggregate/contrast basis required by StarCoder."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_shared_private_round25 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round28_starcoder_invariant_basis"
DEGREES = (1, 2, 3, 4, 5)
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def exponents(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (a_power, d_power)
        for total in range(1, degree + 1)
        for a_power in range(total + 1)
        for d_power in [total - a_power]
    )


def invariants(panel: paired.PairedPanel, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    policies = panel.weights if weights is None else weights
    rare0 = policies[:, 0, 1]
    rare1 = policies[:, 1, 1]
    aggregate = panel.alpha0 * rare0 + (1.0 - panel.alpha0) * rare1
    contrast = rare1 - rare0
    return aggregate, contrast


def design(
    panel: paired.PairedPanel, degree: int, weights: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    aggregate, contrast = invariants(panel, weights)
    powers = exponents(degree)
    matrix = np.column_stack([aggregate**a_power * contrast**d_power for a_power, d_power in powers])
    names = tuple(f"a^{a_power} d^{d_power}" for a_power, d_power in powers)
    return matrix, names


def fit_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(matrix[train])
    scaled = scaler.transform(matrix)
    model = Ridge(alpha=ridge).fit(scaled[train], target[train])
    natural = model.coef_ / np.maximum(scaler.scale_, 1e-12)
    return model.predict(scaled[test]), natural, scaler.mean_


def global_oof(panel: paired.PairedPanel, degree: int, ridge: float) -> np.ndarray:
    matrix, _ = design(panel, degree)
    prediction = np.full(panel.n, np.nan)
    for train, test in starcoder.surface_folds(panel):
        prediction[test] = fit_predict(matrix, panel.two_phase_target, train, test, ridge)[0]
    return prediction


def inner_score(panel: paired.PairedPanel, indices: np.ndarray, degree: int, ridge: float, offset: int) -> float:
    matrix, _ = design(panel, degree)
    prediction = np.full(panel.n, np.nan)
    for train, test in audit.inner_folds(panel, indices, offset):
        prediction[test] = fit_predict(matrix, panel.two_phase_target, train, test, ridge)[0]
    return float(np.sqrt(np.mean((prediction[indices] - panel.two_phase_target[indices]) ** 2)))


def nested(panel: paired.PairedPanel) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan)
    rows = []
    for fold, (train, test) in enumerate(starcoder.surface_folds(panel)):
        scores = [
            (inner_score(panel, train, degree, ridge, fold), degree, ridge)
            for degree, ridge in product(DEGREES, RIDGE_GRID)
        ]
        score, degree, ridge = min(scores)
        matrix, _ = design(panel, degree)
        prediction[test] = fit_predict(matrix, panel.two_phase_target, train, test, ridge)[0]
        rows.append({"surface": panel.name, "fold": fold, "inner_rmse": score, "degree": degree, "ridge": ridge})
    return prediction, pd.DataFrame(rows)


def raw_optimum(panel: paired.PairedPanel, degree: int, ridge: float) -> dict[str, Any]:
    fit_matrix, _ = design(panel, degree)
    grid = np.linspace(0.0, 1.0, 401)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    grid_matrix, _ = design(panel, degree, weights)
    prediction, _, _ = fit_predict(
        np.vstack([fit_matrix, grid_matrix]),
        np.concatenate([panel.two_phase_target, np.zeros(len(grid_matrix))]),
        np.arange(panel.n),
        np.arange(panel.n, panel.n + len(grid_matrix)),
        ridge,
    )
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "degree": degree,
        "ridge": ridge,
        "predicted_p0": float(p0.ravel()[best]),
        "predicted_p1": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_p0": float(panel.weights[observed, 0, 1]),
        "observed_p1": float(panel.weights[observed, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed]),
        "optimum_distance": float(
            np.hypot(p0.ravel()[best] - panel.weights[observed, 0, 1], p1.ravel()[best] - panel.weights[observed, 1, 1])
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    path_rows = []
    coefficient_rows = []
    nested_rows = []
    optimum_rows = []
    for panel in panels:
        for degree, ridge in product(DEGREES, RIDGE_GRID):
            prediction = global_oof(panel, degree, ridge)
            path_rows.append(
                {
                    "surface": panel.name,
                    "degree": degree,
                    "ridge": ridge,
                    **metrics.scalar_metrics(panel.two_phase_target, prediction),
                }
            )
        surface_path = pd.DataFrame(path_rows)
        selected = surface_path[surface_path["surface"] == panel.name].sort_values("rmse").iloc[0]
        degree = int(selected["degree"])
        ridge = float(selected["ridge"])
        matrix, names = design(panel, degree)
        _, coefficients, _ = fit_predict(matrix, panel.two_phase_target, np.arange(panel.n), np.arange(panel.n), ridge)
        coefficient_rows.extend(
            {"surface": panel.name, "degree": degree, "ridge": ridge, "term": name, "coefficient": float(value)}
            for name, value in zip(names, coefficients, strict=True)
        )
        nested_prediction, selections = nested(panel)
        nested_rows.append(
            {
                "surface": panel.name,
                **{
                    f"nested_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, nested_prediction).items()
                },
                "modal_degree": int(selections["degree"].mode().iloc[0]),
                "modal_ridge": float(selections["ridge"].mode().iloc[0]),
            }
        )
        selections.to_csv(args.output_dir / f"{panel.name}__nested_selections.csv", index=False)
        optimum_rows.append(raw_optimum(panel, degree, ridge))

    paths = pd.DataFrame(path_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    nested_table = pd.DataFrame(nested_rows)
    optima = pd.DataFrame(optimum_rows)
    paths.to_csv(args.output_dir / "degree_ridge_path.csv", index=False)
    coefficients.to_csv(args.output_dir / "selected_coefficients.csv", index=False)
    nested_table.to_csv(args.output_dir / "nested_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    report = [
        "# Round 28 diagnostic: StarCoder invariant basis",
        "",
        "This is a descriptive identification diagnostic, not an admissible candidate. It was not evaluated on Delphi historical or adversarial outcomes.",
        "",
        "## Nested metrics",
        "",
        nested_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima of globally selected descriptive fits",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Globally selected coefficients",
        "",
        coefficients.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(nested_table.to_string(index=False))
    print("\nRaw optima")
    print(optima.to_string(index=False))
    print("\nSelected coefficients")
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
