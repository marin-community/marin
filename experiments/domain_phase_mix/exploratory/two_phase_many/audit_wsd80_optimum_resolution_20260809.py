# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Audit whether the WSD80 optimum-distance gate is resolution limited.

The promotion memo perturbed every panel coordinate independently by one pooled
seed standard deviation. The available repeats use the same five data seeds at
eleven coordinates, allowing that variance to be separated into a common seed
shift and coordinate-specific ranking noise. Only the latter can move an
argmin within one same-seed surface.

Neither bootstrap below estimates an oracle model's probability of passing the
frozen gate. They are explicitly labelled sensitivity analyses around the
single observed surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_gated_absorption_wsd80_20260807 as model

DEFAULT_REPEAT_PATH = (
    Path(__file__).resolve().parent
    / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_measured_fiber_observations.csv"
)
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 20260809
GATE_DISTANCE = 0.05
COORDINATE_COLUMNS = ["phase_0_starcoder", "phase_1_starcoder"]


def repeated_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return complete repeated coordinates by seed."""
    counts = frame.groupby(COORDINATE_COLUMNS)["wsd80_bpb"].size()
    repeated = counts[counts > 1].index
    rows = frame.set_index(COORDINATE_COLUMNS).loc[repeated].reset_index()
    matrix = rows.pivot(index=COORDINATE_COLUMNS, columns="data_seed", values="wsd80_bpb")
    assert not matrix.isna().any().any()
    assert matrix.shape == (11, 5)
    return matrix, rows.to_numpy()


def variance_decomposition(matrix: pd.DataFrame) -> dict[str, object]:
    """Separate common seed shifts from coordinate-specific residuals."""
    values = matrix.to_numpy()
    residuals = values - values.mean(axis=1, keepdims=True)
    seed_effects = residuals.mean(axis=0)
    idiosyncratic = residuals - seed_effects[None, :]
    correlations = np.corrcoef(residuals)
    upper = correlations[np.triu_indices_from(correlations, k=1)]
    coordinate_sd = values.std(axis=1, ddof=1)
    residual_degrees_of_freedom = (values.shape[0] - 1) * (values.shape[1] - 1)
    return {
        "pooled_coordinate_sd": float(np.sqrt(np.mean(coordinate_sd**2))),
        "coordinate_sd_min": float(coordinate_sd.min()),
        "coordinate_sd_max": float(coordinate_sd.max()),
        "common_seed_effect_sd": float(seed_effects.std(ddof=1)),
        "idiosyncratic_sd": float(np.sqrt(np.sum(idiosyncratic**2) / residual_degrees_of_freedom)),
        "mean_pairwise_coordinate_residual_correlation": float(upper.mean()),
        "seed_effects": seed_effects.tolist(),
    }


def argmin_sensitivity(sigma: float, draws: int, seed: int) -> dict[str, float]:
    """Perturb the observed panel independently and reselect its raw argmin."""
    response = model.TARGETS.values[:, model.TARGETS.names.index(model.harness.PRIMARY_TARGET)]
    rows = np.flatnonzero(model.INTERIOR)
    observed = rows[np.argmin(response[rows])]
    observed_coordinate = np.array([model.PANEL.phase_0[observed, 1], model.PANEL.phase_1[observed, 1]])
    coordinates = np.column_stack([model.PANEL.phase_0[rows, 1], model.PANEL.phase_1[rows, 1]])
    rng = np.random.default_rng(seed)
    selected = np.empty(draws, dtype=int)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        perturbed = response[rows, None] + rng.normal(0.0, sigma, (len(rows), stop - start))
        selected[start:stop] = np.argmin(perturbed, axis=0)
    distance = np.linalg.norm(coordinates[selected] - observed_coordinate[None, :], axis=1)
    return {
        "sigma": sigma,
        "fraction_within_gate_strict_floating_point": float(np.mean(distance <= GATE_DISTANCE)),
        "fraction_within_gate_with_tolerance": float(np.mean(distance <= GATE_DISTANCE + 1e-12)),
        "median_distance": float(np.median(distance)),
        "p90_distance": float(np.quantile(distance, 0.9)),
    }


def repeated_optimum_summary(matrix: pd.DataFrame) -> dict[str, object]:
    """Summarize the replicated observed optimum and tied comparator."""
    optimum = matrix.loc[(0.1, 0.5)].to_numpy()
    tied = matrix.loc[(0.3, 0.3)].to_numpy()
    gain = tied - optimum
    critical = float(t.ppf(0.975, len(gain) - 1))
    half_width = critical * float(gain.std(ddof=1)) / np.sqrt(len(gain))
    winners = matrix.index[np.argmin(matrix.to_numpy(), axis=0)].tolist()
    return {
        "optimum_mean": float(optimum.mean()),
        "optimum_sd": float(optimum.std(ddof=1)),
        "reference_seed_optimum": float(optimum[0]),
        "reference_minus_repeat_mean": float(optimum[0] - optimum.mean()),
        "paired_gain_mean": float(gain.mean()),
        "paired_gain_sd": float(gain.std(ddof=1)),
        "paired_gain_95_ci": [float(gain.mean() - half_width), float(gain.mean() + half_width)],
        "gain_positive_seed_count": int(np.sum(gain > 0)),
        "winner_among_repeated_coordinates_by_seed": [list(map(float, coordinate)) for coordinate in winners],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-path", type=Path, default=DEFAULT_REPEAT_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.repeat_path)
    matrix, _ = repeated_matrix(frame)
    decomposition = variance_decomposition(matrix)
    result = {
        "repeat_coordinates": int(matrix.shape[0]),
        "seeds_per_repeated_coordinate": int(matrix.shape[1]),
        "variance_decomposition": decomposition,
        "replicated_optimum": repeated_optimum_summary(matrix),
        "iid_pooled_sd_sensitivity": argmin_sensitivity(
            float(decomposition["pooled_coordinate_sd"]), args.draws, args.seed
        ),
        "iid_coordinate_specific_sd_sensitivity": argmin_sensitivity(
            float(decomposition["idiosyncratic_sd"]), args.draws, args.seed
        ),
        "interpretation": (
            "These are plug-in perturbation sensitivities around a noisy observed surface, not the probability that an "
            "oracle expected-surface predictor passes the frozen distance gate."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
