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
"""Audit phase-order identifiability from exposed aggregate-matched panels.

This script does not fit a deployment surrogate. It estimates which parts of
the phase-order response are identified by the exposed random-fiber and
aggressive antithetic panels:

* the direction-averaged response as asymmetry grows;
* the RMS odd response and its cross-radius stability;
* the best observed sign along sampled directions;
* the Fisher-information envelope for a local 38-dimensional phase gradient;
* target-matched outcomes across every exposed aggressive design family.

The targeted pairwise phase-order panel is sealed and is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import null_space
from scipy.stats import binomtest, chi2, norm

NoiseSource = dict[str, float | int]
NoiseSources = dict[str, dict[str, NoiseSource]]

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_HELDOUTS = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
DEFAULT_ANTITHETIC_PAIRS = (
    REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723" / "balanced_antithetic_pairs.csv"
)
DEFAULT_AGGRESSIVE_RESULTS = (
    REFERENCE_OUTPUTS
    / "delphi_3e18_aggressive_phase_asymmetry_results_20260723"
    / "observed_results_with_control_deltas.csv"
)
DEFAULT_RESOLUTION = REFERENCE_OUTPUTS / "delphi_3e18_broad_mixture_tail_audit_20260724" / "statistical_resolution.json"
DEFAULT_AGGRESSIVE_DESIGN_SUMMARY = (
    REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_20260722" / "summary.json"
)
DEFAULT_LOW_EPSILON_PATHS = (
    REFERENCE_OUTPUTS
    / "decoupled_phase_information_low_epsilon_validation_results_20260712"
    / "combined_uncheatable_paths.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "phase_order_identifiability_envelope_20260724"

RANDOM_SERIES = "delphi_3e18_frontier_random_phase_population_20260720"
TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
TARGET_ANCHORS = {
    "uncheatable": "uncheatable_frontier",
    "table9": "table9_frontier",
}
RADIUS_VALUES = np.array([0.25, 0.50, 0.75], dtype=float)
TV_VALUES = np.array([0.10, 0.25, 0.50], dtype=float)
PHASE_TANGENT_DIMENSION = 38
GOAL_GAIN_BPB = 0.01
TOTAL_FIT_BUDGET = 280
PHASE_RUN_BUDGETS = (80, 96, 120, 140, 160, 200, 280)
BEST_SIGN_NULL_FRACTION = 2 / 3
POWER_ALPHA = 0.05
POWER_TARGET = 0.80
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260724
POWER_GRID = np.linspace(0.25, 4.0, 751)
LOW_EPSILON_MAX = 0.01
UPSTREAM_NOISE_SOURCE_IDS = ("archive_median", "proportional", "matched_frontier")
PANEL_NOISE_SOURCE_ID = "matched_frontier_panel16"
ARCHIVE_FRONTIER_NOISE_SOURCE_ID = "matched_frontier_archive_n4"
NOISE_SOURCE_IDS = (
    "archive_median",
    "proportional",
    ARCHIVE_FRONTIER_NOISE_SOURCE_ID,
    PANEL_NOISE_SOURCE_ID,
)
AGGREGATE_ROLLUP_COLUMNS = {
    "phase_0_broad_share",
    "phase_0_dolmino_share",
}
SEED_TAG_RANDOM_RADIUS = 101
SEED_TAG_GEOMETRY = 307
SEED_TAG_NULL_SELECTION = 401
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    parser.add_argument("--antithetic-pairs", type=Path, default=DEFAULT_ANTITHETIC_PAIRS)
    parser.add_argument("--aggressive-results", type=Path, default=DEFAULT_AGGRESSIVE_RESULTS)
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--aggressive-design-summary",
        type=Path,
        default=DEFAULT_AGGRESSIVE_DESIGN_SUMMARY,
    )
    parser.add_argument("--low-epsilon-paths", type=Path, default=DEFAULT_LOW_EPSILON_PATHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def assert_sealed_panel_absent(frame: pd.DataFrame, label: str) -> None:
    assert not any("targeted_pairwise" in str(column).lower() for column in frame.columns), label
    string_columns = frame.select_dtypes(include=["object", "string"])
    if string_columns.empty:
        return
    sealed = string_columns.astype("string").apply(
        lambda column: column.str.contains("targeted_pairwise", case=False, na=False)
    )
    assert not sealed.any().any(), label


def metadata_column(frame: pd.DataFrame, key: str) -> pd.Series:
    metadata = frame["proposal_metadata_json"].map(json.loads)
    return metadata.map(lambda row: float(row[key]))


def low_epsilon_headroom(paths: pd.DataFrame) -> pd.DataFrame:
    """Separate fixed-aggregate gains from gains over the best tied aggregate."""
    required_columns = {
        "candidate",
        "objective",
        "anchor_tag",
        "family",
        "phase_information_budget",
        "phase_tv",
        "observed_target_bpb",
        "tied_observed_target_bpb",
    }
    assert required_columns.issubset(paths.columns), sorted(required_columns - set(paths.columns))
    assert set(paths["objective"]) == set(TARGET_COLUMNS), sorted(paths["objective"].unique())

    controls = paths.loc[paths["family"].eq("control")].copy()
    assert controls.groupby(["objective", "anchor_tag"]).size().eq(1).all()
    global_tied = controls.groupby("objective")["observed_target_bpb"].min()

    low_epsilon = paths.loc[paths["family"].ne("control") & paths["phase_information_budget"].le(LOW_EPSILON_MAX)].copy()
    records: list[dict[str, Any]] = []
    for (objective, anchor_tag, family), frame in low_epsilon.groupby(
        ["objective", "anchor_tag", "family"],
        sort=True,
    ):
        best = frame.loc[frame["observed_target_bpb"].idxmin()]
        tied_bpb = float(best["tied_observed_target_bpb"])
        best_bpb = float(best["observed_target_bpb"])
        best_tied_bpb = float(global_tied.loc[objective])
        records.append(
            {
                "objective": objective,
                "anchor_tag": anchor_tag,
                "family": family,
                "candidate_count": len(frame),
                "selected_candidate": best["candidate"],
                "selected_epsilon_phase": float(best["phase_information_budget"]),
                "selected_phase_tv": float(best["phase_tv"]),
                "selected_observed_bpb": best_bpb,
                "anchor_tied_bpb": tied_bpb,
                "best_tied_bpb": best_tied_bpb,
                "conditional_gain_vs_anchor_tied_bpb": tied_bpb - best_bpb,
                "global_gain_vs_best_tied_bpb": best_tied_bpb - best_bpb,
                "selected_at_low_epsilon_boundary": math.isclose(
                    float(best["phase_information_budget"]),
                    LOW_EPSILON_MAX,
                ),
                "single_training_seed": True,
            }
        )

    summary = pd.DataFrame.from_records(records)
    assert len(summary) == 6, len(summary)
    return summary


def random_population_deltas(heldouts: pd.DataFrame) -> pd.DataFrame:
    panel = heldouts.loc[heldouts["training_series"].eq(RANDOM_SERIES)].copy()
    assert len(panel) == 296, len(panel)
    assert panel[list(TARGET_COLUMNS.values())].notna().all().all()
    panel["phase_tv"] = metadata_column(panel, "phase_tv")
    panel["realized_radius"] = metadata_column(panel, "realized_radius")
    panel["feasible_radius"] = metadata_column(panel, "feasible_radius")
    assert np.allclose(
        panel["realized_radius"],
        panel["radius_fraction"] * panel["feasible_radius"],
        atol=1e-12,
    )

    controls = panel.loc[panel["radius_fraction"].eq(0)].copy()
    assert len(controls) == 8
    control_lookup = controls.set_index(["anchor_id", "seed_block"])
    treatments = panel.loc[panel["radius_fraction"].gt(0)].copy()
    assert len(treatments) == 288
    for target, column in TARGET_COLUMNS.items():
        treatments[f"{target}_control_bpb"] = [
            control_lookup.loc[(anchor, seed), column]
            for anchor, seed in zip(
                treatments["anchor_id"],
                treatments["seed_block"],
                strict=True,
            )
        ]
        treatments[f"{target}_delta_bpb"] = treatments[column] - treatments[f"{target}_control_bpb"]
    return treatments


def nested_direction_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap seed blocks and directions while preserving all three radii."""
    pivot = frame.pivot(
        index=["seed_block", "direction_id"],
        columns="radius_fraction",
        values=value_column,
    ).reindex(columns=RADIUS_VALUES)
    seed_blocks = np.sort(frame["seed_block"].unique())
    by_seed = [pivot.loc[seed].sort_index().to_numpy(dtype=float) for seed in seed_blocks]
    direction_counts = {len(values) for values in by_seed}
    assert len(direction_counts) == 1, direction_counts
    values = np.stack(by_seed)
    assert np.isfinite(values).all()
    seed_count, directions_per_seed, _ = values.shape
    sampled_seeds = rng.integers(0, seed_count, size=(draws, seed_count))
    sampled_directions = rng.integers(
        0,
        directions_per_seed,
        size=(draws, seed_count, directions_per_seed),
    )
    sampled = values[sampled_seeds[:, :, None], sampled_directions]
    return sampled.mean(axis=(1, 2))


def signed_power_fit(means: np.ndarray) -> tuple[float, float, float, float]:
    """Fit m(rho)=coefficient*rho**power by finite-grid least squares."""
    designs = RADIUS_VALUES[None, :] ** POWER_GRID[:, None]
    denominators = np.sum(designs * designs, axis=1)
    coefficients = designs @ means / denominators
    residual = designs * coefficients[:, None] - means[None, :]
    squared_error = np.sum(residual * residual, axis=1)
    selected = int(np.argmin(squared_error))
    minimum = float(squared_error[selected])
    profile_ratio = float(np.max(squared_error) / minimum) if minimum > 0 else math.inf
    return (
        float(coefficients[selected]),
        float(POWER_GRID[selected]),
        minimum,
        profile_ratio,
    )


def random_radius_analysis(
    treatments: pd.DataFrame,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(sorted(treatments["anchor_id"].unique())):
        anchor_frame = treatments.loc[treatments["anchor_id"].eq(anchor)]
        for target_index, target in enumerate(TARGET_COLUMNS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        BOOTSTRAP_SEED,
                        SEED_TAG_RANDOM_RADIUS,
                        anchor_index,
                        target_index,
                    ]
                )
            )
            value_column = f"{target}_delta_bpb"
            means = anchor_frame.groupby("radius_fraction")[value_column].mean().reindex(RADIUS_VALUES).to_numpy()
            bootstrap = nested_direction_bootstrap(
                anchor_frame,
                value_column,
                draws,
                rng,
            )
            for radius_index, radius in enumerate(RADIUS_VALUES):
                values = anchor_frame.loc[anchor_frame["radius_fraction"].eq(radius), value_column].to_numpy()
                ci_low, ci_high = np.quantile(bootstrap[:, radius_index], [0.025, 0.975])
                strata_rows.append(
                    {
                        "anchor_id": anchor,
                        "target": target,
                        "target_matched": anchor == TARGET_ANCHORS[target],
                        "radius_fraction": radius,
                        "n_directions": len(values),
                        "mean_delta_bpb": float(np.mean(values)),
                        "median_delta_bpb": float(np.median(values)),
                        "sd_delta_bpb": float(np.std(values, ddof=1)),
                        "ci95_low_bpb": float(ci_low),
                        "ci95_high_bpb": float(ci_high),
                        "fraction_better": float(np.mean(values < 0)),
                        "mean_phase_tv": float(
                            anchor_frame.loc[anchor_frame["radius_fraction"].eq(radius), "phase_tv"].mean()
                        ),
                        "max_phase_tv": float(
                            anchor_frame.loc[anchor_frame["radius_fraction"].eq(radius), "phase_tv"].max()
                        ),
                        "mean_full_boundary_phase_tv": float(
                            (anchor_frame.loc[anchor_frame["radius_fraction"].eq(radius), "phase_tv"] / radius).mean()
                        ),
                        "max_full_boundary_phase_tv": float(
                            (anchor_frame.loc[anchor_frame["radius_fraction"].eq(radius), "phase_tv"] / radius).max()
                        ),
                    }
                )
            coefficient, power, squared_error, profile_ratio = signed_power_fit(means)
            bootstrap_fits = np.array([signed_power_fit(row)[:2] for row in bootstrap])
            coefficient_ci = np.quantile(bootstrap_fits[:, 0], [0.025, 0.975])
            power_ci = np.quantile(bootstrap_fits[:, 1], [0.025, 0.975])
            power_rows.append(
                {
                    "anchor_id": anchor,
                    "target": target,
                    "target_matched": anchor == TARGET_ANCHORS[target],
                    "coefficient_at_radius_1": coefficient,
                    "power": power,
                    "squared_error": squared_error,
                    "profile_sse_max_over_min": profile_ratio,
                    "coefficient_ci95_low": float(coefficient_ci[0]),
                    "coefficient_ci95_high": float(coefficient_ci[1]),
                    "power_ci95_low": float(power_ci[0]),
                    "power_ci95_high": float(power_ci[1]),
                    "power_at_grid_boundary_fraction": float(
                        np.mean((bootstrap_fits[:, 1] == POWER_GRID[0]) | (bootstrap_fits[:, 1] == POWER_GRID[-1]))
                    ),
                    "power_identified": False,
                }
            )
    return pd.DataFrame(strata_rows), pd.DataFrame(power_rows)


def aggressive_analysis(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for target, anchor in TARGET_ANCHORS.items():
        frame = pairs.loc[pairs["anchor_id"].eq(anchor)].copy()
        odd_column = f"{target}_odd_effect"
        curvature_column = f"{target}_curvature"
        best_column = f"{target}_best_sign_delta"
        for tv in TV_VALUES:
            radius_frame = frame.loc[frame["target_phase_tv"].eq(tv)]
            odd = radius_frame[odd_column].to_numpy()
            curvature = radius_frame[curvature_column].to_numpy()
            best = radius_frame[best_column].to_numpy()
            mean_curvature = float(np.mean(curvature))
            row: dict[str, Any] = {
                "anchor_id": anchor,
                "target": target,
                "target_phase_tv": tv,
                "n_directions": len(radius_frame),
                "odd_rms_bpb": float(np.sqrt(np.mean(odd * odd))),
                "mean_curvature_bpb": mean_curvature,
                "median_curvature_bpb": float(np.median(curvature)),
                "mean_best_sign_delta_bpb": float(np.mean(best)),
                "best_observed_sign_delta_bpb": float(np.min(best)),
                "fraction_best_sign_better": float(np.mean(best < 0)),
                "fraction_best_sign_better_null": BEST_SIGN_NULL_FRACTION,
                "fraction_best_sign_better_null_pvalue": float(
                    binomtest(
                        int(np.sum(best < 0)),
                        len(best),
                        BEST_SIGN_NULL_FRACTION,
                    ).pvalue
                ),
                "count_gain_ge_0p005": int(np.sum(best <= -0.005)),
                "count_gain_ge_0p010": int(np.sum(best <= -0.010)),
                "paired_seed_correlation_identified": False,
            }
            summary_rows.append(row)

        indexed = {tv: frame.loc[frame["target_phase_tv"].eq(tv)].set_index("direction_id") for tv in TV_VALUES}
        for train_tv, test_tv in ((0.10, 0.25), (0.10, 0.50), (0.25, 0.50)):
            train = indexed[train_tv]
            test = indexed[test_tv]
            common = train.index.intersection(test.index)
            train_odd = train.loc[common, odd_column].to_numpy()
            test_odd = test.loc[common, odd_column].to_numpy()
            predicted_odd = train_odd * test_tv / train_tv
            zero_rmse = float(np.sqrt(np.mean(test_odd * test_odd)))
            transfer_rmse = float(np.sqrt(np.mean((predicted_odd - test_odd) ** 2)))
            selected_sign = np.where(
                train.loc[common, f"{target}_plus_delta"].to_numpy()
                < train.loc[common, f"{target}_minus_delta"].to_numpy(),
                "plus",
                "minus",
            )
            selected_delta = np.array(
                [
                    test.loc[direction, f"{target}_{sign}_delta"]
                    for direction, sign in zip(common, selected_sign, strict=True)
                ],
                dtype=float,
            )
            transfer_rows.append(
                {
                    "anchor_id": anchor,
                    "target": target,
                    "train_tv": train_tv,
                    "test_tv": test_tv,
                    "n_directions": len(common),
                    "odd_sign_agreement": float(np.mean(np.sign(train_odd) == np.sign(test_odd))),
                    "odd_pearson": float(np.corrcoef(train_odd, test_odd)[0, 1]),
                    "linear_odd_rmse": transfer_rmse,
                    "zero_odd_rmse": zero_rmse,
                    "rmse_ratio_vs_zero": transfer_rmse / zero_rmse,
                    "selected_sign_mean_delta_bpb": float(np.mean(selected_delta)),
                    "selected_sign_best_delta_bpb": float(np.min(selected_delta)),
                    "selected_sign_fraction_better": float(np.mean(selected_delta < 0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(transfer_rows)


def aggressive_family_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, anchor in TARGET_ANCHORS.items():
        delta_column = f"{target}_delta_vs_control"
        frame = results.loc[results["anchor_id"].eq(anchor) & ~results["contrast_family"].eq("center_control")]
        for family, family_frame in frame.groupby("contrast_family"):
            best_index = family_frame[delta_column].idxmin()
            rows.append(
                {
                    "target": target,
                    "anchor_id": anchor,
                    "contrast_family": family,
                    "n_policies": len(family_frame),
                    "mean_delta_bpb": float(family_frame[delta_column].mean()),
                    "best_observed_delta_bpb": float(family_frame.loc[best_index, delta_column]),
                    "best_candidate_id": family_frame.loc[best_index, "candidate_id"],
                    "count_gain_ge_0p005": int((family_frame[delta_column] <= -0.005).sum()),
                    "count_gain_ge_0p010": int((family_frame[delta_column] <= -0.010).sum()),
                }
            )
    return pd.DataFrame(rows)


def anchor_aggregate_weights(
    results: pd.DataFrame,
    anchor: str,
    phase_0_columns: list[str],
) -> np.ndarray:
    control = results.loc[results["anchor_id"].eq(anchor) & results["contrast_family"].eq("center_control")].iloc[0]
    weights = control[phase_0_columns].to_numpy(dtype=float)
    assert np.isclose(weights.sum(), 1, atol=1e-10)
    assert np.all(weights > 0)
    return weights


def maximum_symmetric_pair_tv(
    aggregate: np.ndarray,
    first_index: int,
    second_index: int,
    phase_0_fraction: float,
    phase_1_fraction: float,
) -> float:
    """Return the largest TV for which both signs of a pair transport are feasible."""
    pair_weights = aggregate[[first_index, second_index]]
    bounds = np.concatenate(
        [
            pair_weights / phase_0_fraction,
            pair_weights / phase_1_fraction,
            (1 - pair_weights) / phase_0_fraction,
            (1 - pair_weights) / phase_1_fraction,
        ]
    )
    return float(np.min(bounds))


def maximum_feasible_linear_response(
    gradient: np.ndarray,
    aggregate: np.ndarray,
    phase_tv: float,
    phase_0_fraction: float,
    phase_1_fraction: float,
) -> tuple[float, float]:
    """Solve the box-constrained TV transport problem by greedy mass matching."""
    positive_capacity = np.minimum(
        aggregate / phase_1_fraction,
        (1 - aggregate) / phase_0_fraction,
    )
    negative_capacity = np.minimum(
        aggregate / phase_0_fraction,
        (1 - aggregate) / phase_1_fraction,
    )
    positive_order = np.argsort(-gradient)
    negative_order = np.argsort(gradient)
    positive_remaining = positive_capacity.copy()
    negative_remaining = negative_capacity.copy()
    positive_position = 0
    negative_position = 0
    transported = 0.0
    response = 0.0
    while transported < phase_tv - 1e-15:
        while positive_position < len(gradient) and positive_remaining[positive_order[positive_position]] <= 1e-15:
            positive_position += 1
        while negative_position < len(gradient) and negative_remaining[negative_order[negative_position]] <= 1e-15:
            negative_position += 1
        if positive_position == len(gradient) or negative_position == len(gradient):
            break
        recipient = positive_order[positive_position]
        donor = negative_order[negative_position]
        marginal_gain = gradient[recipient] - gradient[donor]
        if marginal_gain <= 0:
            break
        amount = min(
            positive_remaining[recipient],
            negative_remaining[donor],
            phase_tv - transported,
        )
        response += amount * marginal_gain
        transported += amount
        positive_remaining[recipient] -= amount
        negative_remaining[donor] -= amount
    return response, transported


def balanced_partition_geometry(
    results: pd.DataFrame,
    phase_0_fraction: float,
    phase_1_fraction: float,
    draws: int,
) -> pd.DataFrame:
    phase_0_columns = [
        column for column in results if column.startswith("phase_0_") and column not in AGGREGATE_ROLLUP_COLUMNS
    ]
    assert len(phase_0_columns) == PHASE_TANGENT_DIMENSION + 1, len(phase_0_columns)
    domains = [column.removeprefix("phase_0_") for column in phase_0_columns]
    phase_1_columns = [f"phase_1_{domain}" for domain in domains]
    rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(sorted(results["anchor_id"].unique())):
        frame = results.loc[
            results["anchor_id"].eq(anchor)
            & results["contrast_family"].eq("balanced_partition")
            & results["sign"].eq("plus")
            & results["target_phase_tv"].eq(TV_VALUES[0])
        ].sort_values("direction_id")
        assert len(frame) == 16, (anchor, len(frame))
        contrast = frame[phase_1_columns].to_numpy(dtype=float) - frame[phase_0_columns].to_numpy(dtype=float)
        direction_tv = 0.5 * np.abs(contrast).sum(axis=1)
        assert np.allclose(direction_tv, TV_VALUES[0], atol=1e-10), direction_tv
        unit_directions = contrast / direction_tv[:, None]
        assert np.allclose(0.5 * np.abs(unit_directions).sum(axis=1), 1)
        aggregate = anchor_aggregate_weights(results, anchor, phase_0_columns)
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, SEED_TAG_GEOMETRY, anchor_index]))
        for phase_tv in TV_VALUES:
            for support_size in (len(domains), 10, 4, 2):
                unconstrained_ratios = []
                feasible_ratios = []
                transported_fractions = []
                for _ in range(draws):
                    gradient = np.zeros(len(domains))
                    support = rng.choice(
                        len(domains),
                        size=support_size,
                        replace=False,
                    )
                    gradient[support] = rng.normal(size=support_size)
                    sampled_rms = phase_tv * float(np.sqrt(np.mean((unit_directions @ gradient) ** 2)))
                    if sampled_rms <= 1e-12:
                        continue
                    unconstrained_response = phase_tv * float(np.max(gradient) - np.min(gradient))
                    feasible_response, transported = maximum_feasible_linear_response(
                        gradient,
                        aggregate,
                        phase_tv,
                        phase_0_fraction,
                        phase_1_fraction,
                    )
                    unconstrained_ratios.append(unconstrained_response / sampled_rms)
                    feasible_ratios.append(feasible_response / sampled_rms)
                    transported_fractions.append(transported / phase_tv)
                unconstrained = np.asarray(unconstrained_ratios)
                feasible = np.asarray(feasible_ratios)
                transported = np.asarray(transported_fractions)
                rows.append(
                    {
                        "anchor_id": anchor,
                        "target_phase_tv": phase_tv,
                        "gradient_support_size": support_size,
                        "n_simulations": len(feasible),
                        "sampled_direction_l2_norm_mean": float(np.linalg.norm(unit_directions, axis=1).mean()),
                        "unconstrained_max_over_sampled_rms_median": float(np.median(unconstrained)),
                        "feasible_max_over_sampled_rms_median": float(np.median(feasible)),
                        "feasible_max_over_sampled_rms_q025": float(np.quantile(feasible, 0.025)),
                        "feasible_max_over_sampled_rms_q975": float(np.quantile(feasible, 0.975)),
                        "transported_tv_fraction_median": float(np.median(transported)),
                    }
                )
    return pd.DataFrame(rows)


def phase_design_information(
    treatments: pd.DataFrame,
    aggressive_results: pd.DataFrame,
    noise_sources: NoiseSources,
    phase_0_fraction: float,
    phase_1_fraction: float,
) -> pd.DataFrame:
    r"""Audit local phase-gradient information under the exposed designs.

    The local model is \(L(a, d)-L(a, 0)=g(a)^T d+\epsilon\), where phase
    contrast \(d=p_1-p_0\) sums to zero. Random treatment-minus-control
    differences have variance \(2(1-\rho)\sigma^2\). Balanced antithetic odd
    responses \((L(+d)-L(-d))/2\) have variance
    \((1-\rho)\sigma^2/2\) under compound-symmetric run noise.
    """
    phase_0_columns = [
        column
        for column in aggressive_results
        if column.startswith("phase_0_") and column not in AGGREGATE_ROLLUP_COLUMNS
    ]
    assert len(phase_0_columns) == PHASE_TANGENT_DIMENSION + 1, len(phase_0_columns)
    domains = [column.removeprefix("phase_0_") for column in phase_0_columns]
    phase_1_columns = [f"phase_1_{domain}" for domain in domains]
    tangent_basis = null_space(np.ones((1, len(domains))))
    assert tangent_basis.shape == (len(domains), PHASE_TANGENT_DIMENSION)

    z_alpha = norm.ppf(1 - POWER_ALPHA / 2)
    z_power = norm.ppf(POWER_TARGET)
    target_se_for_goal = GOAL_GAIN_BPB / (z_alpha + z_power)
    rows: list[dict[str, Any]] = []
    for target, anchor in TARGET_ANCHORS.items():
        random_frame = treatments.loc[treatments["anchor_id"].eq(anchor)]
        random_contrasts = np.array(
            [
                [
                    json.loads(row.phase_1_weights_json)[domain] - json.loads(row.phase_0_weights_json)[domain]
                    for domain in domains
                ]
                for row in random_frame.itertuples()
            ],
            dtype=float,
        )
        balanced_frame = aggressive_results.loc[
            aggressive_results["anchor_id"].eq(anchor)
            & aggressive_results["contrast_family"].eq("balanced_partition")
            & aggressive_results["sign"].eq("plus")
        ]
        balanced_contrasts = balanced_frame[phase_1_columns].to_numpy(dtype=float) - balanced_frame[
            phase_0_columns
        ].to_numpy(dtype=float)
        assert random_contrasts.shape == (144, len(domains)), random_contrasts.shape
        assert balanced_contrasts.shape == (48, len(domains)), balanced_contrasts.shape
        assert np.allclose(random_contrasts.sum(axis=1), 0, atol=1e-10)
        assert np.allclose(balanced_contrasts.sum(axis=1), 0, atol=1e-10)

        random_information = np.zeros((PHASE_TANGENT_DIMENSION, PHASE_TANGENT_DIMENSION))
        for seed_block in sorted(random_frame["seed_block"].unique()):
            block_mask = random_frame["seed_block"].eq(seed_block).to_numpy()
            block_design = random_contrasts[block_mask] @ tangent_basis
            normalized_covariance = np.eye(len(block_design)) + np.ones((len(block_design), len(block_design)))
            random_information += block_design.T @ np.linalg.solve(normalized_covariance, block_design)
        balanced_design = balanced_contrasts @ tangent_basis
        balanced_information = 2 * balanced_design.T @ balanced_design
        information_matrices = {
            "random_fiber_shared_control_gls": random_information,
            "balanced_antithetic_odd": balanced_information,
            "combined": random_information + balanced_information,
        }
        model_response_counts = {
            "random_fiber_shared_control_gls": len(random_contrasts),
            "balanced_antithetic_odd": len(balanced_contrasts),
            "combined": len(random_contrasts) + len(balanced_contrasts),
        }
        random_control_runs = int(random_frame["seed_block"].nunique())
        training_run_counts = {
            "random_fiber_shared_control_gls": len(random_contrasts) + random_control_runs,
            "balanced_antithetic_odd": 2 * len(balanced_contrasts),
            "combined": len(random_contrasts) + random_control_runs + 2 * len(balanced_contrasts),
        }
        repeat_sd = float(noise_sources[target][PANEL_NOISE_SOURCE_ID]["sd"])
        aggregate = anchor_aggregate_weights(
            aggressive_results,
            anchor,
            phase_0_columns,
        )
        for design_id, information in information_matrices.items():
            eigenvalues = np.linalg.eigvalsh(information)
            positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-12]
            rank = len(positive)
            condition_number = float(positive[-1] / positive[0]) if rank > 0 else math.inf
            for correlation in (0.0, 0.5, 0.9):
                for requested_phase_tv in (0.1, 0.5):
                    feasible_phase_tvs = []
                    query_contrasts = []
                    for first_index in range(len(domains)):
                        for second_index in range(first_index):
                            maximum_tv = maximum_symmetric_pair_tv(
                                aggregate,
                                first_index,
                                second_index,
                                phase_0_fraction,
                                phase_1_fraction,
                            )
                            actual_tv = min(requested_phase_tv, maximum_tv)
                            sparse_contrast = np.zeros(len(domains))
                            sparse_contrast[first_index] = actual_tv
                            sparse_contrast[second_index] = -actual_tv
                            feasible_phase_tvs.append(actual_tv)
                            query_contrasts.append(sparse_contrast @ tangent_basis)
                    feasible_tvs = np.asarray(feasible_phase_tvs)
                    tangent_queries = np.asarray(query_contrasts)
                    row: dict[str, Any] = {
                        "target": target,
                        "anchor_id": anchor,
                        "design": design_id,
                        "n_model_responses": model_response_counts[design_id],
                        "n_training_runs": training_run_counts[design_id],
                        "tangent_dimension": PHASE_TANGENT_DIMENSION,
                        "design_rank": rank,
                        "condition_number_on_identified_subspace": condition_number,
                        "assumed_common_seed_correlation": correlation,
                        "data_seed_repeat_sd_bpb": repeat_sd,
                        "requested_sparse_pair_phase_tv": requested_phase_tv,
                        "n_sparse_pairs": len(feasible_tvs),
                        "n_pairs_reaching_requested_tv": int(
                            np.sum(
                                np.isclose(
                                    feasible_tvs,
                                    requested_phase_tv,
                                    atol=1e-12,
                                )
                            )
                        ),
                        "actual_feasible_phase_tv_median": float(np.median(feasible_tvs)),
                        "actual_feasible_phase_tv_q90": float(np.quantile(feasible_tvs, 0.9)),
                        "actual_feasible_phase_tv_max": float(np.max(feasible_tvs)),
                        "goal_gain_bpb": GOAL_GAIN_BPB,
                    }
                    if rank < PHASE_TANGENT_DIMENSION:
                        row.update(
                            {
                                "median_prediction_se_bpb": math.nan,
                                "q90_prediction_se_bpb": math.nan,
                                "max_prediction_se_bpb": math.nan,
                                "median_detectable_gain_80pct_power_bpb": math.nan,
                                "q90_detectable_gain_80pct_power_bpb": math.nan,
                                "fraction_pairs_resolving_goal": 0.0,
                                "same_design_replication_multiplier_for_80pct_power": math.inf,
                                "same_design_replication_equivalent_observations": math.inf,
                            }
                        )
                        rows.append(row)
                        continue

                    covariance_without_noise = np.linalg.inv(information)
                    noise_scale = repeat_sd * math.sqrt(1 - correlation)
                    variances = noise_scale**2 * np.einsum(
                        "bi,ij,bj->b",
                        tangent_queries,
                        covariance_without_noise,
                        tangent_queries,
                    )
                    standard_errors = np.sqrt(np.maximum(variances, 0))
                    detectable_gains = (z_alpha + z_power) * standard_errors
                    median_se = float(np.median(standard_errors))
                    replication_multiplier = (median_se / target_se_for_goal) ** 2
                    row.update(
                        {
                            "median_prediction_se_bpb": median_se,
                            "q90_prediction_se_bpb": float(np.quantile(standard_errors, 0.9)),
                            "max_prediction_se_bpb": float(np.max(standard_errors)),
                            "median_detectable_gain_80pct_power_bpb": float(np.median(detectable_gains)),
                            "q90_detectable_gain_80pct_power_bpb": float(np.quantile(detectable_gains, 0.9)),
                            "fraction_pairs_resolving_goal": float(np.mean(detectable_gains <= GOAL_GAIN_BPB)),
                            "same_design_replication_multiplier_for_80pct_power": replication_multiplier,
                            "same_design_replication_equivalent_observations": (
                                training_run_counts[design_id] * replication_multiplier
                            ),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def ideal_phase_budget_envelope(noise_sources: NoiseSources) -> pd.DataFrame:
    r"""Best-case information benchmark for a phase-order budget.

    Each phase direction uses an antithetic pair, so ``phase_runs / 2`` odd
    responses are available. For an ideal isotropic design whose directions
    and queries have the same TV radius, the minimum average prediction
    variance of a 38-dimensional linear phase field is

    \[
    \operatorname{Var}(\hat g^\top d)
      = \sigma^2 (1-\rho) q / B,
    \]

    where ``q`` is the tangent dimension, ``B`` is the number of training runs,
    and ``rho`` is the within-pair run-noise correlation. The TV radius cancels
    only because both design and query are scaled together under a linear
    model. This is an information-theoretic best case, not a claim that a
    simplex-feasible isotropic design exists at a frontier anchor.
    """
    z_alpha = norm.ppf(1 - POWER_ALPHA / 2)
    z_power = norm.ppf(POWER_TARGET)
    rows: list[dict[str, Any]] = []
    minimum_full_rank_runs = 2 * PHASE_TANGENT_DIMENSION
    for target, sources in noise_sources.items():
        repeat_sd = float(sources[PANEL_NOISE_SOURCE_ID]["sd"])
        for correlation in (0.0, 0.5, 0.9):
            for phase_runs in PHASE_RUN_BUDGETS:
                phase_pairs = phase_runs // 2
                full_rank_possible = phase_pairs >= PHASE_TANGENT_DIMENSION
                best_case_se = (
                    repeat_sd * math.sqrt((1 - correlation) * PHASE_TANGENT_DIMENSION / phase_runs)
                    if full_rank_possible
                    else math.inf
                )
                detectable_gain = (z_alpha + z_power) * best_case_se if full_rank_possible else math.inf
                rows.append(
                    {
                        "target": target,
                        "total_fit_budget": TOTAL_FIT_BUDGET,
                        "phase_run_budget": phase_runs,
                        "tied_run_budget": TOTAL_FIT_BUDGET - phase_runs,
                        "antithetic_phase_pairs": phase_pairs,
                        "minimum_full_rank_phase_runs": minimum_full_rank_runs,
                        "full_rank_possible": full_rank_possible,
                        "assumed_within_pair_correlation": correlation,
                        "data_seed_repeat_sd_bpb": repeat_sd,
                        "best_case_isotropic_average_prediction_se_bpb": best_case_se,
                        "best_case_detectable_gain_80pct_power_bpb": detectable_gain,
                        "goal_gain_bpb": GOAL_GAIN_BPB,
                        "goal_resolvable_in_best_case": detectable_gain <= GOAL_GAIN_BPB,
                        "simplex_feasibility_assumed": True,
                        "linear_radius_transfer_assumed": True,
                    }
                )
    return pd.DataFrame(rows)


def noise_sources_for_panel(
    resolution: dict[str, Any],
    aggressive_results: pd.DataFrame,
) -> NoiseSources:
    """Load sensitivity noise sources and replace matched noise with fresh controls."""
    sources: NoiseSources = {}
    for target in TARGET_COLUMNS:
        target_sources: dict[str, NoiseSource] = {}
        for source in resolution["targets"][target]["noise_sources"]:
            source_id = source["id"]
            if source_id not in UPSTREAM_NOISE_SOURCE_IDS:
                continue
            if source_id == "matched_frontier":
                source_id = ARCHIVE_FRONTIER_NOISE_SOURCE_ID
            target_sources[source_id] = {
                "sd": float(source["sd"]),
                "n": int(source["n"]),
            }
        sources[target] = target_sources

    controls = aggressive_results.loc[aggressive_results["contrast_family"].eq("center_control")]
    for target, anchor in TARGET_ANCHORS.items():
        values = controls.loc[controls["anchor_id"].eq(anchor), TARGET_COLUMNS[target]].to_numpy(dtype=float)
        assert len(values) == 16, (target, anchor, len(values))
        sources[target][PANEL_NOISE_SOURCE_ID] = {
            "sd": float(np.std(values, ddof=1)),
            "n": len(values),
        }
    assert all(set(target_sources) == set(NOISE_SOURCE_IDS) for target_sources in sources.values())
    return sources


def paired_seed_correlation_estimates(
    aggressive_results: pd.DataFrame,
    noise_sources: NoiseSources,
) -> pd.DataFrame:
    """Estimate same-seed delta variance from replicated Dolmino endpoints."""
    rows = []
    for target, anchor in TARGET_ANCHORS.items():
        frame = aggressive_results.loc[
            aggressive_results["anchor_id"].eq(anchor)
            & aggressive_results["contrast_family"].eq("dolmino_late_continuum")
        ]
        delta_column = f"{target}_delta_vs_control"
        pooled_sum_squares = 0.0
        degrees_of_freedom = 0
        for _, group in frame.groupby("direction_id"):
            values = group[delta_column].to_numpy(dtype=float)
            assert len(values) == 3
            pooled_sum_squares += (len(values) - 1) * np.var(values, ddof=1)
            degrees_of_freedom += len(values) - 1
        pooled_variance = pooled_sum_squares / degrees_of_freedom
        repeat_sd = float(noise_sources[target][PANEL_NOISE_SOURCE_ID]["sd"])
        lower_variance = degrees_of_freedom * pooled_variance / chi2.ppf(0.975, degrees_of_freedom)
        upper_variance = degrees_of_freedom * pooled_variance / chi2.ppf(0.025, degrees_of_freedom)
        rows.append(
            {
                "target": target,
                "anchor_id": anchor,
                "replicated_endpoint_groups": 3,
                "variance_degrees_of_freedom": degrees_of_freedom,
                "data_seed_repeat_sd_bpb": repeat_sd,
                "same_seed_delta_sd_bpb": math.sqrt(pooled_variance),
                "implied_pair_correlation": 1 - pooled_variance / (2 * repeat_sd**2),
                "implied_pair_correlation_ci95_low": 1 - upper_variance / (2 * repeat_sd**2),
                "implied_pair_correlation_ci95_high": 1 - lower_variance / (2 * repeat_sd**2),
                "correlation_identified": False,
            }
        )
    return pd.DataFrame(rows)


def detection_table(noise_sources: NoiseSources) -> pd.DataFrame:
    rows = []
    z_alpha = norm.ppf(0.975)
    z_power = norm.ppf(0.80)
    for target, sources in noise_sources.items():
        for source_id, source in sources.items():
            sd = source["sd"]
            for delta in (0.001, 0.002, 0.005, 0.010):
                for correlation in (0.0, 0.5, 0.9):
                    variance = 2 * (1 - correlation) * sd**2
                    repeats = math.ceil((z_alpha + z_power) ** 2 * variance / delta**2)
                    rows.append(
                        {
                            "target": target,
                            "noise_source": source_id,
                            "data_seed_repeat_sd_bpb": sd,
                            "gain_bpb": delta,
                            "assumed_pair_correlation": correlation,
                            "replicates_per_policy_for_80pct_power": max(repeats, 1),
                            "gain_in_repeat_sd": delta / sd,
                        }
                    )
    return pd.DataFrame(rows)


def null_selection_table(
    noise_sources: NoiseSources,
    aggressive_results: pd.DataFrame,
    draws: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        target_anchor = aggressive_results.loc[
            aggressive_results["anchor_id"].eq(TARGET_ANCHORS[target])
            & ~aggressive_results["contrast_family"].eq("center_control")
        ]
        candidates_per_shared_control = target_anchor.groupby("seed_block").size().sort_index().tolist()
        assert len(candidates_per_shared_control) == 16
        assert sum(candidates_per_shared_control) == 129
        selection_sets = {
            "balanced_partition_per_tv": [2] * 16,
            "all_aggressive_target_matched": candidates_per_shared_control,
        }
        sources = noise_sources[target]
        for source_id in NOISE_SOURCE_IDS:
            source = sources[source_id]
            source_index = NOISE_SOURCE_IDS.index(source_id)
            for correlation_index, correlation in enumerate((0.0, 0.5, 0.9)):
                residual_sd = source["sd"] * math.sqrt(1 - correlation)
                for selection_index, (selection_set, group_sizes) in enumerate(selection_sets.items()):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                BOOTSTRAP_SEED,
                                SEED_TAG_NULL_SELECTION,
                                target_index,
                                source_index,
                                correlation_index,
                                selection_index,
                            ]
                        )
                    )
                    group_minima = []
                    for candidates_per_control in group_sizes:
                        candidate_residual = rng.normal(
                            scale=residual_sd,
                            size=(draws, candidates_per_control),
                        )
                        control_residual = rng.normal(
                            scale=residual_sd,
                            size=(draws, 1),
                        )
                        group_minima.append(np.min(candidate_residual - control_residual, axis=1))
                    selected = np.min(np.column_stack(group_minima), axis=1)
                    rows.append(
                        {
                            "target": target,
                            "noise_source": source_id,
                            "selection_set": selection_set,
                            "n_candidates": sum(group_sizes),
                            "n_control_groups": len(group_sizes),
                            "min_candidates_per_control": min(group_sizes),
                            "max_candidates_per_control": max(group_sizes),
                            "assumed_common_seed_correlation": correlation,
                            "assumption": "compound-symmetric run noise with the panel's actual shared-control groups",
                            "expected_best_selected_delta_bpb": float(np.mean(selected)),
                            "probability_best_le_minus_0p01": float(np.mean(selected <= -GOAL_GAIN_BPB)),
                        }
                    )
    return pd.DataFrame(rows)


def write_random_plot(strata: pd.DataFrame, output_path: Path) -> None:
    target_matched = strata.loc[strata["target_matched"]]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable frontier anchor", "Table-9 frontier anchor"),
        shared_yaxes=False,
    )
    colors = {"uncheatable": "#2A788E", "table9": "#D95F0E"}
    for column_index, target in enumerate(("uncheatable", "table9"), start=1):
        frame = target_matched.loc[target_matched["target"].eq(target)]
        figure.add_trace(
            go.Scatter(
                x=frame["radius_fraction"],
                y=frame["mean_delta_bpb"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": frame["ci95_high_bpb"] - frame["mean_delta_bpb"],
                    "arrayminus": frame["mean_delta_bpb"] - frame["ci95_low_bpb"],
                },
                mode="lines+markers",
                marker={"size": 10, "color": colors[target]},
                line={"width": 3, "color": colors[target]},
                name=target,
                customdata=np.stack(
                    [
                        frame["mean_phase_tv"],
                        frame["max_phase_tv"],
                        frame["fraction_better"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "radius fraction %{x:.2f}<br>"
                    "mean delta %{y:+.6f} BPB<br>"
                    "mean TV %{customdata[0]:.4f}<br>"
                    "max TV %{customdata[1]:.4f}<br>"
                    "fraction better %{customdata[2]:.1%}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
        figure.add_hline(y=0, line_dash="dash", line_color="#334E68", row=1, col=column_index)
    figure.update_xaxes(title_text="Fraction of first-boundary feasible radius")
    figure.update_yaxes(title_text="Mean BPB minus same-seed tied control")
    figure.update_layout(
        title="Random-fiber radius scaling: direction-averaged response",
        template="plotly_white",
        height=560,
        width=1200,
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        config=PLOT_CONFIG,
        div_id="phase-order-random-radius-scaling",
    )


def write_aggressive_plot(summary: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable frontier anchor", "Table-9 frontier anchor"),
    )
    metric_styles = {
        "odd_rms_bpb": ("Observed odd RMS", "#2A788E", "circle"),
        "mean_curvature_bpb": ("Mean symmetric cost", "#D95F0E", "square"),
    }
    for column_index, target in enumerate(("uncheatable", "table9"), start=1):
        frame = summary.loc[summary["target"].eq(target)]
        for metric, (label, color, symbol) in metric_styles.items():
            figure.add_trace(
                go.Scatter(
                    x=frame["target_phase_tv"],
                    y=frame[metric],
                    mode="lines+markers",
                    marker={"size": 9, "symbol": symbol, "color": color},
                    line={"width": 2.5, "color": color},
                    name=label,
                    legendgroup=metric,
                    showlegend=column_index == 1,
                ),
                row=1,
                col=column_index,
            )
    figure.update_xaxes(title_text="Phase total variation")
    figure.update_yaxes(title_text="BPB scale")
    figure.update_layout(
        title=("Aggressive antithetic panel: measured odd signal and symmetric cost"),
        template="plotly_white",
        height=600,
        width=1200,
        legend={"orientation": "h", "y": -0.18},
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        config=PLOT_CONFIG,
        div_id="phase-order-aggressive-odd-even",
    )


def write_low_epsilon_headroom_plot(summary: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable", "Table-9 macro"),
        shared_yaxes=True,
    )
    colors = {
        "conditional_gain_vs_anchor_tied_bpb": "#1a9850",
        "global_gain_vs_best_tied_bpb": "#f46d43",
    }
    labels = {
        "conditional_gain_vs_anchor_tied_bpb": "Conditional gain vs anchor tied",
        "global_gain_vs_best_tied_bpb": "Gain vs best tied aggregate",
    }
    for column_index, objective in enumerate(("uncheatable", "table9"), start=1):
        frame = summary.loc[summary["objective"].eq(objective)].copy()
        frame["path"] = frame["anchor_tag"] + " / " + frame["family"].str.replace("_", " ")
        for metric in labels:
            figure.add_trace(
                go.Bar(
                    x=frame["path"],
                    y=frame[metric],
                    marker={"color": colors[metric]},
                    name=labels[metric],
                    legendgroup=metric,
                    showlegend=column_index == 1,
                    customdata=np.column_stack(
                        [
                            frame["selected_candidate"],
                            frame["selected_epsilon_phase"],
                            frame["selected_phase_tv"],
                            frame["selected_observed_bpb"],
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>" + labels[metric] + ": %{y:+.6f} BPB<br>"
                        "candidate: %{customdata[0]}<br>"
                        "epsilon_phase: %{customdata[1]:.4f}<br>"
                        "phase TV: %{customdata[2]:.4f}<br>"
                        "observed BPB: %{customdata[3]:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
        figure.add_hline(y=0, line={"color": "#18354a", "width": 1.5}, row=1, col=column_index)

    figure.update_yaxes(title_text="Observed gain (BPB; positive is better)", row=1, col=1)
    figure.update_xaxes(tickangle=-20)
    figure.update_layout(
        title="Low-epsilon phase-order headroom: conditional versus global",
        template="plotly_white",
        height=620,
        width=1250,
        barmode="group",
        legend={"orientation": "h", "y": -0.24},
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        config=PLOT_CONFIG,
        div_id="phase-order-low-epsilon-headroom",
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".6f")


def write_report(
    output_dir: Path,
    random_strata: pd.DataFrame,
    power_fits: pd.DataFrame,
    aggressive: pd.DataFrame,
    aggressive_families: pd.DataFrame,
    geometry: pd.DataFrame,
    transfer: pd.DataFrame,
    detection: pd.DataFrame,
    null_selection: pd.DataFrame,
    paired_correlation: pd.DataFrame,
    design_information: pd.DataFrame,
    ideal_budget: pd.DataFrame,
    low_epsilon: pd.DataFrame,
) -> None:
    random_primary = random_strata.loc[random_strata["target_matched"]]
    random_transfer = random_strata.loc[~random_strata["target_matched"]]
    power_primary = power_fits.loc[power_fits["target_matched"]]
    detection_goal = detection.loc[
        detection["gain_bpb"].eq(GOAL_GAIN_BPB)
        & detection["assumed_pair_correlation"].eq(0)
        & detection["noise_source"].eq(PANEL_NOISE_SOURCE_ID)
    ]
    null_goal = null_selection.loc[null_selection["noise_source"].eq(PANEL_NOISE_SOURCE_ID)]
    matched_sd = detection_goal.set_index("target")["data_seed_repeat_sd_bpb"]
    combined_information = design_information.loc[
        design_information["design"].eq("combined") & design_information["assumed_common_seed_correlation"].eq(0)
    ]
    pair_feasibility = combined_information.loc[
        combined_information["requested_sparse_pair_phase_tv"].eq(0.1),
        [
            "target",
            "requested_sparse_pair_phase_tv",
            "actual_feasible_phase_tv_median",
            "actual_feasible_phase_tv_q90",
            "actual_feasible_phase_tv_max",
            "n_pairs_reaching_requested_tv",
        ],
    ]
    ideal_budget_independent = ideal_budget.loc[
        ideal_budget["assumed_within_pair_correlation"].eq(0)
        & ideal_budget["phase_run_budget"].isin((80, 140, 200, 280))
    ]
    uncheatable_low_epsilon = low_epsilon.loc[
        low_epsilon["objective"].eq("uncheatable")
        & low_epsilon["anchor_tag"].eq("unch05")
        & low_epsilon["family"].eq("effective_exposure")
    ].iloc[0]
    table9_stable_low_epsilon = low_epsilon.loc[
        low_epsilon["objective"].eq("table9")
        & low_epsilon["anchor_tag"].eq("t9s05")
        & low_epsilon["family"].eq("effective_exposure")
    ].iloc[0]
    table9_best_tied_low_epsilon = low_epsilon.loc[
        low_epsilon["objective"].eq("table9")
        & low_epsilon["anchor_tag"].eq("t9b075")
        & low_epsilon["family"].eq("effective_exposure")
    ].iloc[0]

    report = rf"""# Phase-order identifiability and headroom audit

## Decision

The exposed 3e18 panels do **not yet identify a transferable phase-order
model** that supports a 0.01-BPB gain over a tied frontier policy. They also do
not prove that such a direction is globally absent.

The corrected diagnosis is not that 280 runs or evaluation noise make phase
order unmeasurable:

* The random isotropic panel reached `{random_primary["max_phase_tv"].max():.4f}`
  maximum phase TV. Even extending every ray to its first simplex boundary
  would reach at most `{random_primary["max_full_boundary_phase_tv"].max():.4f}`;
  this is a hard geometric ceiling for those directions, not merely an
  incomplete radius sweep.
* The balanced-partition panel reached TV 0.50, but every target-matched
  direction was worse than tied at that radius. Its lower-radius ordering signs
  do not reliably select beneficial higher-radius policies.
* Under a favorable local linear model, the exposed combined design can estimate
  many **simplex-feasible** pair transports to a precision finer than the
  0.01-BPB target. The failure is transfer: the ordering response changes across
  radii and anchors, while symmetric harm grows.
* The one handcrafted Table-9 observation below -0.01 BPB was selected from
  129 target-matched policies. At the fresh-control data-seed SD of
  {matched_sd["table9"]:.4f} BPB, such a selected excursion is compatible with
  the multiplicity-aware null for same-seed correlations up to 0.5, spanning
  most of the estimated confidence interval.

No sealed targeted-pairwise outcomes were accessed.

## Observed low-epsilon headroom

Two estimands answer different questions:

\[
H_\mathrm{{conditional}}(a)
=L(a,0)-\min_d L(a,d)
\]

holds the aggregate policy \(a\) fixed, while

\[
H_\mathrm{{global}}
=\min_a L(a,0)-\min_{{a,d}}L(a,d)
\]

compares against the best tested tied aggregate. Positive values are
improvements. The table selects the best observed point only within the
prespecified low-epsilon range \(\epsilon_\mathrm{{phase}}\le
{LOW_EPSILON_MAX:.2f}\):

{
        markdown_table(
            low_epsilon,
            [
                "objective",
                "anchor_tag",
                "family",
                "candidate_count",
                "selected_candidate",
                "selected_epsilon_phase",
                "selected_phase_tv",
                "selected_observed_bpb",
                "anchor_tied_bpb",
                "best_tied_bpb",
                "conditional_gain_vs_anchor_tied_bpb",
                "global_gain_vs_best_tied_bpb",
                "selected_at_low_epsilon_boundary",
            ],
        )
    }

The Uncheatable effective-exposure path provides the cleanest observed
frontier evidence: its best tested point improves its tied anchor and the best
tied aggregate by
`{uncheatable_low_epsilon["conditional_gain_vs_anchor_tied_bpb"]:.6f}` BPB.
That is the reported approximately 0.003-BPB dip. It is still a one-seed,
selected path point; the source analysis estimated it at 2.06 independent-run
difference SDs and found the neighboring epsilon values worse.

For Table-9, the approximately 0.009-BPB result is real as a **conditional**
statement: asymmetry improves the weaker `t9s05` tied aggregate by
`{table9_stable_low_epsilon["conditional_gain_vs_anchor_tied_bpb"]:.6f}` BPB.
But its gain over the best tested tied aggregate is only
`{table9_stable_low_epsilon["global_gain_vs_best_tied_bpb"]:.6f}` BPB.
Moreover, applying the same family to the better `t9b075` aggregate loses
`{-table9_best_tied_low_epsilon["conditional_gain_vs_anchor_tied_bpb"]:.6f}`
BPB at its best low-epsilon point. The `t9s05` winner also lies at the upper
boundary of this low-epsilon sweep.

This is useful mechanistic evidence: phase order can rescue a suboptimal
aggregate, and the phase response depends strongly on aggregate location.
It is not evidence for 0.009 BPB of globally recoverable Table-9 headroom.
Deepening the dip therefore requires learning \(d^\star(a)\), including its
radius-dependent symmetric cost, rather than extrapolating one phase direction
or optimizing aggregate and phase order independently.

## Random-fiber radius scaling

Negative deltas improve over the same-seed tied control. Confidence intervals
are a nested empirical bootstrap over the four seed blocks and their
directions; they are not a model-based causal interval. The panel uses the same
48 one-sided directions and four controls at every radius, so this is not an
antithetic even-response estimate.

{
        markdown_table(
            random_primary,
            [
                "target",
                "radius_fraction",
                "n_directions",
                "mean_phase_tv",
                "max_phase_tv",
                "mean_delta_bpb",
                "sd_delta_bpb",
                "ci95_low_bpb",
                "ci95_high_bpb",
                "fraction_better",
            ],
        )
    }

The signed power fit \(m(\rho)=c\rho^p\) is **not identified** from only three
radii and one residual degree of freedom. Across all four anchor-by-target
fits, three point estimates sit at the power-grid floor; the coefficient and
power are coupled, and the reported coefficient at radius 1 extrapolates
beyond the sampled range:

{
        markdown_table(
            power_primary,
            [
                "target",
                "coefficient_at_radius_1",
                "power",
                "coefficient_ci95_low",
                "coefficient_ci95_high",
                "power_at_grid_boundary_fraction",
                "profile_sse_max_over_min",
                "power_identified",
            ],
        )
    }

The strongest cross-target offset is also not evidence for a radius law. Phase
tilts around the Uncheatable anchor improve Table-9 by a nearly flat amount
across radii, which is compatible with a four-control offset:

{
        markdown_table(
            random_transfer,
            [
                "anchor_id",
                "target",
                "radius_fraction",
                "mean_delta_bpb",
                "ci95_low_bpb",
                "ci95_high_bpb",
                "fraction_better",
            ],
        )
    }

## Aggressive balanced-antithetic panel

For each balanced direction and TV, the odd response is
\((L(+d)-L(-d))/2\), while curvature is
\((L(+d)+L(-d))/2-L(0)\). The better of \(+d\) and \(-d\) beats its
direction-specific tied control with null probability 2/3 under exchangeability.
The two-sided binomial p-value shows which fractions differ from that null.

{
        markdown_table(
            aggressive,
            [
                "target",
                "target_phase_tv",
                "odd_rms_bpb",
                "mean_curvature_bpb",
                "mean_best_sign_delta_bpb",
                "best_observed_sign_delta_bpb",
                "fraction_best_sign_better",
                "fraction_best_sign_better_null",
                "fraction_best_sign_better_null_pvalue",
                "count_gain_ge_0p005",
                "count_gain_ge_0p010",
            ],
        )
    }

The control is shared by the two signs and three radii within each direction,
not across directions. At TV 0.50, 0 of 16 directions beat tied for either
target, while the symmetric cost is larger than the odd RMS.

## Feasible geometry

The unconstrained identity
\(\max_{{\sum_i d_i=0,\,\|d\|_1/2\le t}} g^\top d
=t(\max_i g_i-\min_i g_i)\)
ignores the phase-simplex box constraints. The corrected simulation solves the
actual transport problem
\[
p_0=a-\alpha_1d\ge0,\qquad p_1=a+\alpha_0d\ge0
\]
at each TV. `feasible_max_over_sampled_rms` is therefore an achievable linear
envelope at the anchor; the unconstrained column is retained only as an
algebraic reference.

{geometry.to_markdown(index=False, floatfmt=".4f")}

The geometry table describes an optimal multi-bucket transport, not one isolated
donor-recipient pair. At requested TV 0.25 and 0.50, the achievable fraction
falls sharply as scarce anchor mass saturates. Exact both-signs-feasible pair
transports are substantially smaller:

{pair_feasibility.to_markdown(index=False, floatfmt=".6f")}

The median pair reaches only about 0.008--0.009 phase TV, and only a few pairs
reach requested TV 0.10. This correction reduces, rather than increases,
evidence for a hidden 0.01-BPB sparse gain.

## All aggressive design families

The balanced table above is the only family with antithetic odd/even
decomposition. Target-matched raw outcomes across every exposed family are:

{aggressive_families.to_markdown(index=False, floatfmt=".6f")}

## Cross-radius selection

A useful phase-order model must do more than detect that order matters: its
preferred sign at one radius must select a beneficial policy at another.

{
        markdown_table(
            transfer,
            [
                "target",
                "train_tv",
                "test_tv",
                "odd_sign_agreement",
                "odd_pearson",
                "rmse_ratio_vs_zero",
                "selected_sign_mean_delta_bpb",
                "selected_sign_fraction_better",
            ],
        )
    }

At TV 0.50, selecting the sign from either TV 0.10 or 0.25 produces no
target-matched improvements in either objective. Uncheatable ordering signs
are more stable than Table-9 signs, but symmetric harm dominates them for this
balanced-partition design.

## Noise and the 0.01-BPB target

Under an independent-pair reference calculation, the number below gives
per-policy replications needed for 80% power at two-sided 5% significance.
The matched-anchor SD comes from the 16 fresh tied controls in this exact panel.
These controls vary the data seed while holding the trainer seed fixed.
Shared-seed covariance is not identified, so this is a sensitivity calculation:

{
        markdown_table(
            detection_goal,
            [
                "target",
                "noise_source",
                "data_seed_repeat_sd_bpb",
                "gain_bpb",
                "gain_in_repeat_sd",
                "replicates_per_policy_for_80pct_power",
            ],
        )
    }

The replicated Dolmino endpoints provide only six variance degrees of freedom
for estimating same-seed correlation:

{paired_correlation.to_markdown(index=False, floatfmt=".6f")}

The confidence intervals are too wide to identify the correlation. The null
selection table therefore spans correlations 0, 0.5, and 0.9. It distinguishes
the 32 balanced candidates at one TV from selection over all 129 target-matched
aggressive policies and reproduces the latter panel's 16 shared controls and
uneven candidate counts:

{null_goal.to_markdown(index=False, floatfmt=".6f")}

For Uncheatable, noise is too small to explain the absence of a 0.01 gain among
balanced partitions. For Table-9, selected excursions near -0.01 are common
under several defensible null settings. The complete source sensitivity,
including archive-median and proportional-repeat SDs, is in
`null_best_of_direction_selection.csv`.

## Local phase-gradient information

Consider the deliberately favorable local model
\[
L(a,d)-L(a,0)=g(a)^\top d+\epsilon,
\]
with no curvature, interactions, or aggregate uncertainty. Random
treatment-minus-control responses share one control across each 36-treatment
seed block, so their generalized least-squares covariance is proportional to
\(I+J\). Balanced antithetic odd responses contribute information
\(2D^\top D\), but span only 16 dimensions. The combined design is full rank.

For every bucket pair, the query TV is clipped to the largest value for which
**both signs** remain phase-simplex feasible. This matters: at requested TV
0.1, median feasible TV is only about 0.008-0.009.

{
        markdown_table(
            combined_information,
            [
                "target",
                "requested_sparse_pair_phase_tv",
                "actual_feasible_phase_tv_median",
                "actual_feasible_phase_tv_q90",
                "actual_feasible_phase_tv_max",
                "n_pairs_reaching_requested_tv",
                "n_training_runs",
                "design_rank",
                "condition_number_on_identified_subspace",
                "median_prediction_se_bpb",
                "q90_prediction_se_bpb",
                "max_prediction_se_bpb",
                "median_detectable_gain_80pct_power_bpb",
                "q90_detectable_gain_80pct_power_bpb",
                "fraction_pairs_resolving_goal",
            ],
        )
    }

The combined exposed design has enough nominal precision for many feasible
local transports. This is a favorable sensitivity result, not a validated
phase model: the random panel is one-sided rather than antithetic, its
full-rank contribution assumes the local linear form, and the observed
cross-radius failures show that curvature and radius transfer cannot be
ignored.

## Is 280 runs fundamentally insufficient?

No. For an ideal isotropic antithetic design, with design and query directions
at the same TV radius, the best-case average prediction standard error is
\[
\sigma\sqrt{{(1-\rho)q/B}},
\]
where \(q=38\) and \(B\) is the number of phase-training runs. At least 76
runs are needed merely to obtain 38 antithetic pairs and full tangent rank.

{
        markdown_table(
            ideal_budget_independent,
            [
                "target",
                "phase_run_budget",
                "tied_run_budget",
                "antithetic_phase_pairs",
                "best_case_isotropic_average_prediction_se_bpb",
                "best_case_detectable_gain_80pct_power_bpb",
                "goal_resolvable_in_best_case",
            ],
        )
    }

This is a lower envelope on average directional variance, not an achieved
design. Equality requires an isotropic full-rank set of contrasts with equal
norm. Frontier simplex constraints can make large antithetic moves infeasible
for low-mass buckets, and nonlinear curvature invalidates radius transfer.
Nevertheless, the calculation shows why a deliberate tied-plus-phase split
within 280 runs remains scientifically viable: the obstacle is intervention
geometry and mechanism identification, not raw evaluation noise or dimension
alone.

## Modeling consequence

1. Combined with the prior 99-route registry, exact-equivalence audits, and
   cross-session finite-potential-transport failures, more scalar recency,
   retention, or output-link variants are not a justified way to repair this
   identification gap.
2. A globally transferable additive phase gradient is not supported by the
   exposed pair-direction and cross-radius evidence, especially for Table-9.
3. The exposed panels can estimate small local transports, but they do not
   identify the nonlinear transition law required to move from those effects
   to a raw two-phase optimum. Fitting a new 38-D phase field to these
   development outcomes would be unsupported model selection.
4. Aggregate modeling remains separable and useful. In the prior development
   audits, adding the tested phase heads to a tied aggregate backbone did not
   clear the existing Observatory frontier.

This is a development-data identification result, not a claim that the true
global two-phase optimum is phase-tied.

## Evidence scope

The route-exhaustion and transfer statements above are supported by:

* `mechanistic_surrogate_discovery_20260719/final_synthesis/final_report.md`
* `mechanistic_surrogate_discovery_20260719/final_synthesis/approach_registry.csv`
* `mechanistic_surrogate_discovery_20260719/round53_partial_identification/report.md`
* `cross_session_phase_transport_20260723/FINAL_SYNTHESIS.md`
* `fixed_budget_aggregate_phase_survivor_uncertainty_20260724/report.md`

All paths are relative to `reference_outputs/`. No sealed targeted-pairwise
outcomes were inspected.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in (
        args.heldouts,
        args.antithetic_pairs,
        args.aggressive_results,
        args.resolution,
        args.aggressive_design_summary,
        args.low_epsilon_paths,
    ):
        assert "targeted_pairwise" not in str(input_path).lower(), input_path
    heldouts = pd.read_csv(args.heldouts)
    assert_sealed_panel_absent(heldouts, "heldouts")
    treatments = random_population_deltas(heldouts)
    pairs = pd.read_csv(args.antithetic_pairs)
    aggressive_results = pd.read_csv(args.aggressive_results)
    assert_sealed_panel_absent(pairs, "antithetic pairs")
    assert_sealed_panel_absent(aggressive_results, "aggressive results")
    low_epsilon_paths = pd.read_csv(args.low_epsilon_paths)
    assert_sealed_panel_absent(low_epsilon_paths, "low-epsilon paths")
    resolution = json.loads(args.resolution.read_text())
    design_summary = json.loads(args.aggressive_design_summary.read_text())
    phase_0_fraction = float(design_summary["realized_phase_fractions"]["phase_0"])
    phase_1_fraction = float(design_summary["realized_phase_fractions"]["phase_1"])
    assert np.isclose(phase_0_fraction + phase_1_fraction, 1)
    noise_sources = noise_sources_for_panel(resolution, aggressive_results)

    random_strata, power_fits = random_radius_analysis(
        treatments,
        args.bootstrap_draws,
    )
    aggressive, transfer = aggressive_analysis(pairs)
    aggressive_families = aggressive_family_summary(aggressive_results)
    geometry = balanced_partition_geometry(
        aggressive_results,
        phase_0_fraction,
        phase_1_fraction,
        draws=max(args.bootstrap_draws, 10_000),
    )
    detection = detection_table(noise_sources)
    null_selection = null_selection_table(
        noise_sources,
        aggressive_results,
        draws=max(args.bootstrap_draws * 20, 100_000),
    )
    paired_correlation = paired_seed_correlation_estimates(
        aggressive_results,
        noise_sources,
    )
    design_information = phase_design_information(
        treatments,
        aggressive_results,
        noise_sources,
        phase_0_fraction,
        phase_1_fraction,
    )
    ideal_budget = ideal_phase_budget_envelope(noise_sources)
    low_epsilon = low_epsilon_headroom(low_epsilon_paths)

    random_strata.to_csv(args.output_dir / "random_radius_strata.csv", index=False)
    power_fits.to_csv(args.output_dir / "random_radius_power_fits.csv", index=False)
    aggressive.to_csv(args.output_dir / "aggressive_odd_even_envelope.csv", index=False)
    aggressive_families.to_csv(
        args.output_dir / "aggressive_family_goal_summary.csv",
        index=False,
    )
    geometry.to_csv(args.output_dir / "balanced_partition_geometry.csv", index=False)
    transfer.to_csv(args.output_dir / "cross_radius_transfer.csv", index=False)
    detection.to_csv(args.output_dir / "detection_power.csv", index=False)
    null_selection.to_csv(args.output_dir / "null_best_of_direction_selection.csv", index=False)
    paired_correlation.to_csv(
        args.output_dir / "paired_seed_correlation_estimates.csv",
        index=False,
    )
    design_information.to_csv(
        args.output_dir / "phase_design_information.csv",
        index=False,
    )
    ideal_budget.to_csv(
        args.output_dir / "ideal_phase_budget_envelope.csv",
        index=False,
    )
    low_epsilon.to_csv(
        args.output_dir / "low_epsilon_conditional_global_headroom.csv",
        index=False,
    )
    write_random_plot(random_strata, args.output_dir / "random_radius_scaling.html")
    write_aggressive_plot(aggressive, args.output_dir / "aggressive_odd_even_envelope.html")
    write_low_epsilon_headroom_plot(
        low_epsilon,
        args.output_dir / "low_epsilon_conditional_global_headroom.html",
    )
    write_report(
        args.output_dir,
        random_strata,
        power_fits,
        aggressive,
        aggressive_families,
        geometry,
        transfer,
        detection,
        null_selection,
        paired_correlation,
        design_information,
        ideal_budget,
        low_epsilon,
    )

    target_matched_goal_count = int(aggressive_families["count_gain_ge_0p010"].sum())
    combined_independent = design_information.loc[
        design_information["design"].eq("combined")
        & design_information["assumed_common_seed_correlation"].eq(0)
        & design_information["requested_sparse_pair_phase_tv"].eq(0.1)
    ]
    summary = {
        "random_panel_rows": len(treatments),
        "aggressive_pair_rows": len(pairs),
        "bootstrap_draws": args.bootstrap_draws,
        "sealed_targeted_pairwise_panel_accessed": False,
        "goal_gain_bpb": GOAL_GAIN_BPB,
        "decision": "phase_order_gain_not_identified_from_exposed_panels",
        "phase_0_fraction": phase_0_fraction,
        "phase_1_fraction": phase_1_fraction,
        "random_panel_max_phase_tv": float(random_strata["max_phase_tv"].max()),
        "random_panel_full_boundary_max_phase_tv": float(random_strata["max_full_boundary_phase_tv"].max()),
        "balanced_partition_target_matched_goal_count": int(aggressive["count_gain_ge_0p010"].sum()),
        "all_family_target_matched_goal_count": target_matched_goal_count,
        "combined_design_rank": int(combined_independent["design_rank"].min()),
        "combined_design_median_feasible_pair_tv": {
            row.target: float(row.actual_feasible_phase_tv_median) for row in combined_independent.itertuples()
        },
        "combined_design_median_detectable_gain_80pct_power_bpb": {
            row.target: float(row.median_detectable_gain_80pct_power_bpb) for row in combined_independent.itertuples()
        },
        "paired_seed_correlation_identified": bool(paired_correlation["correlation_identified"].all()),
        "low_epsilon_headroom_bpb": {
            f"{row.objective}:{row.anchor_tag}:{row.family}": {
                "conditional_gain_vs_anchor_tied": float(row.conditional_gain_vs_anchor_tied_bpb),
                "global_gain_vs_best_tied": float(row.global_gain_vs_best_tied_bpb),
            }
            for row in low_epsilon.itertuples()
        },
        "raw_transferable_phase_optimum_identified": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
