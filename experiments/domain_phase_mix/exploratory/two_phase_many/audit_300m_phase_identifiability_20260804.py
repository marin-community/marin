# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy==1.16.3"]
# ///
"""Audit endpoint-noise and phase-field identifiability in the 300M panel."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import nnls
from scipy.stats import chi2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "three_hundred_m_phase_identifiability_20260804"
TRAJECTORY_PATH = SCRIPT_DIR / "reference_outputs" / "tied_two_phase_trajectory_audit_20260726" / "wandb_histories.csv"
BASELINE_PATH = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731" / "baseline_metrics.csv"
PROPORTIONAL_REFERENCE_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "one_phase_swarm_scores_export_300m_20260630"
    / "proportional_reference_uncheatable_table9_scores_300m.csv"
)
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {
    "uncheatable": "eval_uncheatable_eval_bpb",
    "table9": "table9_macro_bpb",
}
TRAJECTORY_STEPS = (19000, 20000, 21000, 22000)
NEIGHBOR_COUNTS = (1, 2, 4, 8)
SEMIVARIOGRAM_POWERS = (1, 2)
ROUND_DECIMALS = 12
CANONICAL_NULL_DRAWS = 400
CANONICAL_NULL_SEED = 20260804
PROTOCOL = {
    "candidate_id": "WSD80-SUR-079",
    "date": "2026-08-04",
    "purpose": "zero-training endpoint-noise and phase-design identifiability audit",
    "data_use": {
        "300m_panel": "expanded 520-row development panel",
        "proportional_seed_repeats": str(PROPORTIONAL_REFERENCE_PATH.relative_to(REPO_ROOT)),
        "trajectory": str(TRAJECTORY_PATH.relative_to(REPO_ROOT)),
        "baseline": str(BASELINE_PATH.relative_to(REPO_ROOT)),
        "sealed_outcomes_used": False,
    },
    "endpoint_repeat_rule": {
        "coordinate_round_decimals": ROUND_DECIMALS,
        "proportional_reference_alias_is_independent": False,
        "proportional_seed_panel_scope": (
            "total run-level endpoint variance at one tied coordinate; initialization, data order, "
            "and simulated-epoch subset are not decomposed"
        ),
    },
    "trajectory_proxy": {
        "steps": TRAJECTORY_STEPS,
        "finite_difference": "y22000 - 3*y21000 + 3*y20000 - y19000",
        "independent_noise_variance_divisor": 20,
        "interpretation": "sensitivity proxy contaminated by curvature and temporal correlation",
    },
    "semivariogram_proxy": {
        "distance": "0.5*(beta0*L1(w0-w0') + beta1*L1(w1-w1'))",
        "neighbor_counts": NEIGHBOR_COUNTS,
        "powers": SEMIVARIOGRAM_POWERS,
        "fits": ("unconstrained_ols", "nonnegative_intercept_and_slope"),
        "interpretation": "local-stationarity sensitivity proxy, not a seed-variance bound",
    },
    "phase_design": {
        "contrast": "delta = w1 - w0 projected to the simplex tangent space",
        "operator": "delta^T B h(wbar)",
        "aggregate_bases": ("constant", "declared_bucket_family_masses", "full_linear_tangent"),
        "canonical_row_permutation_draws": CANONICAL_NULL_DRAWS,
        "canonical_row_permutation_seed": CANONICAL_NULL_SEED,
        "rank_restricted_operator_assessed": False,
        "scalar_output_rank_shortcut_forbidden": True,
    },
    "decision_rule": (
        "No model is promoted. Endpoint noise is identified only by independent same-coordinate seeds; "
        "an unrestricted aggregate-conditioned phase field is identified only if its bilinear design is injective."
    ),
}


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_keys(weights: np.ndarray) -> np.ndarray:
    return np.asarray([hashlib.sha256(np.round(row, ROUND_DECIMALS).tobytes()).hexdigest()[:16] for row in weights])


def exact_coordinate_repeats() -> pd.DataFrame:
    proportional = pd.read_csv(PROPORTIONAL_REFERENCE_PATH)
    reference_means = {target: float(proportional[TARGET_COLUMNS[target]].mean()) for target in TARGETS}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        dataset = benchmark.load_300m(target)
        frame = dataset.frame.copy()
        frame["coordinate_key"] = policy_keys(dataset.weights.reshape(dataset.n, -1))
        for coordinate_key, group in frame.groupby("coordinate_key", sort=True):
            if len(group) < 2:
                continue
            for left_index, right_index in combinations(group.index.tolist(), 2):
                left = frame.loc[left_index]
                right = frame.loc[right_index]
                y_left = float(dataset.y[left_index])
                y_right = float(dataset.y[right_index])
                left_is_reference_mean = bool(np.isclose(y_left, reference_means[target], rtol=0.0, atol=1e-12))
                right_is_reference_mean = bool(np.isclose(y_right, reference_means[target], rtol=0.0, atol=1e-12))
                contains_reference_mean_alias = left_is_reference_mean or right_is_reference_mean
                rows.append(
                    {
                        "target": target,
                        "coordinate_key": coordinate_key,
                        "run_left": left["run_name"],
                        "run_right": right["run_name"],
                        "policy_family_left": left["policy_family"],
                        "policy_family_right": right["policy_family"],
                        "correspondence_key": left["phase_correspondence_key"],
                        "value_left": y_left,
                        "value_right": y_right,
                        "signed_difference_right_minus_left": y_right - y_left,
                        "absolute_difference": abs(y_right - y_left),
                        "left_is_proportional_reference_mean": left_is_reference_mean,
                        "right_is_proportional_reference_mean": right_is_reference_mean,
                        "contains_proportional_reference_mean_alias": contains_reference_mean_alias,
                        "independent_physical_pair": not contains_reference_mean_alias,
                        "tied_neutrality_check": not contains_reference_mean_alias,
                        "wandb_left": left["training_wandb_id"],
                        "wandb_right": right["training_wandb_id"],
                    }
                )
    return pd.DataFrame(rows)


def proportional_run_variance(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(PROPORTIONAL_REFERENCE_PATH)
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        values = runs[TARGET_COLUMNS[target]].to_numpy(dtype=float)
        degrees_of_freedom = len(values) - 1
        standard_deviation = float(np.std(values, ddof=1))
        variance_numerator = degrees_of_freedom * standard_deviation**2
        lower = float(np.sqrt(variance_numerator / chi2.ppf(0.975, degrees_of_freedom)))
        upper = float(np.sqrt(variance_numerator / chi2.ppf(0.025, degrees_of_freedom)))
        upper_one_sided = float(np.sqrt(variance_numerator / chi2.ppf(0.05, degrees_of_freedom)))
        hpr_rmse = float(baseline.loc[baseline["target"].eq(target), "all_rmse"].iloc[0])
        rows.append(
            {
                "target": target,
                "n": len(values),
                "degrees_of_freedom": degrees_of_freedom,
                "mean": float(np.mean(values)),
                "run_level_endpoint_sd": standard_deviation,
                "sd_ci95_lower": lower,
                "sd_ci95_upper": upper,
                "sd_ucl95": upper_one_sided,
                "hpr_all_rmse": hpr_rmse,
                "hpr_all_rmse_in_run_sd": hpr_rmse / standard_deviation,
                "five_percent_hpr_rmse_in_run_sd": 0.05 * hpr_rmse / standard_deviation,
            }
        )
    return runs, pd.DataFrame(rows)


def trajectory_jitter() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(TRAJECTORY_PATH)
    metric = "eval/uncheatable_eval/bpb"
    keys = ["scale_key", "pair_id", "policy_class", "wandb_run_id", "wandb_run_name", "wandb_data_seed"]
    selected = source.loc[source["global_step"].isin(TRAJECTORY_STEPS), [*keys, "global_step", metric]]
    pivot = selected.pivot_table(index=keys, columns="global_step", values=metric, aggfunc="first").reset_index()
    complete = pivot.dropna(subset=list(TRAJECTORY_STEPS)).copy()
    complete["third_difference"] = complete[22000] - 3.0 * complete[21000] + 3.0 * complete[20000] - complete[19000]
    complete["independent_noise_equivalent_abs"] = complete["third_difference"].abs() / np.sqrt(20.0)

    summaries: list[dict[str, Any]] = []
    grouping: list[tuple[str, pd.DataFrame]] = [("all", complete)]
    grouping.extend(
        (f"{scale}:{policy}", group) for (scale, policy), group in complete.groupby(["scale_key", "policy_class"])
    )
    normal_median_abs = 0.6744897501960817
    for label, group in grouping:
        values = group["third_difference"].to_numpy(dtype=float)
        median = float(np.median(values))
        summaries.append(
            {
                "target": "uncheatable",
                "group": label,
                "n": len(values),
                "window_start_step": TRAJECTORY_STEPS[0],
                "window_end_step": TRAJECTORY_STEPS[-1],
                "mean_bpb_change_over_window": float(np.mean(group[22000] - group[19000])),
                "third_difference_mean": float(np.mean(values)),
                "third_difference_median": median,
                "independent_noise_equivalent_rms": float(np.sqrt(np.mean(values**2) / 20.0)),
                "independent_noise_equivalent_centered_sd": float(np.std(values, ddof=1) / np.sqrt(20.0)),
                "independent_noise_equivalent_centered_mad": float(
                    np.median(np.abs(values - median)) / (normal_median_abs * np.sqrt(20.0))
                ),
            }
        )
    return complete, pd.DataFrame(summaries)


def unique_coordinate_panel(target: str) -> tuple[np.ndarray, np.ndarray, float]:
    dataset = benchmark.load_300m(target)
    proportional = pd.read_csv(PROPORTIONAL_REFERENCE_PATH)
    reference_mean = float(proportional[TARGET_COLUMNS[target]].mean())
    beta0 = benchmark.geometry_300m(dataset).phase_0_fraction
    keys = policy_keys(dataset.weights.reshape(dataset.n, -1))
    unique_weights: list[np.ndarray] = []
    unique_targets: list[float] = []
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        unique_weights.append(dataset.weights[indices[0]])
        values = dataset.y[indices]
        has_reference_mean = np.any(np.isclose(values, reference_mean, rtol=0.0, atol=1e-12))
        unique_targets.append(reference_mean if has_reference_mean else float(np.mean(values)))
    return np.stack(unique_weights), np.asarray(unique_targets), beta0


def fit_semivariogram(distance: np.ndarray, gamma: np.ndarray, power: int) -> dict[str, float]:
    design = np.column_stack([np.ones(len(distance)), distance**power])
    ols, *_ = np.linalg.lstsq(design, gamma, rcond=None)
    positive, _ = nnls(design, gamma)
    ols_residual = gamma - design @ ols
    positive_residual = gamma - design @ positive
    return {
        "ols_intercept": float(ols[0]),
        "ols_slope": float(ols[1]),
        "ols_sigma_proxy": float(np.sqrt(max(ols[0], 0.0))),
        "ols_rmse": float(np.sqrt(np.mean(ols_residual**2))),
        "nonnegative_intercept": float(positive[0]),
        "nonnegative_slope": float(positive[1]),
        "nonnegative_sigma_proxy": float(np.sqrt(positive[0])),
        "nonnegative_rmse": float(np.sqrt(np.mean(positive_residual**2))),
    }


def semivariogram_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        weights, values, beta0 = unique_coordinate_panel(target)
        phase0 = weights[:, 0]
        phase1 = weights[:, 1]
        distance = 0.5 * (
            beta0 * np.abs(phase0[:, None, :] - phase0[None, :, :]).sum(axis=2)
            + (1.0 - beta0) * np.abs(phase1[:, None, :] - phase1[None, :, :]).sum(axis=2)
        )
        np.fill_diagonal(distance, np.inf)
        for neighbors in NEIGHBOR_COUNTS:
            pair_set: set[tuple[int, int]] = set()
            nearest = np.argpartition(distance, kth=neighbors - 1, axis=1)[:, :neighbors]
            for left, right_indices in enumerate(nearest):
                pair_set.update((min(left, int(right)), max(left, int(right))) for right in right_indices)
            pairs = np.asarray(sorted(pair_set), dtype=int)
            local_distance = distance[pairs[:, 0], pairs[:, 1]]
            gamma = 0.5 * (values[pairs[:, 0]] - values[pairs[:, 1]]) ** 2
            for power in SEMIVARIOGRAM_POWERS:
                rows.append(
                    {
                        "target": target,
                        "neighbors_per_coordinate": neighbors,
                        "distance_power": power,
                        "pair_count": len(pairs),
                        "distance_min": float(np.min(local_distance)),
                        "distance_median": float(np.median(local_distance)),
                        "distance_max": float(np.max(local_distance)),
                        **fit_semivariogram(local_distance, gamma, power),
                    }
                )
    return pd.DataFrame(rows)


def standardized_rank(matrix: np.ndarray) -> dict[str, float | int]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=1)
    retained = scale > 1e-12
    standardized = centered[:, retained] / scale[retained]
    singular = np.linalg.svd(standardized, compute_uv=False)
    threshold = max(standardized.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > threshold))
    return {
        "raw_parameter_count": matrix.shape[1],
        "nonconstant_parameter_count": int(retained.sum()),
        "matrix_rank": rank,
        "nullity_after_constant_removal": int(retained.sum() - rank),
        "condition_number": float(singular[0] / singular[rank - 1]),
        "stable_rank": float(np.sum(singular**2) / singular[0] ** 2),
    }


def bilinear_features(contrast: np.ndarray, aggregate_basis: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", contrast, aggregate_basis).reshape(len(contrast), -1)


def canonical_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_u, left_s, _ = np.linalg.svd(left_centered, full_matrices=False)
    right_u, right_s, _ = np.linalg.svd(right_centered, full_matrices=False)
    left_rank = int(np.sum(left_s > left_s[0] * 1e-10))
    right_rank = int(np.sum(right_s > right_s[0] * 1e-10))
    return np.linalg.svd(left_u[:, :left_rank].T @ right_u[:, :right_rank], compute_uv=False)


def phase_design_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = benchmark.load_300m("uncheatable")
    beta0 = benchmark.geometry_300m(dataset).phase_0_fraction
    asymmetric = np.max(np.abs(dataset.weights[:, 0] - dataset.weights[:, 1]), axis=1) > 1e-10
    weights = dataset.weights[asymmetric]
    aggregate = beta0 * weights[:, 0] + (1.0 - beta0) * weights[:, 1]
    contrast = weights[:, 1] - weights[:, 0]
    tangent = null_space(np.ones((1, aggregate.shape[1])))
    aggregate_tangent = (aggregate - aggregate.mean(axis=0, keepdims=True)) @ tangent
    contrast_tangent = contrast @ tangent

    family_count = int(dataset.family_index.max()) + 1
    family_mass = np.column_stack(
        [aggregate[:, dataset.family_index == family].sum(axis=1) for family in range(family_count)]
    )
    family_mass_keys = policy_keys(family_mass)
    family_mass_l1 = np.abs(family_mass[:, None, :] - family_mass[None, :, :]).sum(axis=2)
    np.fill_diagonal(family_mass_l1, np.inf)
    family_mass_upper = family_mass_l1[np.triu_indices(len(family_mass_l1), k=1)]
    independent_family_mass = family_mass[:, :-1] - family_mass[:, :-1].mean(axis=0, keepdims=True)
    family_u, family_s, _ = np.linalg.svd(independent_family_mass, full_matrices=False)
    family_rank = int(np.sum(family_s > family_s[0] * 1e-10))
    bases = {
        "constant": np.ones((len(aggregate), 1)),
        "declared_bucket_family_masses": np.column_stack([np.ones(len(aggregate)), family_u[:, :family_rank]]),
        "full_linear_tangent": np.column_stack([np.ones(len(aggregate)), aggregate_tangent]),
    }
    rows: list[dict[str, Any]] = []
    for name, basis in bases.items():
        features = bilinear_features(contrast_tangent, basis)
        rows.append(
            {
                "aggregate_basis": name,
                "observations": len(features),
                "contrast_dimension": contrast_tangent.shape[1],
                "aggregate_basis_dimension": basis.shape[1],
                **standardized_rank(features),
            }
        )

    left_rank = int(np.linalg.matrix_rank(aggregate_tangent))
    right_rank = int(np.linalg.matrix_rank(contrast_tangent))
    canonical = canonical_correlations(aggregate_tangent, contrast_tangent)
    observed_canonical_energy = float(np.sum(canonical**2))
    rng = np.random.default_rng(CANONICAL_NULL_SEED)
    null_energy = np.asarray(
        [
            np.sum(
                canonical_correlations(aggregate_tangent, contrast_tangent[rng.permutation(len(contrast_tangent))]) ** 2
            )
            for _ in range(CANONICAL_NULL_DRAWS)
        ]
    )
    aggregate_keys = policy_keys(aggregate)
    direction_norm = np.linalg.norm(contrast_tangent, axis=1, keepdims=True)
    direction_keys = policy_keys(contrast_tangent / direction_norm)
    metadata = {
        "asymmetric_rows": int(asymmetric.sum()),
        "unique_asymmetric_aggregates": len(np.unique(aggregate_keys)),
        "maximum_rows_at_one_aggregate": int(pd.Series(aggregate_keys).value_counts().max()),
        "unique_declared_family_mass_aggregates": len(np.unique(family_mass_keys)),
        "maximum_rows_at_one_declared_family_mass_aggregate": int(pd.Series(family_mass_keys).value_counts().max()),
        "declared_family_mass_nearest_neighbor_l1_median": float(np.median(np.min(family_mass_l1, axis=1))),
        "declared_family_mass_pair_count_l1_below_0p01": int(np.sum(family_mass_upper < 0.01)),
        "unique_normalized_contrast_directions": len(np.unique(direction_keys)),
        "aggregate_tangent_rank": left_rank,
        "contrast_tangent_rank": right_rank,
        "canonical_correlation_min": float(np.min(canonical)),
        "canonical_correlation_median": float(np.median(canonical)),
        "canonical_correlation_max": float(np.max(canonical)),
        "canonical_correlation_energy": observed_canonical_energy,
        "canonical_null_draws": CANONICAL_NULL_DRAWS,
        "canonical_null_energy_median": float(np.median(null_energy)),
        "canonical_null_energy_q975": float(np.quantile(null_energy, 0.975)),
        "canonical_null_energy_p_one_sided": float(
            (1 + np.sum(null_energy >= observed_canonical_energy)) / (CANONICAL_NULL_DRAWS + 1)
        ),
        "canonical_null_exceedances": int(np.sum(null_energy >= observed_canonical_energy)),
        "full_linear_rank_one_nominal_dof": int(contrast_tangent.shape[1] + bases["full_linear_tangent"].shape[1] - 1),
        "full_linear_rank_two_nominal_dof": int(
            2 * (contrast_tangent.shape[1] + bases["full_linear_tangent"].shape[1] - 2)
        ),
        "full_linear_rank_three_nominal_dof": int(
            3 * (contrast_tangent.shape[1] + bases["full_linear_tangent"].shape[1] - 3)
        ),
    }
    return pd.DataFrame(rows), metadata


def hpr_metrics() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_PATH)
    columns = ["target", "all_rmse", "asymmetric_rmse", "tied_rmse"]
    return frame.loc[frame["model"].eq("hierarchical_phase_replay"), columns].reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(
    protocol_hash: str,
    repeats: pd.DataFrame,
    proportional_summary: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    semivariogram: pd.DataFrame,
    design: pd.DataFrame,
    design_metadata: dict[str, Any],
) -> None:
    tied_neutrality = repeats.loc[repeats["tied_neutrality_check"]]
    repeat_summary = tied_neutrality[["target", "run_left", "run_right", "signed_difference_right_minus_left"]].round(6)
    semivariogram_range = (
        semivariogram.groupby("target")
        .agg(
            ols_intercept_min=("ols_intercept", "min"),
            ols_intercept_max=("ols_intercept", "max"),
            nonnegative_sigma_proxy_max=("nonnegative_sigma_proxy", "max"),
        )
        .reset_index()
    )
    uncheatable_run_sd = float(
        proportional_summary.loc[proportional_summary["target"].eq("uncheatable"), "run_level_endpoint_sd"].iloc[0]
    )
    all_trajectory_proxy = float(
        trajectory_summary.loc[trajectory_summary["group"].eq("all"), "independent_noise_equivalent_rms"].iloc[0]
    )
    proportional_table = markdown_table(
        proportional_summary.round(6),
        [
            "target",
            "n",
            "run_level_endpoint_sd",
            "sd_ci95_lower",
            "sd_ci95_upper",
            "sd_ucl95",
            "hpr_all_rmse_in_run_sd",
            "five_percent_hpr_rmse_in_run_sd",
        ],
    )
    trajectory_table = markdown_table(
        trajectory_summary.round(6),
        [
            "target",
            "group",
            "n",
            "mean_bpb_change_over_window",
            "third_difference_mean",
            "independent_noise_equivalent_rms",
        ],
    )
    design_table = markdown_table(
        design.round(3),
        [
            "aggregate_basis",
            "observations",
            "raw_parameter_count",
            "matrix_rank",
            "nullity_after_constant_removal",
            "condition_number",
            "stable_rank",
        ],
    )
    report = f"""# 300M endpoint-noise and phase-design identifiability audit

Protocol: `{protocol_hash}`

Registry candidate: `WSD80-SUR-079`

## Decision

**Total run-level endpoint variance is identified at the tied proportional policy, but not as a function
of policy. The unrestricted aggregate-conditioned phase field is not identified. No surrogate term is
promoted.**

The purpose-built proportional panel has 11 independent runs and 10 degrees of freedom per target. Ten
runs sweep `trainer_seed`; because `data_seed` and `simulated_epoch_subset_seed` are unset, that seed also
changes data order and simulated-epoch subset membership. The measured SD therefore combines
initialization, data-order, and subset variation at one tied coordinate; it does not decompose those sources
or identify variance across policy space.

{proportional_table}

HPR's grouped-OOF RMSE is 6.03 proportional run-level SD on Uncheatable and 3.90 SD on Table-9.
Five percent of those RMSEs is 0.30 and 0.20 run-level SD, respectively. The observed RMSE is therefore not
at this measured run-level floor. Those single-run ratios do not quantify uncertainty in a paired panel-level
RMSE difference; compare models with paired or bootstrap uncertainty rather than a point estimate.

The expanded 520-row fitting panel contains 518 distinct policy coordinates. Proportional and UniMax each
appear twice as tied policies. The proportional rows are not seed replicates: one or both target values are
the 11-run reference mean. UniMax is one physical cross-pipeline tied-neutrality comparison, not a variance
sample. Its two-phase-minus-one-phase differences are:

{markdown_table(repeat_summary, ['target', 'run_left', 'run_right', 'signed_difference_right_minus_left'])}

These differences are -0.13 proportional run-level SD on Uncheatable and -1.00 SD on Table-9, consistent
with a tied-neutrality check at the observed run-to-run scale.

The phase design has `{design_metadata['asymmetric_rows']}` asymmetric rows at
`{design_metadata['unique_asymmetric_aggregates']}` distinct aggregates, with at most
`{design_metadata['maximum_rows_at_one_aggregate']}` contrast per aggregate. Therefore the design cannot
nonparametrically separate aggregate conditioning from phase direction. A constant or predeclared
family-conditioned operator can be numerically fit, but that restriction is a model assumption, not a
mechanism identified by repeated directions.

## Temporal-jitter sensitivity

This diagnostic is Uncheatable-only and uses steps 19,000--22,000 of the 300M WSD decay, not the terminal
step 22,887. Mean BPB changes by about 0.098 over the window. For independent endpoint noise, the
third-difference coefficient norm would be `sqrt(20)`, but the observed third difference also contains
smooth cubic training dynamics and temporally correlated evaluation noise.

{trajectory_table}

The pooled RMS proxy is {all_trajectory_proxy / uncheatable_run_sd:.2f} times the directly measured
proportional Uncheatable run-level SD. This agreement is descriptive; it does not turn a trajectory
finite difference into a run-level variance estimator.

## Semivariogram sensitivity

The semivariogram intercept is not an endpoint-noise estimate here. The smallest-distance region is a
designed proportional domain-deletion substructure, not a generic local sample; the response model is
extrapolated from distances as large as roughly 0.47 back to zero. Linear-distance fits yield negative
intercepts in the closest neighborhood, while nonnegative fitting clamps them to zero. Quadratic-distance
fits produce much larger positive intercepts. The sensitivity range is retained only to demonstrate this
instability:

{markdown_table(
    semivariogram_range.round(6),
    ['target', 'ols_intercept_min', 'ols_intercept_max', 'nonnegative_sigma_proxy_max'],
)}

No value in this table is used as a noise floor or model-acceptance threshold.

## Phase-design rank

{design_table}

The table is descriptive. The full linear operator has more parameters than observations, so its rank
deficiency is primarily parameter counting rather than a discovered geometric obstruction. The
load-bearing design fact is one contrast per aggregate.

At the declared family-mass resolution all 238 vectors are distinct at 12 decimals, but exact identity is
not an informative overlap criterion in this continuous two-dimensional basis. The median nearest-neighbor
L1 distance is `{design_metadata['declared_family_mass_nearest_neighbor_l1_median']:.4f}`, and
`{design_metadata['declared_family_mass_pair_count_l1_below_0p01']}` unordered pairs are within L1 distance
0.01. This is a dense cloud, not the full 39-dimensional non-repetition obstruction; the family-conditioned
operator remains numerically fittable under an imposed family basis.

The aggregate and contrast sample subspaces have squared-canonical-correlation energy
`{design_metadata['canonical_correlation_energy']:.3f}` versus a row-permutation null median
`{design_metadata['canonical_null_energy_median']:.3f}` and 97.5% quantile
`{design_metadata['canonical_null_energy_q975']:.3f}`
(`p<={design_metadata['canonical_null_energy_p_one_sided']:.4f}`, the 400-draw resolution limit).
They therefore vary together more than random row pairing would imply. Simplex feasibility contributes to
this coupling; the result does not identify causality or the correct aggregate basis.

A rank-one full-linear bilinear operator would have nominally
`{design_metadata['full_linear_rank_one_nominal_dof']}` degrees of freedom, but rank-restricted operator
injectivity and recovery were **not assessed**. The audit therefore rejects only an unrestricted phase
field; it does not claim that every preregistered low-rank operator is unidentified.

## Additional data requirements

Do not repeat the proportional policy again merely to estimate its total run-level endpoint variance.
Before adding noise-calibration runs, specify whether the estimand is total endpoint variance at another
policy, a decomposition of initialization/data-order/subset contributions, or the variance of a same-seed
policy contrast. Independent runs across policies support pooled total-variance estimation; shared seed
identities support paired contrasts but require a crossed analysis with only four seed-main-effect degrees
of freedom for five seeds. No run count is frozen by this audit.

For phase-field identification, endpoint repeats are not enough. The missing intervention is multiple
linearly independent contrast directions at the same preregistered aggregates, preferably with antithetic
signs and shared seeds. Regularization cannot substitute for those directions.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "protocol_hash": payload_hash(PROTOCOL)}
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    baseline = hpr_metrics()
    repeats = exact_coordinate_repeats()
    proportional_runs, proportional_summary = proportional_run_variance(baseline)
    trajectory, trajectory_summary = trajectory_jitter()
    semivariogram = semivariogram_sensitivity()
    design, design_metadata = phase_design_audit()

    repeats.to_csv(OUTPUT_DIR / "exact_coordinate_repeats.csv", index=False)
    proportional_runs.to_csv(OUTPUT_DIR / "proportional_run_level_observations.csv", index=False)
    proportional_summary.to_csv(OUTPUT_DIR / "proportional_run_level_variance.csv", index=False)
    trajectory.to_csv(OUTPUT_DIR / "trajectory_jitter.csv", index=False)
    trajectory_summary.to_csv(OUTPUT_DIR / "trajectory_jitter_summary.csv", index=False)
    semivariogram.to_csv(OUTPUT_DIR / "semivariogram_sensitivity.csv", index=False)
    design.to_csv(OUTPUT_DIR / "phase_design_rank.csv", index=False)
    (OUTPUT_DIR / "phase_design_metadata.json").write_text(json.dumps(design_metadata, indent=2, sort_keys=True) + "\n")
    decision = {
        "candidate_id": "WSD80-SUR-079",
        "decision": "partial_identification_no_model_promoted",
        "total_run_level_endpoint_variance_at_proportional_identified": True,
        "run_level_variance_components_decomposed": False,
        "endpoint_variance_as_function_of_policy_identified": False,
        "unrestricted_phase_field_identified": False,
        "rank_restricted_phase_operator_assessed": False,
        "protocol_hash": protocol["protocol_hash"],
        "physical_tied_neutrality_checks": (
            repeats.groupby("target")["tied_neutrality_check"].sum().astype(int).to_dict()
        ),
        "recommended_noise_calibration_runs": None,
    }
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_report(
        protocol["protocol_hash"],
        repeats,
        proportional_summary,
        trajectory_summary,
        semivariogram,
        design,
        design_metadata,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
