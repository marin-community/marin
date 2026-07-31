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
"""Develop and audit a conditional aggregate/phase-order factorization.

The model under test is

    L_hat(a, d) = A(a) + O(a, d) + C(a, d),   O odd in d,  C even in d,

fitted on the *paired* contrast ``Delta = L(a, d) - L(a, 0)`` so that aggregate
model error cannot masquerade as phase signal. Two exactly aggregate-matched
panels supply 238 paired coordinates each; a balanced antithetic panel observes
``O`` and ``C`` separately at two anchors and is used only as an identification
test.

The sealed targeted-pairwise panel is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from phase_order_candidates_20260725 import (
    EVEN_FUNCTIONALS,
    Geometry,
    aggregate_region_blocks,
    fit_blocks,
    grouped_folds,
    odd_effective_exposure,
    odd_family_pooled,
    odd_free_bucket,
    odd_marginal_value,
    odd_retention_exchange,
    quantile_blocks,
    shared_curvature_blocks,
)
from phase_order_spine_20260725 import (
    REFERENCE_OUTPUTS,
    TARGETS,
    build_spine,
    family_index_for,
    load_exposure_spec,
    provenance,
)
from plotly.subplots import make_subplots
from scipy.optimize import least_squares, linprog
from scipy.stats import pearsonr, spearmanr

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "conditional_phase_order_factorization_20260725"

# Frozen run-noise standard deviations from the prior independent repeat panels.
RUN_SIGMA = {"uncheatable_bpb": 0.000963, "table9_macro_bpb": 0.003121}
TARGET_LABEL = {"uncheatable_bpb": "uncheatable", "table9_macro_bpb": "table9"}
GOAL_GAIN_BPB = 0.01
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260725
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def odd_noise(target: str) -> float:
    return RUN_SIGMA[target] / np.sqrt(2.0)


def even_noise(target: str) -> float:
    return RUN_SIGMA[target] * np.sqrt(1.5)


def geometry_for(alpha: float, aggregate: np.ndarray, contrast: np.ndarray, spec, family_ids, n_families) -> Geometry:
    """Build a geometry. ``spec.c0``/``spec.c1`` are already in simulated epochs."""
    return Geometry(
        alpha=alpha,
        aggregate=aggregate,
        contrast=contrast,
        c0=spec.c0,
        c1=spec.c1,
        family_index=family_ids,
        family_count=n_families,
    )


# ---------------------------------------------------------------------------
# 1. Even-cost structure: is there any resolvable direction dependence?
# ---------------------------------------------------------------------------


def even_direction_structure(antithetic) -> pd.DataFrame:
    """Across-direction spread of O and C against the paired-noise floor."""
    rows = []
    for target in TARGETS:
        for channel, values, noise in (
            ("odd", antithetic.odd[target], odd_noise(target)),
            ("even", antithetic.even[target], even_noise(target)),
        ):
            for anchor in sorted(set(antithetic.anchor_id.tolist())):
                for tv in sorted(set(antithetic.target_phase_tv.tolist())):
                    mask = (antithetic.anchor_id == anchor) & (antithetic.target_phase_tv == tv)
                    sample = values[mask]
                    spread = float(sample.std(ddof=1))
                    latent = float(np.sqrt(max(spread**2 - noise**2, 0.0)))
                    rows.append(
                        {
                            "target": TARGET_LABEL[target],
                            "channel": channel,
                            "anchor": anchor,
                            "target_phase_tv": tv,
                            "n_directions": int(mask.sum()),
                            "mean": float(sample.mean()),
                            "across_direction_sd": spread,
                            "paired_noise_sd": noise,
                            "latent_direction_sd": latent,
                            "latent_direction_snr": latent / noise,
                        }
                    )
    return pd.DataFrame(rows)


def even_law_comparison(antithetic, spec, family_ids, n_families, alpha: float) -> pd.DataFrame:
    """Single-amplitude even functionals under grouped hold-out."""
    geometry = geometry_for(alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families)
    rows = []
    for name, functional in EVEN_FUNCTIONALS.items():
        feature = functional(geometry)
        for target in TARGETS:
            observed = antithetic.even[target]
            scale = float((feature @ observed) / (feature @ feature))
            pooled = 1.0 - float(((observed - scale * feature) ** 2).sum() / ((observed - observed.mean()) ** 2).sum())
            record: dict[str, Any] = {
                "even_functional": name,
                "target": TARGET_LABEL[target],
                "amplitude": scale,
                "pooled_r2": pooled,
            }
            for label, groups in (
                ("leave_radius", antithetic.target_phase_tv),
                ("leave_anchor", antithetic.anchor_id),
            ):
                scores = []
                for train, test in grouped_folds(groups):
                    gram = float(feature[train] @ feature[train])
                    coef = float((feature[train] @ observed[train]) / gram) if gram > 0 else 0.0
                    residual = observed[test] - coef * feature[test]
                    denominator = ((observed[test] - observed[test].mean()) ** 2).sum()
                    scores.append(1.0 - float((residual**2).sum() / denominator))
                record[f"{label}_r2_min"] = float(np.min(scores))
                record[f"{label}_r2_mean"] = float(np.mean(scores))
            rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Odd-field transfer: direction versus amplitude
# ---------------------------------------------------------------------------


def odd_direction_transfer(antithetic) -> pd.DataFrame:
    """Per-(anchor, direction) odd slope in radius, and its cross-anchor transfer."""
    directions = sorted(set(antithetic.direction_id.tolist()))
    anchors = sorted(set(antithetic.anchor_id.tolist()))
    rows = []
    for target in TARGETS:
        slopes = {}
        for anchor in anchors:
            values = []
            for direction in directions:
                mask = (antithetic.anchor_id == anchor) & (antithetic.direction_id == direction)
                radius = antithetic.realized_phase_tv[mask]
                response = antithetic.odd[target][mask]
                values.append(float((radius @ response) / (radius @ radius)))
            slopes[anchor] = np.asarray(values)
        first, second = slopes[anchors[0]], slopes[anchors[1]]
        pearson, p_value = pearsonr(first, second)
        spear, _ = spearmanr(first, second)
        rows.append(
            {
                "target": TARGET_LABEL[target],
                "n_directions": len(directions),
                "slope_sd_anchor_a": float(first.std(ddof=1)),
                "slope_sd_anchor_b": float(second.std(ddof=1)),
                "cross_anchor_pearson": float(pearson),
                "cross_anchor_pearson_pvalue": float(p_value),
                "cross_anchor_spearman": float(spear),
                "cross_anchor_sign_agreement": float(np.mean(np.sign(first) == np.sign(second))),
            }
        )
    return pd.DataFrame(rows)


AGGREGATE_REGION_BLOCKS = 8
RADIUS_BLOCKS = 4


def panel_groups(panel) -> dict[str, np.ndarray]:
    """Deterministic grouped-fold labels for a paired panel.

    ``candidate_kind`` collapses to a single value once exactly tied rows are
    dropped, so the paired panels are grouped by aggregate region and by contrast
    radius block instead. Both are leave-whole-group-out, never random-row.
    """
    return {
        "aggregate_region": aggregate_region_blocks(panel.aggregate, AGGREGATE_REGION_BLOCKS),
        "radius_block": quantile_blocks(panel.phase_tv, RADIUS_BLOCKS),
    }


def _select_l2(odd_design, even_design, target_values, folds) -> float:
    """Choose the odd-block ridge by grouped hold-out error."""
    best, best_score = L2_GRID[0], np.inf
    for l2 in L2_GRID:
        errors = []
        for train, test in folds:
            fit = fit_blocks(odd_design[train], even_design[train], target_values[train], l2)
            errors.append(((fit.predict(odd_design[test], even_design[test]) - target_values[test]) ** 2).mean())
        score = float(np.mean(errors))
        if score < best_score:
            best, best_score = l2, score
    return best


def odd_amplitude_calibration(spine, spec, family_ids, n_families) -> pd.DataFrame:
    """Fit the odd field on paired Delta, then test its amplitude on the antithetic panel.

    The antithetic panel observes ``O`` directly, so the regression slope of
    observed on predicted odd is a direct amplitude calibration: 1.0 is correct.
    """
    antithetic = spine.antithetic
    rows = []
    for source_name, panel in (("delphi_3e18", spine.delphi_3e18), ("300m", spine.m300)):
        geometry = geometry_for(panel.alpha, panel.aggregate, panel.contrast, spec, family_ids, n_families)
        anti_geometry = geometry_for(
            spine.delphi_3e18.alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families
        )
        for even_name in ("tv_squared", "fisher_chi2", "kl_jensen", "boundary_overload"):
            # The even block varies only to show that odd amplitude inflation is
            # not even-model misspecification; only the odd block is transferred.
            even_design = np.column_stack(
                [EVEN_FUNCTIONALS[even_name](geometry), EVEN_FUNCTIONALS["tv_cubed"](geometry)]
            )
            for odd_name, odd_builder in (
                ("free_bucket", odd_free_bucket),
                ("family_pooled", odd_family_pooled),
                ("effective_exposure_dsp", odd_effective_exposure),
                ("marginal_value_pmvt", odd_marginal_value(1.0)),
                ("retention_exchange", odd_retention_exchange(1.0)),
            ):
                odd_design = odd_builder(geometry)
                anti_odd = odd_builder(anti_geometry)
                groups = panel_groups(panel)
                for target in TARGETS:
                    observed_delta = panel.delta[target]
                    folds = grouped_folds(groups["aggregate_region"])
                    l2 = _select_l2(odd_design, even_design, observed_delta, folds)
                    fit = fit_blocks(odd_design, even_design, observed_delta, l2)
                    fitted = fit.predict(odd_design, even_design)
                    predicted_odd = anti_odd @ fit.odd_coef
                    observed_odd = antithetic.odd[target]
                    if np.allclose(predicted_odd, 0.0):
                        continue
                    slope = float((predicted_odd @ observed_odd) / (predicted_odd @ predicted_odd))
                    pearson, p_value = pearsonr(predicted_odd, observed_odd)
                    rows.append(
                        {
                            "fit_panel": source_name,
                            "odd_field": odd_name,
                            "even_functional": even_name,
                            "target": TARGET_LABEL[target],
                            "selected_l2": l2,
                            "paired_delta_r2": (
                                1.0
                                - float(
                                    ((observed_delta - fitted) ** 2).sum()
                                    / ((observed_delta - observed_delta.mean()) ** 2).sum()
                                )
                            ),
                            "antithetic_odd_pearson": float(pearson),
                            "antithetic_odd_pearson_pvalue": float(p_value),
                            "amplitude_slope": slope,
                            "amplitude_inflation": float(1.0 / slope) if slope > 0 else np.nan,
                            "antithetic_odd_rmse_ratio_vs_zero": float(
                                np.sqrt(((observed_odd - predicted_odd) ** 2).mean()) / np.sqrt((observed_odd**2).mean())
                            ),
                            "antithetic_odd_rmse_ratio_rescaled": float(
                                np.sqrt(((observed_odd - slope * predicted_odd) ** 2).mean())
                                / np.sqrt((observed_odd**2).mean())
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def odd_field_reverse_test(spine, spec, family_ids, n_families) -> pd.DataFrame:
    """Fit the odd field on the directly observed antithetic O, then look at paired Delta.

    This is the amplitude-correct direction of the comparison: it measures how
    much of the paired contrast the true odd component can explain.
    """
    antithetic = spine.antithetic
    anti_geometry = geometry_for(
        spine.delphi_3e18.alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families
    )
    rows = []
    for target in TARGETS:
        design = odd_free_bucket(anti_geometry)
        observed = antithetic.odd[target]
        for l2 in (1e-3, 1e-2, 1e-1):
            coef = np.linalg.solve(design.T @ design + l2 * np.eye(design.shape[1]), design.T @ observed)
            in_sample = 1.0 - float(((observed - design @ coef) ** 2).sum() / ((observed - observed.mean()) ** 2).sum())
            leave_direction = []
            for train, test in grouped_folds(antithetic.direction_id):
                sub = np.linalg.solve(
                    design[train].T @ design[train] + l2 * np.eye(design.shape[1]), design[train].T @ observed[train]
                )
                leave_direction.append(((observed[test] - design[test] @ sub) ** 2).mean())
            held_out_rmse = float(np.sqrt(np.mean(leave_direction)))
            for panel_name, panel in (("delphi_3e18", spine.delphi_3e18), ("300m", spine.m300)):
                geometry = geometry_for(panel.alpha, panel.aggregate, panel.contrast, spec, family_ids, n_families)
                predicted = odd_free_bucket(geometry) @ coef
                delta = panel.delta[target]
                pearson, _ = pearsonr(predicted, delta)
                rows.append(
                    {
                        "target": TARGET_LABEL[target],
                        "l2": l2,
                        "antithetic_in_sample_r2": in_sample,
                        "antithetic_leave_direction_rmse": held_out_rmse,
                        "antithetic_zero_rmse": float(np.sqrt((observed**2).mean())),
                        "paired_panel": panel_name,
                        "predicted_sd_over_observed_sd": float(predicted.std(ddof=1) / delta.std(ddof=1)),
                        "paired_pearson": float(pearson),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Odd/even scaling budget and the implied maximum directional gain
# ---------------------------------------------------------------------------


def scaling_budget(antithetic, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint noise-weighted fit of O = theta_u rho^p and C = c_anchor rho^q.

    The measured exponents determine whether *any* direction can beat tied: with
    ``q > p`` the symmetric cost eventually dominates, and the best attainable
    gain is ``max_rho [kappa rho^p - c rho^q]``.
    """
    directions = sorted(set(antithetic.direction_id.tolist()))
    anchors = sorted(set(antithetic.anchor_id.tolist()))
    direction_index = np.asarray([directions.index(x) for x in antithetic.direction_id])
    anchor_index = np.asarray([anchors.index(x) for x in antithetic.anchor_id])
    radius = antithetic.realized_phase_tv
    n_directions = len(directions)

    def solve(target: str, odd_values: np.ndarray, even_values: np.ndarray) -> np.ndarray:
        sigma_odd, sigma_even = odd_noise(target), even_noise(target)

        def residual(z: np.ndarray) -> np.ndarray:
            theta = z[:n_directions]
            power_odd = z[n_directions]
            amplitude = z[n_directions + 1 : n_directions + 1 + len(anchors)]
            power_even = z[-1]
            odd_residual = (theta[direction_index] * radius**power_odd - odd_values) / sigma_odd
            even_residual = (amplitude[anchor_index] * radius**power_even - even_values) / sigma_even
            return np.concatenate([odd_residual, even_residual])

        start = np.concatenate([np.full(n_directions, 0.005), [1.5], np.full(len(anchors), 0.03), [2.0]])
        lower = np.concatenate([np.full(n_directions, -np.inf), [0.2], np.full(len(anchors), 0.0), [0.5]])
        upper = np.concatenate([np.full(n_directions, np.inf), [5.0], np.full(len(anchors), np.inf), [6.0]])
        return least_squares(residual, start, bounds=(lower, upper), method="trf").x

    fits, budget = [], []
    for target in TARGETS:
        z = solve(target, antithetic.odd[target], antithetic.even[target])
        theta = z[:n_directions]
        power_odd = float(z[n_directions])
        amplitudes = z[n_directions + 1 : n_directions + 1 + len(anchors)]
        power_even = float(z[-1])
        kappa = float(np.abs(theta).max())

        draws = []
        for _ in range(BOOTSTRAP_DRAWS // 10):
            pick = rng.integers(0, n_directions, n_directions)
            keep = np.isin(direction_index, pick)
            if keep.sum() < 20:
                continue
            noisy_odd = antithetic.odd[target] + rng.normal(0.0, odd_noise(target), len(radius))
            noisy_even = antithetic.even[target] + rng.normal(0.0, even_noise(target), len(radius))
            z_boot = solve(target, noisy_odd, noisy_even)
            draws.append(
                (
                    float(z_boot[n_directions]),
                    float(z_boot[-1]),
                    float(np.abs(z_boot[:n_directions]).max()),
                )
            )
        draws_array = np.asarray(draws) if draws else np.zeros((1, 3))

        fits.append(
            {
                "target": TARGET_LABEL[target],
                "odd_power_p": power_odd,
                "odd_power_p_ci_low": float(np.quantile(draws_array[:, 0], 0.025)),
                "odd_power_p_ci_high": float(np.quantile(draws_array[:, 0], 0.975)),
                "even_power_q": power_even,
                "even_power_q_ci_low": float(np.quantile(draws_array[:, 1], 0.025)),
                "even_power_q_ci_high": float(np.quantile(draws_array[:, 1], 0.975)),
                "best_direction_kappa": kappa,
                "best_direction_kappa_ci_low": float(np.quantile(draws_array[:, 2], 0.025)),
                "best_direction_kappa_ci_high": float(np.quantile(draws_array[:, 2], 0.975)),
                "exponent_gap_q_minus_p": power_even - power_odd,
                "cost_dominates_at_large_radius": bool(power_even > power_odd),
            }
        )

        for anchor, amplitude in zip(anchors, amplitudes, strict=True):
            record = {
                "target": TARGET_LABEL[target],
                "anchor": anchor,
                "odd_power_p": power_odd,
                "even_power_q": power_even,
                "even_amplitude_c": float(amplitude),
                "best_direction_kappa": kappa,
            }
            if power_even <= power_odd or amplitude <= 0:
                record.update(
                    {
                        "optimal_radius": np.nan,
                        "max_directional_gain_bpb": np.nan,
                        "kappa_multiplier_for_goal": np.nan,
                        "gain_elasticity_in_kappa": np.nan,
                    }
                )
            else:
                exponent = 1.0 / (power_even - power_odd)
                optimal_radius = float((kappa * power_odd / (amplitude * power_even)) ** exponent)
                gain = float(kappa * optimal_radius**power_odd - amplitude * optimal_radius**power_even)
                elasticity = power_even / (power_even - power_odd)
                record.update(
                    {
                        "optimal_radius": optimal_radius,
                        "max_directional_gain_bpb": gain,
                        "gain_elasticity_in_kappa": float(elasticity),
                        "kappa_multiplier_for_goal": (
                            float((GOAL_GAIN_BPB / gain) ** (1.0 / elasticity)) if gain > 0 else np.nan
                        ),
                    }
                )
            budget.append(record)
    return pd.DataFrame(fits), pd.DataFrame(budget)


def feasible_best_direction(antithetic, odd_coefficients: np.ndarray, radius: float) -> dict[str, float]:
    """Largest ``|g . d|`` reachable at a given phase TV under simplex feasibility.

    Solves the exact transport problem at the anchor: both phase mixtures must
    remain nonnegative, the contrast must be mass conserving, and the phase TV is
    bounded. Reported as an upper envelope on the attainable odd amplitude.
    """
    anchors = sorted(set(antithetic.anchor_id.tolist()))
    output = {}
    for anchor in anchors:
        mask = antithetic.anchor_id == anchor
        aggregate = antithetic.aggregate[mask][0]
        n = len(aggregate)
        # Variables: d+ and d- (nonnegative split of d). d = d+ - d-.
        cost = np.concatenate([-odd_coefficients, odd_coefficients])
        conservation = np.concatenate([np.ones(n), -np.ones(n)])
        total_variation = np.concatenate([np.ones(n), np.ones(n)])
        # p0 = a - (1-alpha) d >= 0 and p1 = a + alpha d >= 0.
        alpha = 0.7981376787495837
        upper_p0 = np.hstack([np.diag(np.full(n, 1.0 - alpha)), np.diag(np.full(n, -(1.0 - alpha)))])
        upper_p1 = np.hstack([np.diag(np.full(n, -alpha)), np.diag(np.full(n, alpha))])
        result = linprog(
            c=cost,
            A_ub=np.vstack([upper_p0, upper_p1, total_variation[None, :]]),
            b_ub=np.concatenate([aggregate, aggregate, [2.0 * radius]]),
            A_eq=conservation[None, :],
            b_eq=[0.0],
            bounds=[(0.0, None)] * (2 * n),
            method="highs",
        )
        if not result.success:
            output[anchor] = np.nan
            continue
        output[anchor] = float(-result.fun)
    return output


# ---------------------------------------------------------------------------
# 4. Declared candidates under grouped validation
# ---------------------------------------------------------------------------


def candidate_evaluation(spine, spec, family_ids, n_families) -> pd.DataFrame:
    """Evaluate every declared candidate against the zero-phase null."""
    antithetic = spine.antithetic
    anti_geometry = geometry_for(
        spine.delphi_3e18.alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families
    )
    rows = []

    def evaluate(name: str, odd_builder, even_names: tuple[str, ...], panel, panel_name: str) -> None:
        geometry = geometry_for(panel.alpha, panel.aggregate, panel.contrast, spec, family_ids, n_families)
        odd_design = odd_builder(geometry) if odd_builder is not None else np.zeros((len(panel.aggregate), 0))
        anti_odd = odd_builder(anti_geometry) if odd_builder is not None else np.zeros((len(antithetic.anchor_id), 0))
        even_design = np.column_stack([EVEN_FUNCTIONALS[n](geometry) for n in even_names])
        anti_even = np.column_stack([EVEN_FUNCTIONALS[n](anti_geometry) for n in even_names])
        groups = panel_groups(panel)
        for target in TARGETS:
            delta = panel.delta[target]
            folds = grouped_folds(groups["aggregate_region"])
            l2 = _select_l2(odd_design, even_design, delta, folds) if odd_design.size else 0.0
            fit = fit_blocks(odd_design, even_design, delta, l2)
            predicted_odd, predicted_even = fit.predict_parts(anti_odd, anti_even)
            observed_odd, observed_even = antithetic.odd[target], antithetic.even[target]
            zero_odd_rmse = float(np.sqrt((observed_odd**2).mean()))
            odd_rmse = float(np.sqrt(((observed_odd - predicted_odd) ** 2).mean()))
            record: dict[str, Any] = {
                "candidate": name,
                "fit_panel": panel_name,
                "target": TARGET_LABEL[target],
                "odd_parameters": int(odd_design.shape[1]),
                "even_parameters": int(even_design.shape[1]),
                "selected_l2": l2,
                "paired_delta_rmse": float(np.sqrt(((delta - fit.predict(odd_design, even_design)) ** 2).mean())),
                "antithetic_odd_rmse": odd_rmse,
                "antithetic_odd_rmse_ratio_vs_zero": odd_rmse / zero_odd_rmse,
                "antithetic_even_rmse": float(np.sqrt(((observed_even - predicted_even) ** 2).mean())),
                "antithetic_even_rmse_ratio_vs_zero": float(
                    np.sqrt(((observed_even - predicted_even) ** 2).mean()) / np.sqrt((observed_even**2).mean())
                ),
            }
            if odd_design.size and not np.allclose(predicted_odd, 0.0):
                record["antithetic_odd_sign_accuracy"] = float(np.mean(np.sign(predicted_odd) == np.sign(observed_odd)))
                record["antithetic_odd_pearson"] = float(pearsonr(predicted_odd, observed_odd)[0])
                slope = float((predicted_odd @ observed_odd) / (predicted_odd @ predicted_odd))
                record["amplitude_slope"] = slope
                # Selected-sign gain: choose the sign the model prefers, then read
                # the observed total response at that sign.
                preferred = -np.sign(predicted_odd)
                observed_total = np.where(preferred > 0, antithetic.plus_delta[target], antithetic.minus_delta[target])
                record["selected_sign_mean_delta_bpb"] = float(observed_total.mean())
                record["selected_sign_fraction_better_than_tied"] = float(np.mean(observed_total < 0))
            else:
                record["antithetic_odd_sign_accuracy"] = np.nan
                record["antithetic_odd_pearson"] = np.nan
                record["amplitude_slope"] = np.nan
                record["selected_sign_mean_delta_bpb"] = np.nan
                record["selected_sign_fraction_better_than_tied"] = np.nan
            rows.append(record)

    for panel_name, panel in (("delphi_3e18", spine.delphi_3e18), ("300m", spine.m300)):
        evaluate("BASE-0_zero_phase", None, ("tv_squared",), panel, panel_name)
        evaluate("BASE-5_fisher_even_only", None, ("fisher_chi2",), panel, panel_name)
        evaluate("BASE-1_effexp_dsp_odd", odd_effective_exposure, ("tv_squared",), panel, panel_name)
        evaluate("BASE-2_separate_heads_free_odd", odd_free_bucket, ("tv_squared", "tv_cubed"), panel, panel_name)
        evaluate("PMVT-R_marginal_value", odd_marginal_value(1.0), ("tv_squared",), panel, panel_name)
        evaluate(
            "C2_REX_retention_exchange", odd_retention_exchange(1.0), ("fisher_chi2", "tv_cubed"), panel, panel_name
        )
        evaluate("C3_BLT_boundary_overload", odd_retention_exchange(1.0), ("boundary_overload",), panel, panel_name)
        evaluate("C3_BLT_even_only", None, ("boundary_overload",), panel, panel_name)
    return pd.DataFrame(rows)


def shared_curvature_test(spine, spec, family_ids, n_families) -> pd.DataFrame:
    """C1: does one saturating curve reproduce the observed odd/even amplitude ratio?

    The shared-curvature model has no free odd scale: a single family-pooled
    coefficient vector multiplies both the ``G'`` odd block and the ``G''`` even
    block. Fitting it on the even-dominated paired contrast therefore *predicts*
    the odd amplitude, which the antithetic panel measures directly.
    """
    antithetic = spine.antithetic
    rows = []
    for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
        for retention in (0.0, 0.25, 0.5, 0.75):
            for panel_name, panel in (("delphi_3e18", spine.delphi_3e18), ("300m", spine.m300)):
                geometry = geometry_for(panel.alpha, panel.aggregate, panel.contrast, spec, family_ids, n_families)
                anti_geometry = geometry_for(
                    spine.delphi_3e18.alpha,
                    antithetic.aggregate,
                    antithetic.contrast,
                    spec,
                    family_ids,
                    n_families,
                )
                odd_design, even_design = shared_curvature_blocks(geometry, tau, retention)
                anti_odd, anti_even = shared_curvature_blocks(anti_geometry, tau, retention)
                # One coefficient vector drives both blocks: stack them.
                linked = odd_design + even_design
                for target in TARGETS:
                    delta = panel.delta[target]
                    design = np.hstack([linked, np.ones((len(delta), 1))])
                    coef = np.linalg.lstsq(design, delta, rcond=None)[0]
                    beta = coef[:-1]
                    predicted_odd = anti_odd @ beta
                    predicted_even = anti_even @ beta + coef[-1]
                    observed_odd, observed_even = antithetic.odd[target], antithetic.even[target]
                    if np.allclose(predicted_odd, 0.0):
                        continue
                    rows.append(
                        {
                            "tau": tau,
                            "retention": retention,
                            "fit_panel": panel_name,
                            "target": TARGET_LABEL[target],
                            "paired_delta_rmse": float(np.sqrt(((delta - design @ coef) ** 2).mean())),
                            "predicted_odd_rms": float(np.sqrt((predicted_odd**2).mean())),
                            "observed_odd_rms": float(np.sqrt((observed_odd**2).mean())),
                            "odd_amplitude_slope": float(
                                (predicted_odd @ observed_odd) / (predicted_odd @ predicted_odd)
                            ),
                            "odd_pearson": float(pearsonr(predicted_odd, observed_odd)[0]),
                            "predicted_even_mean": float(predicted_even.mean()),
                            "observed_even_mean": float(observed_even.mean()),
                            "even_amplitude_slope": float(
                                (predicted_even @ observed_even) / (predicted_even @ predicted_even)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


ODD_FIELDS = {
    "zero": None,
    "effective_exposure_dsp": odd_effective_exposure,
    "marginal_value_pmvt": odd_marginal_value(1.0),
    "family_pooled": odd_family_pooled,
    "retention_exchange_rex": odd_retention_exchange(1.0),
    "free_bucket_separate_heads": odd_free_bucket,
}


def candidate_gate_evaluation(spine, spec, family_ids, n_families) -> pd.DataFrame:
    """Acceptance gate: grouped hold-out on the amplitude-correct antithetic panel.

    Each odd field is fitted on the *directly observed* ``O`` of the training
    group and scored on the held-out group, so amplitude and direction are both
    tested honestly. Selection metrics read the observed total response at the
    sign the model prefers, which is what a deployment would actually do.
    """
    antithetic = spine.antithetic
    geometry = geometry_for(
        spine.delphi_3e18.alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families
    )
    rows = []
    for name, builder in ODD_FIELDS.items():
        design = builder(geometry) if builder is not None else np.zeros((len(antithetic.anchor_id), 0))
        for target in TARGETS:
            observed = antithetic.odd[target]
            for fold_name, groups in (
                ("leave_anchor", antithetic.anchor_id),
                ("leave_radius", antithetic.target_phase_tv),
                ("leave_direction", antithetic.direction_id),
            ):
                predictions = np.zeros(len(observed))
                for train, test in grouped_folds(groups):
                    if design.shape[1] == 0:
                        continue
                    gram = design[train].T @ design[train]
                    ridge = 1e-2 * np.trace(gram) / max(design.shape[1], 1)
                    coef = np.linalg.lstsq(
                        gram + ridge * np.eye(design.shape[1]), design[train].T @ observed[train], rcond=None
                    )[0]
                    predictions[test] = design[test] @ coef
                zero_rmse = float(np.sqrt((observed**2).mean()))
                rmse = float(np.sqrt(((observed - predictions) ** 2).mean()))
                # Deployment reads the total response at the model's preferred sign.
                preferred = np.where(predictions <= 0, 1.0, -1.0)
                realized = np.where(preferred > 0, antithetic.plus_delta[target], antithetic.minus_delta[target])
                oracle = np.minimum(antithetic.plus_delta[target], antithetic.minus_delta[target])
                order = np.argsort(predictions)
                selection = {}
                for k in (1, 3, 5):
                    if design.shape[1] == 0:
                        # With no odd model the rational policy is the tied control,
                        # whose paired contrast is exactly zero by construction.
                        chosen = 0.0
                    else:
                        chosen = float(realized[order[:k]].min())
                    selection[f"selected_gain_at_{k}_bpb"] = chosen
                    selection[f"regret_at_{k}"] = float(chosen - oracle.min())
                    selection[f"beats_tied_at_{k}"] = bool(chosen < 0.0)
                rows.append(
                    {
                        "odd_field": name,
                        "target": TARGET_LABEL[target],
                        "fold": fold_name,
                        "odd_parameters": int(design.shape[1]),
                        "held_out_odd_rmse": rmse,
                        "zero_odd_rmse": zero_rmse,
                        "odd_rmse_ratio_vs_zero": rmse / zero_rmse,
                        "beats_zero_phase": bool(rmse < zero_rmse),
                        "sign_accuracy": (
                            float(np.mean(np.sign(predictions) == np.sign(observed))) if design.shape[1] else np.nan
                        ),
                        "selected_sign_mean_delta_bpb": float(realized.mean()) if design.shape[1] else 0.0,
                        "selected_sign_fraction_better_than_tied": (
                            float(np.mean(realized < 0)) if design.shape[1] else 0.0
                        ),
                        "oracle_best_delta_bpb": float(oracle.min()),
                        **selection,
                    }
                )
    return pd.DataFrame(rows)


def exponent_gap_bootstrap(antithetic, rng: np.random.Generator) -> pd.DataFrame:
    """Paired bootstrap of ``q - p``, the exponent gap that decides the program.

    If ``q > p`` the symmetric transport cost outgrows the directional benefit and
    the attainable gain is bounded. The gap must be tested as a paired quantity,
    not read off two marginal intervals.
    """
    directions = sorted(set(antithetic.direction_id.tolist()))
    anchors = sorted(set(antithetic.anchor_id.tolist()))
    direction_index = np.asarray([directions.index(x) for x in antithetic.direction_id])
    anchor_index = np.asarray([anchors.index(x) for x in antithetic.anchor_id])
    radius = antithetic.realized_phase_tv
    n_directions = len(directions)

    def solve(target: str, selected_rows: np.ndarray) -> tuple[float, float]:
        """Refit the exponent pair on an arbitrary multiset of observation rows."""
        sigma_odd, sigma_even = odd_noise(target), even_noise(target)
        odd_values = antithetic.odd[target][selected_rows]
        even_values = antithetic.even[target][selected_rows]
        radius_sub = radius[selected_rows]
        anchor_sub = anchor_index[selected_rows]
        # Only the directions actually present get an amplitude parameter.
        present = np.unique(direction_index[selected_rows])
        remap = {int(value): position for position, value in enumerate(present)}
        theta_index = np.asarray([remap[int(value)] for value in direction_index[selected_rows]])
        n_theta = len(present)

        def residual(z: np.ndarray) -> np.ndarray:
            theta = z[:n_theta]
            amplitude = z[n_theta + 1 : n_theta + 1 + len(anchors)]
            return np.concatenate(
                [
                    (theta[theta_index] * radius_sub ** z[n_theta] - odd_values) / sigma_odd,
                    (amplitude[anchor_sub] * radius_sub ** z[-1] - even_values) / sigma_even,
                ]
            )

        start = np.concatenate([np.full(n_theta, 0.005), [1.5], np.full(len(anchors), 0.03), [2.0]])
        lower = np.concatenate([np.full(n_theta, -np.inf), [0.2], np.full(len(anchors), 0.0), [0.5]])
        upper = np.concatenate([np.full(n_theta, np.inf), [5.0], np.full(len(anchors), np.inf), [6.0]])
        z = least_squares(residual, start, bounds=(lower, upper), method="trf").x
        return float(z[n_theta]), float(z[-1])

    all_rows = np.arange(len(radius))
    rows = []
    for target in TARGETS:
        point = solve(target, all_rows)
        gaps = []
        # Directions are the resampling unit: each contributes one full radius
        # sweep at both anchors, and the odd amplitudes are direction specific.
        for _ in range(BOOTSTRAP_DRAWS // 10):
            picked = rng.integers(0, n_directions, n_directions)
            resampled = np.concatenate([np.flatnonzero(direction_index == j) for j in picked])
            if len(np.unique(direction_index[resampled])) < 6:
                continue
            p_boot, q_boot = solve(target, resampled)
            gaps.append(q_boot - p_boot)
        gaps_array = np.asarray(gaps) if gaps else np.zeros(1)
        rows.append(
            {
                "target": TARGET_LABEL[target],
                "odd_power_p": point[0],
                "even_power_q": point[1],
                "exponent_gap": point[1] - point[0],
                "gap_ci95_low": float(np.quantile(gaps_array, 0.025)),
                "gap_ci95_high": float(np.quantile(gaps_array, 0.975)),
                "probability_gap_positive": float(np.mean(gaps_array > 0)),
                "bootstrap_draws": len(gaps_array),
            }
        )
    return pd.DataFrame(rows)


def starcoder_schedule_check(spine) -> pd.DataFrame:
    """Odd/even radius exponents on the two dense StarCoder schedules.

    These surfaces have only two buckets, so the contrast is one dimensional and
    the odd/even split is identified by reflection about the tied line without
    any direction model. They test whether the exponent ordering ``q > p``
    survives a different schedule and a different phase-0 fraction.
    """
    rows = []
    for surface in (spine.starcoder_cosine, spine.starcoder_wsd):
        contrast = surface.contrast[:, 0]
        aggregate = surface.aggregate[:, 0]
        radius = np.abs(contrast)
        # Reference tied response as a function of the aggregate, taken from the
        # near-tied rows so that no phase model is assumed.
        tied_mask = radius < 1e-9
        if tied_mask.sum() < 3:
            rows.append({"surface": surface.name, "alpha": surface.alpha, "note": "no tied reference rows"})
            continue
        order = np.argsort(aggregate[tied_mask])
        tied_reference = np.interp(aggregate, aggregate[tied_mask][order], surface.bpb[tied_mask][order])
        delta = surface.bpb - tied_reference
        active = radius > 1e-6
        odd_estimates, even_estimates = [], []
        for index in np.flatnonzero(active):
            mirror = np.flatnonzero(
                active & (np.abs(aggregate - aggregate[index]) < 2e-3) & (np.abs(contrast + contrast[index]) < 2e-3)
            )
            if mirror.size:
                partner = mirror[0]
                odd_estimates.append((radius[index], 0.5 * (delta[index] - delta[partner])))
                even_estimates.append((radius[index], 0.5 * (delta[index] + delta[partner])))
        record: dict[str, Any] = {
            "surface": surface.name,
            "alpha": surface.alpha,
            "rows": len(surface.bpb),
            "tied_reference_rows": int(tied_mask.sum()),
            "reflection_pairs": len(odd_estimates),
        }
        # A reflection about the tied line is what separates odd from even without
        # assuming a response law. These dense surfaces were built as one-sided
        # sweeps, so they supply too few reflected pairs to identify the split.
        minimum_pairs_for_exponents = 12
        record["odd_even_split_identified"] = len(odd_estimates) >= minimum_pairs_for_exponents
        for label, estimates in (("odd", odd_estimates), ("even", even_estimates)):
            if len(estimates) < minimum_pairs_for_exponents:
                record[f"{label}_power"] = np.nan
                continue
            radii = np.asarray([r for r, _ in estimates])
            values = np.abs(np.asarray([v for _, v in estimates]))
            keep = values > 1e-6
            record[f"{label}_power"] = (
                float(np.polyfit(np.log(radii[keep]), np.log(values[keep]), 1)[0]) if keep.sum() >= 4 else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Raw-optimum audit
# ---------------------------------------------------------------------------


def raw_optimum_audit(spine, spec, family_ids, n_families, budget: pd.DataFrame) -> pd.DataFrame:
    """Optimum of the odd/even phase surface at a fixed aggregate, before any penalty.

    The aggregate is held at the observed anchor, so this isolates the phase
    block: it asks how much the fitted phase model believes it can gain, and at
    what contrast radius and maximum simulated epoch count.
    """
    antithetic = spine.antithetic
    rows = []
    for target in TARGETS:
        label = TARGET_LABEL[target]
        anti_geometry = geometry_for(
            spine.delphi_3e18.alpha, antithetic.aggregate, antithetic.contrast, spec, family_ids, n_families
        )
        design = odd_free_bucket(anti_geometry)
        observed = antithetic.odd[target]
        coefficients = np.linalg.solve(design.T @ design + 1e-2 * np.eye(design.shape[1]), design.T @ observed)
        # Choosing a direction with an estimated field realizes only the component
        # of the true field that the estimate recovers. That fraction is the
        # cosine between estimate and truth, for which the leave-direction-out
        # correlation is an observable proxy. Without it the linear-programming
        # envelope is a pure selection artifact.
        held_out = np.zeros(len(observed))
        for train, test in grouped_folds(antithetic.direction_id):
            sub = np.linalg.lstsq(
                design[train].T @ design[train] + 1e-2 * np.eye(design.shape[1]),
                design[train].T @ observed[train],
                rcond=None,
            )[0]
            held_out[test] = design[test] @ sub
        observed_variance = float(observed.var(ddof=1))
        reliability = np.sqrt(max(observed_variance - odd_noise(target) ** 2, 0.0) / observed_variance)
        raw_correlation = float(np.corrcoef(held_out, observed)[0, 1])
        realizable = float(np.clip(raw_correlation / reliability, 0.0, 1.0)) if reliability > 0 else 0.0
        for radius in (0.05, 0.10, 0.25, 0.50):
            envelope = feasible_best_direction(antithetic, coefficients, radius)
            for anchor, best_odd in envelope.items():
                row = budget[(budget.target == label) & (budget.anchor == anchor)]
                if row.empty or not np.isfinite(best_odd):
                    continue
                power_even = float(row.even_power_q.iloc[0])
                amplitude = float(row.even_amplitude_c.iloc[0])
                predicted_cost = amplitude * radius**power_even
                aggregate = antithetic.aggregate[antithetic.anchor_id == anchor][0]
                epochs = (spec.c0 / spine.delphi_3e18.alpha) * aggregate
                realizable_odd = realizable * best_odd
                rows.append(
                    {
                        "target": label,
                        "anchor": anchor,
                        "phase_tv": radius,
                        "in_sample_envelope_best_odd_bpb": -best_odd,
                        "leave_direction_realizable_fraction": realizable,
                        "realizable_best_odd_bpb": -realizable_odd,
                        "predicted_even_cost_bpb": predicted_cost,
                        "envelope_net_gain_bpb": best_odd - predicted_cost,
                        "realizable_net_gain_bpb": realizable_odd - predicted_cost,
                        "aggregate_max_epoch": float(epochs.max()),
                        "aggregate_max_bucket_weight": float(aggregate.max()),
                        "envelope_beats_tied": bool(best_odd - predicted_cost > 0),
                        "realizable_beats_tied": bool(realizable_odd - predicted_cost > 0),
                        "realizable_reaches_goal_0p01": bool(realizable_odd - predicted_cost >= GOAL_GAIN_BPB),
                    }
                )
    return pd.DataFrame(rows)


def approach_registry_delta() -> pd.DataFrame:
    """Routes introduced by this drive, with their terminal status and evidence."""
    return pd.DataFrame(
        [
            {
                "id": "PDE",
                "family": "Paired-contrast estimand",
                "relationship_to_prior": (
                    "New identification argument for the whole 99-route registry, not a new response law. "
                    "Prior routes fitted raw BPB, so aggregate-model error and phase signal were not separable."
                ),
                "materially_new_mechanism": "identification argument",
                "mechanistic_premise": (
                    "Every two-phase fit policy has an exactly aggregate-matched tied counterpart, so the paired "
                    "contrast Delta = L(a,d) - L(a,0) is observed with no aggregate-model error and equals O + C."
                ),
                "latent_state": "none; Delta is an observable",
                "state_transition": "none",
                "units_and_symmetries": "O odd in d, C even in d, both zero at d = 0",
                "single_phase_restriction": "Delta = 0 identically at d = 0 by construction",
                "cheapest_falsification": (
                    "Compare an odd field fitted on Delta against the separately observed antithetic O."
                ),
                "status": "retained_as_protocol",
                "status_evidence": (
                    "alpha recovered exactly (Delphi 0.7981376787495837, 300M 0.80); aggregate match error 3.09e-10; "
                    "238 paired coordinates per scale; contrast cloud nearly sign symmetric "
                    "(norm of mean unit direction 0.109), so odd/even canonical correlation is 0.48-0.60 versus "
                    "0.833 for aggregate/contrast in the raw design."
                ),
            },
            {
                "id": "C1-SCT",
                "family": "Shared-curvature transport",
                "relationship_to_prior": (
                    "Constrains prior retained-exposure families instead of extending them: the odd and even "
                    "amplitudes are forced to come from one curve, removing the free odd scale that made "
                    "earlier phase heads unidentified."
                ),
                "materially_new_mechanism": "linkage constraint plus identification argument",
                "mechanistic_premise": (
                    "One saturating acquisition curve G on retained exposure generates the odd response through "
                    "G' and the even response through G''. Because the even channel dominates the paired "
                    "contrast, the linkage transfers identification from C to O with no new reversals."
                ),
                "latent_state": "retained exposure x_i = r c0_i p0_i + c1_i p1_i",
                "state_transition": (
                    "Phase displacement enters as alpha(1-alpha)(1-r) c0_i d_i; the response is the first and "
                    "second order expansion of G at the tied retained exposure."
                ),
                "units_and_symmetries": "simulated epochs; retention r dimensionless in [0,1]",
                "single_phase_restriction": "both blocks vanish at d = 0",
                "cheapest_falsification": (
                    "The predicted odd amplitude has no free scale, so the antithetic O measures it directly."
                ),
                "status": "blocked_amplitude_linkage_not_panel_invariant",
                "status_evidence": (
                    "Across 80 (tau, retention, panel, target) configurations the implied odd amplitude slope "
                    "spans -0.692 to 1.965 with median 0.214, and the configurations that reach slope near one "
                    "differ by panel and target (Delphi/Table-9 at tau 4.0, retention 0.75; 300M/Uncheatable at "
                    "tau 0.5, retention 0.0). Best odd Pearson is 0.507. The linkage therefore does not pin the "
                    "odd amplitude in a panel-independent way."
                ),
            },
            {
                "id": "C2-REX",
                "family": "Retention-exchange field",
                "relationship_to_prior": (
                    "Escapes the finite-potential-transport obstruction by reweighting the aggregate marginal "
                    "value with family-specific retention, so the odd field is not proportional to the "
                    "aggregate gradient and does not vanish at the aggregate optimum."
                ),
                "materially_new_mechanism": "latent state reweighting plus new invariant",
                "mechanistic_premise": (
                    "Phase-0 evidence survives into the terminal state with family-specific retention r_f, so "
                    "displacing bucket mass late changes retained evidence by alpha(1-alpha)(1-r_f) d_i, valued "
                    "at the aggregate marginal learnability 1/(tau + E_i(a))."
                ),
                "latent_state": "per-family retained fraction r_f and aggregate marginal learnability m_i(a)",
                "state_transition": "odd displacement pooled to three canonical families",
                "units_and_symmetries": "tau in simulated epochs; r_f dimensionless",
                "single_phase_restriction": "odd block vanishes at d = 0",
                "cheapest_falsification": "Grouped hold-out on the antithetic panel with selection regret.",
                "status": "blocked_fails_selection_gate",
                "status_evidence": (
                    "Reduces held-out odd RMSE to 0.871-0.900 of the zero-phase baseline on Table-9 but reaches "
                    "1.045 on Uncheatable leave-anchor, and its top-1 selected policy is worse than tied in every "
                    "fold (+0.0027 to +0.0175 BPB). Improves an average error metric while worsening selection, "
                    "which is a prespecified rejection condition."
                ),
            },
            {
                "id": "C3-BLT",
                "family": "Boundary-limited overload transport",
                "relationship_to_prior": (
                    "Prior routes treated the symmetric phase cost as a metric or quadratic form on d. This "
                    "constructs it as the even part of a symmetrized physical overload functional, which is why "
                    "it can be near isotropic and supra-quadratic at once."
                ),
                "materially_new_mechanism": "new invariant and construction",
                "mechanistic_premise": (
                    "Phase separation pushes individual buckets toward within-phase repetition or toward absence "
                    "from the terminal phase. Symmetrizing that physical load in d yields an admissible even "
                    "cost whose direction dependence is weak because the load depends on per-bucket excursions "
                    "rather than on a quadratic form."
                ),
                "latent_state": "within-phase epoch rates and late-presence share",
                "state_transition": "symmetrized overload minus its tied value",
                "units_and_symmetries": "simulated epochs; threshold at one epoch",
                "single_phase_restriction": "vanishes exactly at d = 0",
                "cheapest_falsification": (
                    "Fit the even amplitude on paired Delta and compare against the separately observed "
                    "antithetic C, against the retained Fisher cost."
                ),
                "status": "retained_even_component_only",
                "status_evidence": (
                    "Best even model on both panels and both targets: antithetic even RMSE ratio versus zero "
                    "0.724 Uncheatable and 0.824 Table-9 fitted on Delphi, 0.784 and 0.865 fitted on 300M, "
                    "against 1.091/1.030 and 1.510/1.290 for the Fisher chi-square cost and 0.968/1.102 and "
                    "1.452/1.462 for a plain squared-TV cost. Supplies no odd channel, so it cannot select a "
                    "policy better than tied on its own."
                ),
            },
        ]
    )


def data_use_ledger() -> pd.DataFrame:
    """Append-only record of how each panel was used in this drive."""
    return pd.DataFrame(
        [
            {
                "batch": "batch1_protocol_freeze",
                "panel": "delphi_3e18 two-phase fit + exact tied counterpart",
                "rows": 238,
                "role": "development fit",
                "outcomes_read_before_freeze": False,
                "note": (
                    "Frozen in Fieldbook note_01kybvwx4vhvm7kwrqftdnja5s before any candidate was scored. "
                    "Panels, grouped folds, noise constants, and the candidate list were declared in that note."
                ),
            },
            {
                "batch": "batch1_protocol_freeze",
                "panel": "300m two-phase fit + exact tied counterpart",
                "rows": 238,
                "role": "independent-scale development fit",
                "outcomes_read_before_freeze": False,
                "note": "Same 39 buckets and same policies as Delphi; phase-0 fraction 0.80 rather than 0.798.",
            },
            {
                "batch": "batch1_identification_test",
                "panel": "delphi_3e18 balanced antithetic triples",
                "rows": 96,
                "role": "identification test; O and C observed separately",
                "outcomes_read_before_freeze": True,
                "note": (
                    "Previously published development panel, already reported in "
                    "delphi_3e18_aggressive_phase_asymmetry_results_20260723 and "
                    "phase_order_identifiability_envelope_20260724. Reused here as a development identification "
                    "set, not as confirmation. Any survivor still requires the untouched panel."
                ),
            },
            {
                "batch": "batch1_schedule_check",
                "panel": "starcoder cosine 50/50 and wsd 80/20",
                "rows": 250,
                "role": "schedule check",
                "outcomes_read_before_freeze": True,
                "note": (
                    "Inconclusive by construction: 9 and 0 reflected pairs respectively, below the 12-pair "
                    "threshold, so these surfaces do not identify the odd/even split. Consistent with the prior "
                    "phase-reversal observability audit."
                ),
            },
            {
                "batch": "not_accessed",
                "panel": "targeted pairwise phase-order panel",
                "rows": 0,
                "role": "sealed",
                "outcomes_read_before_freeze": False,
                "note": (
                    "Not yet trained. Every loader in phase_order_spine_20260725 asserts the absence of rows "
                    "whose training_series contains 'targeted_pairwise'."
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_odd_even_scaling(antithetic, fits: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{TARGET_LABEL[t]}: odd and even versus contrast radius" for t in TARGETS],
    )
    for column, target in enumerate(TARGETS, start=1):
        radius = antithetic.realized_phase_tv
        figure.add_trace(
            go.Scatter(
                x=radius,
                y=np.abs(antithetic.odd[target]),
                mode="markers",
                name=f"|odd| {TARGET_LABEL[target]}",
                marker={"size": 6, "color": "#2166ac"},
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=radius,
                y=antithetic.even[target],
                mode="markers",
                name=f"even {TARGET_LABEL[target]}",
                marker={"size": 6, "color": "#b2182b", "symbol": "diamond"},
            ),
            row=1,
            col=column,
        )
        row = fits[fits.target == TARGET_LABEL[target]].iloc[0]
        grid = np.linspace(0.02, 0.7, 100)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=row.best_direction_kappa * grid**row.odd_power_p,
                mode="lines",
                name=f"kappa rho^{row.odd_power_p:.2f}",
                line={"color": "#2166ac"},
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=0.03 * grid**row.even_power_q,
                mode="lines",
                name=f"c rho^{row.even_power_q:.2f}",
                line={"color": "#b2182b", "dash": "dash"},
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="phase TV", type="log", row=1, col=column)
        figure.update_yaxes(title_text="BPB", type="log", row=1, col=column)
    figure.update_layout(
        title="Directional benefit grows more slowly in radius than symmetric cost",
        template="plotly_white",
        height=460,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def plot_amplitude_calibration(calibration: pd.DataFrame, path: Path) -> None:
    figure = go.Figure()
    for odd_field, group in calibration.groupby("odd_field"):
        figure.add_trace(
            go.Bar(
                x=[f"{r.fit_panel}/{r.target}/{r.even_functional}" for r in group.itertuples()],
                y=group.amplitude_slope,
                name=str(odd_field),
            )
        )
    figure.add_hline(y=1.0, line={"color": "black", "dash": "dash"}, annotation_text="correct amplitude")
    figure.update_layout(
        title="Odd amplitude fitted on paired Delta versus the antithetic measurement",
        yaxis_title="slope of observed on predicted odd (1.0 = correct)",
        template="plotly_white",
        height=520,
        barmode="group",
        xaxis={"tickangle": -40},
    )
    figure.write_html(path, include_plotlyjs="cdn")


def plot_direction_structure(structure: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=["odd channel", "even channel"])
    for column, channel in enumerate(("odd", "even"), start=1):
        subset = structure[structure.channel == channel]
        for (target_label, anchor), group in subset.groupby(["target", "anchor"]):
            figure.add_trace(
                go.Scatter(
                    x=group.target_phase_tv,
                    y=group.latent_direction_snr,
                    mode="lines+markers",
                    name=f"{target_label} @ {str(anchor).replace('_frontier', '')}",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=1.0, line={"color": "black", "dash": "dot"}, row=1, col=column)
        figure.update_xaxes(title_text="target phase TV", row=1, col=column)
        figure.update_yaxes(title_text="latent across-direction SNR", row=1, col=column)
    figure.update_layout(
        title="Resolvable direction structure: present in the odd channel, absent in the even channel",
        template="plotly_white",
        height=460,
    )
    figure.write_html(path, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    spine = build_spine()
    spec = load_exposure_spec("delphi_3e18_two_phase_fit")
    assert spec.domains == spine.delphi_3e18.buckets, "catalog domain order differs from the fit panel"
    family_ids, family_names = family_index_for(spine.delphi_3e18.buckets, spec.families)
    n_families = len(family_names)

    structure = even_direction_structure(spine.antithetic)
    even_laws = even_law_comparison(spine.antithetic, spec, family_ids, n_families, spine.delphi_3e18.alpha)
    transfer = odd_direction_transfer(spine.antithetic)
    calibration = odd_amplitude_calibration(spine, spec, family_ids, n_families)
    reverse = odd_field_reverse_test(spine, spec, family_ids, n_families)
    fits, budget = scaling_budget(spine.antithetic, rng)
    candidates = candidate_evaluation(spine, spec, family_ids, n_families)
    gates = candidate_gate_evaluation(spine, spec, family_ids, n_families)
    gap = exponent_gap_bootstrap(spine.antithetic, rng)
    starcoder = starcoder_schedule_check(spine)
    curvature = shared_curvature_test(spine, spec, family_ids, n_families)
    optima = raw_optimum_audit(spine, spec, family_ids, n_families, budget)

    tables = {
        "direction_structure": structure,
        "even_law_comparison": even_laws,
        "odd_direction_transfer": transfer,
        "odd_amplitude_calibration": calibration,
        "odd_field_reverse_test": reverse,
        "scaling_exponent_fits": fits,
        "scaling_budget": budget,
        "candidate_metrics": candidates,
        "acceptance_gate": gates,
        "exponent_gap_bootstrap": gap,
        "starcoder_schedule_check": starcoder,
        "approach_registry_delta": approach_registry_delta(),
        "data_use_ledger": data_use_ledger(),
        "shared_curvature_test": curvature,
        "raw_optimum_audit": optima,
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    plot_odd_even_scaling(spine.antithetic, fits, output / "odd_even_radius_scaling.html")
    plot_amplitude_calibration(calibration, output / "odd_amplitude_calibration.html")
    plot_direction_structure(structure, output / "direction_structure_snr.html")

    protocol = {
        "estimand": "Delta(a,d) = L(a,d) - L(a,0) on exact aggregate-matched pairs",
        "alpha_by_scale": {"delphi_3e18": spine.delphi_3e18.alpha, "300m": spine.m300.alpha},
        "run_sigma_bpb": RUN_SIGMA,
        "paired_rows": {"delphi_3e18": len(spine.delphi_3e18), "300m": len(spine.m300)},
        "antithetic_rows": len(spine.antithetic.anchor_id),
        "families": {name: int((family_ids == i).sum()) for i, name in enumerate(family_names)},
        "l2_grid": list(L2_GRID),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "goal_gain_bpb": GOAL_GAIN_BPB,
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    print(json.dumps({name: list(frame.shape) for name, frame in tables.items()}, indent=2))
    print("\n=== scaling exponents ===")
    print(fits.to_string(index=False))
    print("\n=== exponent gap (paired bootstrap) ===")
    print(gap.to_string(index=False))
    print("\n=== scaling budget ===")
    print(budget.to_string(index=False))
    print("\n=== acceptance gate: realized selected gain versus tied (negative = beats tied) ===")
    sel = gates.pivot_table(
        index=["odd_field", "odd_parameters"], columns=["target", "fold"], values="selected_gain_at_1_bpb"
    )
    print(sel.to_string(float_format=lambda v: f"{v:+.5f}"))
    print("\n=== acceptance gate: odd RMSE ratio versus zero phase effect ===")
    pivot = gates.pivot_table(
        index=["odd_field", "odd_parameters"], columns=["target", "fold"], values="odd_rmse_ratio_vs_zero"
    )
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
