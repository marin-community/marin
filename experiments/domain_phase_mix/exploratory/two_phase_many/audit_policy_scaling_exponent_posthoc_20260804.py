# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Post-hoc asymptote and within-fiber audit for the policy-scaling exponent route."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from audit_policy_scaling_exponent_20260804 import (
    DEFAULT_INPUT,
    FLOOR_MARGIN,
    load_common_panel,
    panel_subset,
    policy_arrays,
)
from scipy.optimize import least_squares, lsq_linear, minimize_scalar
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs" / "policy_scaling_exponent_audit_20260804" / "posthoc"
MODELS = ("shared", "aggregate", "recency")
EXPOSED_STAGE1_PHI = 0.7317667285340376
EXPOSED_STAGE3_PHI = 0.6877371512038374


@dataclass(frozen=True)
class PerPolicyFloorFit:
    """A scaling fit with one independently bounded floor per policy."""

    model: str
    parameters: np.ndarray
    gammas: np.ndarray
    floors: np.ndarray
    amplitudes: np.ndarray
    rmse: float


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def policy_gammas(parameters: np.ndarray, policies: pd.DataFrame, model: str) -> np.ndarray:
    gamma_0 = parameters[-1] if model == "shared" else parameters[-2] if model == "aggregate" else parameters[-3]
    if model == "shared":
        return np.full(len(policies), gamma_0)
    if model == "aggregate":
        return gamma_0 + parameters[-1] * policies["aggregate"].to_numpy(dtype=float)
    return (
        gamma_0
        + parameters[-2] * policies["phase_0_starcoder"].to_numpy(dtype=float)
        + parameters[-1] * policies["phase_1_starcoder"].to_numpy(dtype=float)
    )


def fit_per_policy_floors(
    policies: pd.DataFrame,
    token_ratios: np.ndarray,
    outcomes: np.ndarray,
    model: str,
) -> PerPolicyFloorFit:
    """Fit a curve-shape law while allowing every policy its own floor and amplitude."""

    policy_count = len(policies)
    floor_caps = outcomes.min(axis=1) - FLOOR_MARGIN
    floor_starts = np.maximum(0.0, outcomes.min(axis=1) - 0.08)
    floor_logits = logit(floor_starts / floor_caps)
    amplitude_starts = np.log(np.maximum(outcomes[:, 0] - floor_starts, 1e-3))
    base = [*floor_logits, *amplitude_starts]
    floor_slice = slice(0, policy_count)
    amplitude_slice = slice(policy_count, 2 * policy_count)

    if model == "shared":
        starts = [np.asarray([*base, gamma], dtype=float) for gamma in (0.2, 0.5, 1.0)]
        lower = np.asarray([*([-12.0] * policy_count), *([-12.0] * policy_count), 1e-5])
        upper = np.asarray([*([12.0] * policy_count), *([3.0] * policy_count), 2.0])
    elif model == "aggregate":
        starts = [
            np.asarray([*base, gamma, slope], dtype=float)
            for gamma, slope in ((0.2, 0.5), (0.5, 0.0), (0.2, 1.0), (1.0, -0.5))
        ]
        lower = np.asarray([*([-12.0] * policy_count), *([-12.0] * policy_count), 1e-5, -2.0])
        upper = np.asarray([*([12.0] * policy_count), *([3.0] * policy_count), 2.0, 2.0])
    elif model == "recency":
        starts = [
            np.asarray([*base, gamma, early, late], dtype=float)
            for gamma, early, late in (
                (0.2, 0.1, 0.5),
                (0.5, 0.0, 0.0),
                (0.2, 0.5, 0.1),
                (0.2, 0.1, 1.0),
                (1.0, 0.1, 0.1),
            )
        ]
        lower = np.asarray([*([-12.0] * policy_count), *([-12.0] * policy_count), 1e-5, 0.0, 0.0])
        upper = np.asarray([*([12.0] * policy_count), *([3.0] * policy_count), 2.0, 2.0, 2.0])
    else:
        raise ValueError(f"Unknown model {model}")

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        floors = floor_caps * sigmoid(parameters[floor_slice])
        amplitudes = np.exp(parameters[amplitude_slice])
        gammas = policy_gammas(parameters, policies, model)
        return floors, amplitudes, gammas

    def residual(parameters: np.ndarray) -> np.ndarray:
        floors, amplitudes, gammas = unpack(parameters)
        prediction = floors[:, None] + amplitudes[:, None] * token_ratios[None, :] ** (-gammas[:, None])
        invalid_gamma = np.minimum(gammas - 1e-5, 0.0) * 100.0
        return np.concatenate([(prediction - outcomes).ravel(), invalid_gamma])

    best = None
    best_objective = float("inf")
    for start in starts:
        result = least_squares(residual, x0=start, bounds=(lower, upper), max_nfev=50_000)
        objective = float(np.sum(residual(result.x) ** 2))
        if objective < best_objective:
            best = result
            best_objective = objective
    if best is None:
        raise RuntimeError(f"No finite {model} fit")

    floors, amplitudes, gammas = unpack(best.x)
    return PerPolicyFloorFit(
        model=model,
        parameters=best.x,
        gammas=gammas,
        floors=floors,
        amplitudes=amplitudes,
        rmse=float(np.sqrt(np.mean(residual(best.x)[: outcomes.size] ** 2))),
    )


def fit_held_curve(gamma: float, ratios: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(ratios)), ratios ** (-gamma)])
    result = lsq_linear(
        design,
        outcomes,
        bounds=([0.0, 0.0], [float(outcomes.min()) - FLOOR_MARGIN, np.inf]),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.x[1])


def cross_validate(
    policies: pd.DataFrame,
    ratios: np.ndarray,
    outcomes: np.ndarray,
    model: str,
    split: str,
) -> pd.DataFrame:
    if split == "policy":
        groups = pd.Series(np.arange(len(policies)), index=policies.index)
    elif split == "aggregate":
        groups = policies["aggregate"].round(8)
    else:
        raise ValueError(f"Unknown split {split}")

    rows: list[dict[str, float | str]] = []
    for held_group in groups.unique():
        keep = (groups != held_group).to_numpy()
        held_indices = np.flatnonzero(~keep)
        fit = fit_per_policy_floors(policies.loc[keep].reset_index(drop=True), ratios, outcomes[keep], model)
        held_policies = policies.iloc[held_indices].reset_index(drop=True)
        gammas = policy_gammas(fit.parameters, held_policies, model)
        for local_index, policy_index in enumerate(held_indices):
            floor, amplitude = fit_held_curve(gammas[local_index], ratios[:3], outcomes[policy_index, :3])
            prediction = floor + amplitude * ratios[3] ** (-gammas[local_index])
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "coordinate": policies.iloc[policy_index]["coordinate"],
                    "aggregate": policies.iloc[policy_index]["aggregate"],
                    "observed": outcomes[policy_index, 3],
                    "predicted": prediction,
                    "residual": prediction - outcomes[policy_index, 3],
                    "gamma": gammas[local_index],
                }
            )
    return pd.DataFrame(rows)


def difference_profile_gamma(ratios: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """Fit power-law shape from differences, eliminating the unknown floor exactly."""

    differences = outcomes[:-1] - outcomes[1:]

    def objective(gamma: float) -> float:
        basis = ratios[:-1] ** (-gamma) - ratios[1:] ** (-gamma)
        amplitude = float(np.dot(differences, basis) / np.dot(basis, basis))
        return float(np.sum((differences - amplitude * basis) ** 2))

    result = minimize_scalar(objective, bounds=(1e-5, 3.0), method="bounded")
    return float(result.x), float(np.sqrt(result.fun / len(differences)))


def shape_diagnostics(
    policies: pd.DataFrame,
    ratios: np.ndarray,
    outcomes: np.ndarray,
    panel: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phi = EXPOSED_STAGE1_PHI if panel == "stage1" else EXPOSED_STAGE3_PHI
    rows = []
    for index, policy in policies.iterrows():
        gamma, rmse = difference_profile_gamma(ratios, outcomes[index])
        rows.append(
            {
                **policy.to_dict(),
                "exposed_recency_state": (1.0 - phi) * policy["phase_0_starcoder"] + phi * policy["phase_1_starcoder"],
                "difference_profile_gamma": gamma,
                "difference_profile_rmse": rmse,
            }
        )
    diagnostics = pd.DataFrame(rows)
    summaries = []
    for aggregate, group in diagnostics.groupby(diagnostics["aggregate"].round(8)):
        summaries.append(
            {
                "panel": panel,
                "aggregate": aggregate,
                "coordinates": len(group),
                "recency_gamma_spearman": (
                    float(spearmanr(group["exposed_recency_state"], group["difference_profile_gamma"]).statistic)
                    if len(group) >= 3
                    else float("nan")
                ),
                "gamma_mean": group["difference_profile_gamma"].mean(),
                "gamma_range": group["difference_profile_gamma"].max() - group["difference_profile_gamma"].min(),
            }
        )
    return diagnostics, pd.DataFrame(summaries)


def summarize_cv(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["panel", "split", "model"])
        .agg(
            rows=("residual", "size"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
            bias=("residual", "mean"),
            mae=("residual", lambda values: float(np.mean(np.abs(values)))),
        )
        .reset_index()
    )


def render_report(
    full_fits: pd.DataFrame,
    cv_summary: pd.DataFrame,
    fiber_summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Post-Hoc Policy-Scaling Falsification",
            "",
            "This analysis was requested by the post-run Opus 5 review. It does not alter the frozen protocol or its",
            "reported PASS. It tests a stronger nuisance alternative after outcomes were inspected.",
            "",
            "## Decision",
            "",
            "Reject the recency-conditioned exponent as a new surrogate or development route. Once every policy",
            "has an independently admissible asymptote and amplitude, aggregate-conditioned curve shape survives",
            "but recency conditioning does not improve held-policy or held-fiber prediction consistently.",
            "",
            "The surviving aggregate-to-curve-shape relation is a descriptive rung moderator. It is not a fixed-horizon",
            "policy surrogate and does not reopen the closed scale-coordinate routes.",
            "",
            "## Full fits with per-policy asymptotes",
            "",
            full_fits.to_markdown(index=False),
            "",
            "## Highest-rung warm-start prediction",
            "",
            cv_summary.to_markdown(index=False),
            "",
            "Each held policy contributes its first three rung outcomes to fit its nuisance floor and amplitude. The",
            "aggregate split removes every coordinate at the held aggregate before fitting global curve-shape terms.",
            "Neither split is cold-start mixture prediction.",
            "",
            "## Floor-free difference-profile shape",
            "",
            fiber_summary.to_markdown(index=False),
            "",
            "The token rungs are not exactly geometrically spaced. The diagnostic therefore fits all three successive",
            "differences to their exact power-law basis rather than treating a raw difference ratio as a constant.",
            "Differences eliminate the unknown floor exactly. Stage 3 is coordinate-disjoint but contains only four",
            "aggregate fibers and was designed after Stage-1 outcomes were available.",
            "",
            "## Interpretation boundary",
            "",
            "The result preserves two useful observations: curve shape changes with aggregate StarCoder share, and",
            "the late-recency ordering claimed by the shared-floor model is absent within aggregate-matched fibers.",
            "It does not identify a causal token-horizon exponent because token count, optimizer steps, phase lengths,",
            "and stream identity all change together. At one horizon any policy exponent is absorbed into amplitude.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = load_common_panel(args.input)
    fit_rows = []
    cv_frames = []
    diagnostic_frames = []
    fiber_frames = []
    for panel in ("stage1", "stage3"):
        policies, ratios, outcomes = policy_arrays(panel_subset(frame, panel))
        for model in MODELS:
            fit = fit_per_policy_floors(policies, ratios, outcomes, model)
            phi = float("nan")
            gamma_0 = fit.parameters[-1]
            aggregate_coefficient = float("nan")
            early_coefficient = float("nan")
            late_coefficient = float("nan")
            if model == "aggregate":
                gamma_0 = fit.parameters[-2]
                aggregate_coefficient = fit.parameters[-1]
            if model == "recency":
                gamma_0 = fit.parameters[-3]
                early, late = fit.parameters[-2:]
                early_coefficient = early
                late_coefficient = late
                phi = float(late / (early + late)) if early + late > 1e-12 else float("nan")
            fit_rows.append(
                {
                    "panel": panel,
                    "model": model,
                    "policies": len(policies),
                    "rmse": fit.rmse,
                    "gamma_min": fit.gammas.min(),
                    "gamma_median": np.median(fit.gammas),
                    "gamma_max": fit.gammas.max(),
                    "gamma_0": gamma_0,
                    "aggregate_coefficient": aggregate_coefficient,
                    "early_coefficient": early_coefficient,
                    "late_coefficient": late_coefficient,
                    "phi": phi,
                }
            )
            for split in ("policy", "aggregate"):
                predictions = cross_validate(policies, ratios, outcomes, model, split)
                predictions["panel"] = panel
                cv_frames.append(predictions)
        diagnostics, fiber_summary = shape_diagnostics(policies, ratios, outcomes, panel)
        diagnostics["panel"] = panel
        diagnostic_frames.append(diagnostics)
        fiber_frames.append(fiber_summary)

    full_fits = pd.DataFrame(fit_rows)
    predictions = pd.concat(cv_frames, ignore_index=True)
    cv_summary = summarize_cv(predictions)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    fiber_summary = pd.concat(fiber_frames, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_fits.to_csv(args.output_dir / "per_policy_floor_full_fits.csv", index=False)
    predictions.to_csv(args.output_dir / "per_policy_floor_cv_predictions.csv", index=False)
    cv_summary.to_csv(args.output_dir / "per_policy_floor_cv_summary.csv", index=False)
    diagnostics.to_csv(args.output_dir / "difference_profile_diagnostics.csv", index=False)
    fiber_summary.to_csv(args.output_dir / "difference_profile_fiber_summary.csv", index=False)
    (args.output_dir / "report.md").write_text(render_report(full_fits, cv_summary, fiber_summary))


if __name__ == "__main__":
    main()
