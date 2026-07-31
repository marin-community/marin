# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Evaluate target-shared finite-potential transport on the paired 3e18 swarm."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from run_phase_transport_synthesis import (
    EPS,
    TARGETS,
    direction_splits,
    exact_fiber_predictions,
    fit_sfos,
    load_heldouts,
    load_panel,
    metric_dict,
    phase_design,
    plot_scatter,
    predict_sfos,
    random_splits,
    summarize_fibers,
    summarize_heldouts,
)
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "reference_outputs" / "cross_session_shared_recency_20260723"
MODEL_IDS = (
    "shared_recency_target_jensen",
    "shared_transport",
    "equal_recency_target_jensen",
)
ALPHA_EQUAL_RECENCY = 0.5


@dataclass(frozen=True)
class JointFit:
    model_id: str
    theta: float
    gamma_by_target: dict[str, float]
    rank: int
    condition_number: float


def target_designs(
    panel,
    indices: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    aggregate_fits: dict[str, object] = {}
    designs: dict[str, np.ndarray] = {}
    deltas: dict[str, np.ndarray] = {}
    for target in TARGETS:
        aggregate_fit = fit_sfos(
            panel,
            panel.one_weights[indices, 0],
            panel.one_targets[target][indices],
        )
        design, _, _ = phase_design(
            panel,
            aggregate_fit,
            panel.two_weights[indices],
            "fpt_total_global",
        )
        aggregate_fits[target] = aggregate_fit
        designs[target] = design
        deltas[target] = panel.two_targets[target][indices] - panel.one_targets[target][indices]
    return aggregate_fits, designs, deltas


def joint_system(
    panel,
    designs: dict[str, np.ndarray],
    deltas: dict[str, np.ndarray],
    model_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    fixed_theta = (
        (ALPHA_EQUAL_RECENCY - panel.alpha0) / (panel.alpha0 * panel.alpha1)
        if model_id == "equal_recency_target_jensen"
        else None
    )
    for target_index, target in enumerate(TARGETS):
        design = designs[target]
        response = deltas[target].copy()
        if fixed_theta is not None:
            response -= fixed_theta * design[:, 0]
        scale = max(float(np.sqrt(np.mean(deltas[target] ** 2))), EPS)
        if model_id == "shared_transport":
            block = design.copy()
        elif fixed_theta is not None:
            block = np.zeros((len(design), len(TARGETS)))
            block[:, target_index] = design[:, 1]
        else:
            block = np.zeros((len(design), 1 + len(TARGETS)))
            block[:, 0] = design[:, 0]
            block[:, 1 + target_index] = design[:, 1]
        blocks.append(block / scale)
        responses.append(response / scale)
    matrix = np.vstack(blocks)
    response = np.concatenate(responses)
    return (
        matrix,
        response,
        np.asarray(
            [fixed_theta] if fixed_theta is not None else [],
            dtype=float,
        ),
    )


def fit_joint(
    panel,
    designs: dict[str, np.ndarray],
    deltas: dict[str, np.ndarray],
    model_id: str,
) -> JointFit:
    matrix, response, fixed = joint_system(panel, designs, deltas, model_id)
    if model_id == "shared_transport":
        lower = np.asarray([-np.inf, 0.0])
    elif model_id == "equal_recency_target_jensen":
        lower = np.zeros(len(TARGETS))
    else:
        lower = np.asarray([-np.inf, *([0.0] * len(TARGETS))])
    result = lsq_linear(
        matrix,
        response,
        bounds=(lower, np.full(matrix.shape[1], np.inf)),
        method="trf",
        tol=1e-13,
        lsmr_tol=1e-13,
        max_iter=2000,
    )
    if not result.success:
        raise RuntimeError(f"{model_id} joint fit failed: {result.message}")
    if model_id == "shared_transport":
        theta = float(result.x[0])
        gamma = {target: float(result.x[1]) for target in TARGETS}
    elif model_id == "equal_recency_target_jensen":
        theta = float(fixed[0])
        gamma = {target: float(result.x[index]) for index, target in enumerate(TARGETS)}
    else:
        theta = float(result.x[0])
        gamma = {target: float(result.x[1 + index]) for index, target in enumerate(TARGETS)}
    singular = np.linalg.svd(matrix, compute_uv=False)
    positive = singular[singular > 1e-10]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    return JointFit(
        model_id=model_id,
        theta=theta,
        gamma_by_target=gamma,
        rank=int(np.linalg.matrix_rank(matrix)),
        condition_number=condition,
    )


def predict_delta(
    fit: JointFit,
    target: str,
    design: np.ndarray,
) -> np.ndarray:
    return fit.theta * design[:, 0] + fit.gamma_by_target[target] * design[:, 1]


def run_cv(panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    all_indices = np.arange(len(panel.group_ids))
    schemes = [
        *[(f"random_seed_{seed}", random_splits(all_indices, seed)) for seed in (20260723, 20260724, 20260725)],
        ("direction", direction_splits(panel)),
    ]
    for scheme, splits in schemes:
        for fold, (train, test) in enumerate(splits):
            aggregate_fits, train_designs, train_deltas = target_designs(panel, train)
            for model_id in MODEL_IDS:
                fit = fit_joint(panel, train_designs, train_deltas, model_id)
                fits.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "model_id": model_id,
                        "theta": fit.theta,
                        "implied_recency_share": (panel.alpha0 + fit.theta * panel.alpha0 * panel.alpha1),
                        "condition_number": fit.condition_number,
                        "rank": fit.rank,
                        **{f"gamma::{target}": fit.gamma_by_target[target] for target in TARGETS},
                    }
                )
                for target in TARGETS:
                    test_design, _, _ = phase_design(
                        panel,
                        aggregate_fits[target],
                        panel.two_weights[test],
                        "fpt_total_global",
                    )
                    predicted_delta = predict_delta(fit, target, test_design)
                    observed_delta = panel.two_targets[target][test] - panel.one_targets[target][test]
                    predicted_one = predict_sfos(
                        panel,
                        aggregate_fits[target],
                        panel.one_weights[test, 0],
                    )
                    for local, row_index in enumerate(test):
                        rows.append(
                            {
                                "target": target,
                                "model_id": model_id,
                                "scheme": scheme,
                                "fold": fold,
                                "row_index": int(row_index),
                                "group_id": panel.group_ids[row_index],
                                "observed_one": panel.one_targets[target][row_index],
                                "predicted_one": predicted_one[local],
                                "observed_delta": observed_delta[local],
                                "predicted_delta": predicted_delta[local],
                                "observed_two": panel.two_targets[target][row_index],
                                "predicted_two": (predicted_one[local] + predicted_delta[local]),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(fits)


def summarize_cv(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["target", "model_id", "scheme"],
        sort=False,
    ):
        target, model_id, scheme = keys
        for response in ("delta", "two"):
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "scheme": scheme,
                    "response": response,
                    **metric_dict(
                        group[f"observed_{response}"].to_numpy(float),
                        group[f"predicted_{response}"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_heldouts(panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldouts, weights = load_heldouts(panel)
    aggregate = panel.alpha0 * weights[:, 0] + panel.alpha1 * weights[:, 1]
    all_indices = np.arange(len(panel.group_ids))
    aggregate_fits, train_designs, train_deltas = target_designs(panel, all_indices)
    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        fit = fit_joint(panel, train_designs, train_deltas, model_id)
        fit_rows.append(
            {
                "model_id": model_id,
                "theta": fit.theta,
                "implied_recency_share": (panel.alpha0 + fit.theta * panel.alpha0 * panel.alpha1),
                "condition_number": fit.condition_number,
                "rank": fit.rank,
                **{f"gamma::{target}": fit.gamma_by_target[target] for target in TARGETS},
            }
        )
        for target in TARGETS:
            heldout_design, _, _ = phase_design(
                panel,
                aggregate_fits[target],
                weights,
                "fpt_total_global",
            )
            predicted_aggregate = predict_sfos(
                panel,
                aggregate_fits[target],
                aggregate,
            )
            predicted_phase = predict_delta(fit, target, heldout_design)
            result = heldouts.copy()
            result["fit_target"] = target
            result["model_id"] = model_id
            result["predicted_aggregate"] = predicted_aggregate
            result["predicted_phase_delta"] = predicted_phase
            result["predicted_target"] = predicted_aggregate + predicted_phase
            result["observed_target"] = heldouts[target].to_numpy(float)
            result["residual"] = result["predicted_target"] - result["observed_target"]
            result["optimism"] = result["observed_target"] - result["predicted_target"]
            result["phase_tv"] = 0.5 * np.sum(
                np.abs(weights[:, 1] - weights[:, 0]),
                axis=1,
            )
            aggregate_key = np.round(aggregate, 10)
            result["aggregate_hash"] = [np.asarray(row, dtype=np.float64).tobytes().hex() for row in aggregate_key]
            prediction_frames.append(result)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(fit_rows)


def write_report(
    cv_metrics: pd.DataFrame,
    full_fits: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
    fiber_metrics: pd.DataFrame,
) -> None:
    cv = cv_metrics.loc[(cv_metrics["response"] == "delta") & cv_metrics["scheme"].str.startswith("random")]
    cv = cv.groupby(["target", "model_id"], as_index=False).agg(rmse=("rmse", "mean"), spearman=("spearman", "mean"))
    heldout = heldout_metrics.loc[heldout_metrics["slice"] == "coordinate_disjoint_target_matched"]
    fiber = fiber_metrics.loc[fiber_metrics["slice"] == "all_exact_aggregate_fibers"]
    lines = [
        "# Shared-recency finite-potential transport",
        "",
        "This frozen batch tests whether the dimensionless retention/recency law is shared across smooth targets.",
        "",
        "## Full fits",
        "",
        full_fits.to_markdown(index=False),
        "",
        "## Paired phase-delta CV",
        "",
        cv.to_markdown(index=False),
        "",
        "## Target-matched 3e18 development heldouts",
        "",
        heldout.to_markdown(index=False),
        "",
        "## Exact aggregate fibers",
        "",
        fiber.to_markdown(index=False),
        "",
        (
            "The aggregate spine is unchanged from batch 1; this batch can identify "
            "a transferable phase law but cannot repair aggregate extrapolation."
        ),
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    cv_predictions, cv_fits = run_cv(panel)
    cv_metrics = summarize_cv(cv_predictions)
    heldout_predictions, full_fits = run_heldouts(panel)
    heldout_metrics = summarize_heldouts(heldout_predictions)
    fibers = exact_fiber_predictions(heldout_predictions)
    fiber_metrics = summarize_fibers(fibers)

    cv_predictions.to_csv(OUTPUT / "paired_cv_predictions.csv", index=False)
    cv_metrics.to_csv(OUTPUT / "paired_cv_metrics.csv", index=False)
    cv_fits.to_csv(OUTPUT / "paired_cv_fits.csv", index=False)
    heldout_predictions.to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    heldout_metrics.to_csv(OUTPUT / "heldout_metrics.csv", index=False)
    fibers.to_csv(OUTPUT / "exact_fiber_predictions.csv", index=False)
    fiber_metrics.to_csv(OUTPUT / "exact_fiber_metrics.csv", index=False)
    full_fits.to_csv(OUTPUT / "full_fits.csv", index=False)

    plot_scatter(
        heldout_predictions,
        "observed_target",
        "predicted_target",
        "Shared-recency transport: 3e18 development heldouts",
        "../cross_session_shared_recency_20260723/heldout_predictions.html",
    )
    write_report(cv_metrics, full_fits, heldout_metrics, fiber_metrics)
    ledger = {
        "batch": "cross_session_shared_recency_20260723",
        "candidate_equations_frozen_before_heldout_evaluation": True,
        "prior_exposed_evidence": [
            "batch-1 fpt_total_global coefficients and development metrics",
            "batch-2 compact-FPT coefficients and development metrics",
        ],
        "new_mechanism": "cross-target identification of a schedule-level recency share",
        "heldout_use": "development only",
        "future_confirmation_required": True,
    }
    (OUTPUT / "data_use_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print((OUTPUT / "report.md").read_text())


if __name__ == "__main__":
    main()
