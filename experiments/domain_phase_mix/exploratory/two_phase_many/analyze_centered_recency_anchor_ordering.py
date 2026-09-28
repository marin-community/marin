# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Optimize phase order around the best observed aggregate mixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_optima as optimum_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_anchor_ordering_20260710"
DEFAULT_BENCHMARK_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_residual_20260710"
NOISE_SD = optimum_audit.NOISE_SD
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def fit_signed_residual(
    candidate: centered.FittedCandidate,
    dataset: pooled.Dataset,
    l2: float,
) -> centered.FittedCandidate:
    """Refit the two tied features with unconstrained signed ridge."""
    design = centered.residual_design(
        candidate.backbone.base,
        dataset.weights,
        dataset.c0,
        dataset.c1,
        centered.ResidualKind.TIED,
    )
    backbone_prediction = centered.coverage.predict(
        candidate.backbone,
        dataset.weights,
        candidate.alpha0,
        candidate.alpha1,
    )
    target = dataset.y - backbone_prediction
    gram = design.T @ design + l2 * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ target)
    return centered.FittedCandidate(
        backbone=candidate.backbone,
        residual=centered.ResidualHead(
            kind=centered.ResidualKind.TIED,
            coef=np.asarray(coef, dtype=float),
            l2=l2,
        ),
        c0=candidate.c0,
        c1=candidate.c1,
        alpha0=candidate.alpha0,
        alpha1=candidate.alpha1,
    )


def weights_from_delta(
    aggregate: np.ndarray,
    delta: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    return np.stack(
        [
            aggregate + alpha1 * delta,
            aggregate - alpha0 * delta,
        ]
    )


def phase_schedule_kl(
    weights: np.ndarray,
    reference_weights: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    total = 0.0
    for alpha, phase, reference_phase in (
        (alpha0, weights[0], reference_weights[0]),
        (alpha1, weights[1], reference_weights[1]),
    ):
        reference = np.clip(reference_phase, 1e-15, None)
        clipped = np.clip(phase, 1e-15, None)
        total += alpha * float(np.sum(clipped * np.log(clipped / reference)))
    return total


def phase_order_kl(
    weights: np.ndarray,
    aggregate: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    return phase_schedule_kl(
        weights,
        np.stack([aggregate, aggregate]),
        alpha0,
        alpha1,
    )


def random_start(
    aggregate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = rng.normal(size=len(aggregate))
    direction -= float(np.dot(aggregate, direction))
    delta = aggregate * direction
    positive = delta > 0
    negative = delta < 0
    scales = [1.0]
    if np.any(positive):
        scales.append(float(np.min(upper[positive] / delta[positive])))
    if np.any(negative):
        scales.append(float(np.min(lower[negative] / delta[negative])))
    return delta * min(scales) * 0.25


def optimize_order(
    model: centered.FittedCandidate,
    aggregate: np.ndarray,
    order_kl_reg: float,
    penalty_reference: np.ndarray | None = None,
) -> np.ndarray:
    alpha0, alpha1 = model.alpha0, model.alpha1
    if penalty_reference is None:
        penalty_reference = np.stack([aggregate, aggregate])
    reference_aggregate = alpha0 * penalty_reference[0] + alpha1 * penalty_reference[1]
    if not np.allclose(reference_aggregate, aggregate, atol=1e-9):
        raise ValueError("Penalty reference must have the fixed target aggregate")
    lower = -aggregate / alpha1 + 1e-12
    upper = aggregate / alpha0 - 1e-12

    def objective(delta: np.ndarray) -> float:
        weights = weights_from_delta(aggregate, delta, alpha0, alpha1)
        prediction = float(model.predict(weights[None, :, :])[0])
        return prediction + order_kl_reg * phase_schedule_kl(weights, penalty_reference, alpha0, alpha1)

    rng = np.random.default_rng(0)
    starts = [np.zeros_like(aggregate)]
    starts.extend(random_start(aggregate, lower, upper, rng) for _ in range(8))
    best_value = np.inf
    best_weights: np.ndarray | None = None
    constraints = {"type": "eq", "fun": lambda delta: float(np.sum(delta))}
    bounds = list(zip(lower, upper, strict=True))
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            weights = weights_from_delta(
                aggregate,
                np.asarray(result.x, dtype=float),
                alpha0,
                alpha1,
            )
            if np.min(weights) >= -1e-9:
                best_value = float(result.fun)
                best_weights = weights
    if best_weights is None:
        raise RuntimeError(f"No finite fixed-aggregate result at order KL={order_kl_reg:g}")
    return best_weights


def best_single_phase_anchor(dataset: pooled.Dataset) -> tuple[int, str]:
    mask = dataset.frame["policy_family"].eq("single_phase").to_numpy()
    if not np.any(mask):
        raise ValueError(f"{dataset.name}: no single-phase anchors")
    candidates = np.flatnonzero(mask)
    index = int(candidates[np.argmin(dataset.y[candidates])])
    weights = dataset.weights[index]
    if not np.allclose(weights[0], weights[1], atol=1e-10):
        raise ValueError(f"{dataset.name}: selected anchor is not tied")
    name_column = "run_name" if "run_name" in dataset.frame.columns else dataset.frame.columns[0]
    return index, str(dataset.frame.iloc[index][name_column])


def nearest_observed_tv(dataset: pooled.Dataset, weights: np.ndarray) -> tuple[float, float]:
    distances = 0.5 * np.abs(dataset.weights - weights[None, :, :]).sum(axis=2).mean(axis=1)
    nearest = int(np.argmin(distances))
    return float(distances[nearest]), float(dataset.y[nearest])


def analyze_objective(
    dataset: pooled.Dataset,
    objective: str,
    benchmark_dir: Path,
    order_kl_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    l2 = optimum_audit.selected_l2(benchmark_dir, dataset.name, centered.ResidualKind.TIED)
    nonnegative = centered.fit_full_candidate(dataset, centered.ResidualKind.TIED, l2)
    signed = fit_signed_residual(nonnegative, dataset, l2)
    anchor_index, anchor_name = best_single_phase_anchor(dataset)
    aggregate = np.asarray(dataset.weights[anchor_index, 0], dtype=float)
    tied = np.stack([aggregate, aggregate])
    anchor_observed = float(dataset.y[anchor_index])
    nonnegative_anchor = float(nonnegative.predict(tied[None, :, :])[0])
    signed_anchor = float(signed.predict(tied[None, :, :])[0])
    pair_noise_sd = np.sqrt(2.0) * NOISE_SD[objective]
    rows = []
    weight_rows = []
    for order_kl_reg in order_kl_values:
        print(f"Optimizing {dataset.name} fixed aggregate, order KL={order_kl_reg:g}", flush=True)
        candidate = optimize_order(nonnegative, aggregate, order_kl_reg)
        candidate_aggregate = nonnegative.alpha0 * candidate[0] + nonnegative.alpha1 * candidate[1]
        if not np.allclose(candidate_aggregate, aggregate, atol=1e-9):
            raise ValueError("Fixed-aggregate optimizer changed aggregate exposure")
        nonnegative_prediction = float(nonnegative.predict(candidate[None, :, :])[0])
        signed_prediction = float(signed.predict(candidate[None, :, :])[0])
        nonnegative_gain = nonnegative_anchor - nonnegative_prediction
        signed_gain = signed_anchor - signed_prediction
        nearest_tv, nearest_observed = nearest_observed_tv(dataset, candidate)
        phase_tv = float(0.5 * np.abs(candidate[0] - candidate[1]).sum())
        total_epochs = candidate[0] * dataset.c0 + candidate[1] * dataset.c1
        rows.append(
            {
                "dataset": dataset.name,
                "objective": objective,
                "anchor_name": anchor_name,
                "anchor_observed": anchor_observed,
                "residual_l2": l2,
                "order_kl_reg": order_kl_reg,
                "nonnegative_prediction": nonnegative_prediction,
                "signed_prediction": signed_prediction,
                "nonnegative_ordering_gain": nonnegative_gain,
                "signed_ordering_gain": signed_gain,
                "nonnegative_gain_diff_sd": nonnegative_gain / pair_noise_sd,
                "signed_gain_diff_sd": signed_gain / pair_noise_sd,
                "phase_order_kl": phase_order_kl(
                    candidate,
                    aggregate,
                    nonnegative.alpha0,
                    nonnegative.alpha1,
                ),
                "phase_tv": phase_tv,
                "nearest_observed_tv": nearest_tv,
                "nearest_observed": nearest_observed,
                "max_total_epoch": float(np.max(total_epochs)),
                "signed_coef_signal": float(signed.residual.coef[0]),
                "signed_coef_penalty": float(signed.residual.coef[1]),
                "nnls_coef_signal": float(nonnegative.residual.coef[0]),
                "nnls_coef_penalty": float(nonnegative.residual.coef[1]),
                "passes_support_gate": nearest_tv <= 0.05,
                "passes_power_gate": nonnegative_gain >= 2.0 * pair_noise_sd,
                "passes_signed_direction_gate": signed_gain > 0.0,
                "passes_signed_power_gate": signed_gain >= pair_noise_sd,
            }
        )
        for phase in range(2):
            for domain, weight in zip(dataset.domain_names, candidate[phase], strict=True):
                weight_rows.append(
                    {
                        "dataset": dataset.name,
                        "objective": objective,
                        "anchor_name": anchor_name,
                        "order_kl_reg": order_kl_reg,
                        "phase": phase,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def write_plots(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    for metric in (
        "nonnegative_ordering_gain",
        "signed_ordering_gain",
        "phase_tv",
        "nearest_observed_tv",
    ):
        figure = px.line(
            diagnostics,
            x="order_kl_reg",
            y=metric,
            color="objective",
            markers=True,
            log_x=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Best-observed aggregate, phase-order sweep: {metric}",
        )
        figure.write_html(
            output_dir / f"anchor_order_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--order-kl-values",
        default="0.001,0.003,0.01,0.03,0.1,0.3,1,2,3,5,7.5,10,30,100",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_kl_values = pooled.parse_float_list(args.order_kl_values)
    datasets, _external = centered.load_datasets()
    result_frames = []
    weight_frames = []
    for objective in ("uncheatable", "table9"):
        result, weights = analyze_objective(
            datasets[f"300m_{objective}"],
            objective,
            args.benchmark_dir,
            order_kl_values,
        )
        result_frames.append(result)
        weight_frames.append(weights)
    diagnostics = pd.concat(result_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "anchor_order_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "anchor_order_weights_long.csv", index=False)
    write_plots(diagnostics, args.output_dir)
    report = [
        "# Best-observed aggregate, centered phase-order audit",
        "",
        diagnostics.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote anchor-order audit to {args.output_dir}")


if __name__ == "__main__":
    main()
