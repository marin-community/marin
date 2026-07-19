# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit unconstrained optima of the strongest baseline and closest extensions."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from scipy.special import softmax

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_kish_collision_invariant as collision,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_phase_boundary_adaptation as phase,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_PHASE_METRICS = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/round15_phase_boundary_adaptation/metrics.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/raw_optimum_audit"
POLICIES = ("single_phase", "two_phase")
MODEL_NAMES = ("baseline", "aggregate_collision", "within_phase_collision", "phase_information")


@dataclass(frozen=True)
class Fitted:
    name: str
    model: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--phase-metrics", type=Path, default=DEFAULT_PHASE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    return parser.parse_args()


def phase_config(dataset_id: base.DatasetId, metrics: pd.DataFrame) -> phase.Config:
    selected = metrics.loc[
        metrics["dataset"].eq(dataset_id.value)
        & metrics["split"].eq("fit_oof")
        & metrics["config"].str.startswith("phase_information")
    ].sort_values(["rmse", "regret_at_1"])
    if len(selected) != 1:
        raise ValueError(f"Expected one fit-selected phase-information config for {dataset_id.value}")
    key = str(selected.iloc[0]["config"])
    smoothing = float(key.rsplit("-", maxsplit=1)[-1])
    return phase.Config(phase.Mechanism.PHASE_INFORMATION, smoothing)


def fitted_models(
    dataset_id: base.DatasetId,
    dataset: Any,
    source: pd.DataFrame,
    links: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    indices: np.ndarray,
) -> tuple[Fitted, ...]:
    deficit_config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source)
    link_config = support.selected_link_config(dataset_id, links)
    return (
        Fitted(
            "baseline",
            collision.fit_model(
                dataset,
                deficit_config,
                link_config,
                collision.Config(collision.Mechanism.BASELINE),
                indices,
            ),
        ),
        Fitted(
            "aggregate_collision",
            collision.fit_model(
                dataset,
                deficit_config,
                link_config,
                collision.Config(collision.Mechanism.AGGREGATE_COLLISION, 0.0),
                indices,
            ),
        ),
        Fitted(
            "within_phase_collision",
            collision.fit_model(
                dataset,
                deficit_config,
                link_config,
                collision.Config(collision.Mechanism.WITHIN_PHASE_COLLISION, 0.0),
                indices,
            ),
        ),
        Fitted(
            "phase_information",
            phase.fit_model(
                dataset,
                deficit_config,
                link_config,
                phase_config(dataset_id, phase_metrics),
                indices,
            ),
        ),
    )


def logits_to_weights(logits: np.ndarray, m: int, policy: str) -> np.ndarray:
    if policy == "single_phase":
        tied = softmax(np.asarray(logits, dtype=float))
        return np.stack([tied, tied], axis=0)
    return softmax(np.asarray(logits, dtype=float).reshape(2, m), axis=1)


def weights_to_logits(weights: np.ndarray, policy: str) -> np.ndarray:
    if policy == "single_phase":
        values = 0.5 * (weights[0] + weights[1])
    else:
        values = weights
    logits = np.log(np.maximum(values, 1e-12))
    return np.asarray(logits - np.mean(logits, axis=-1, keepdims=True), dtype=float).ravel()


def optimize(
    fitted: Fitted,
    dataset: Any,
    policy: str,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, bool]:
    def objective(logits: np.ndarray) -> float:
        weights = logits_to_weights(logits, dataset.m, policy)
        return float(fitted.model.predict(weights[None, :, :])[0])

    best: tuple[float, np.ndarray, bool] | None = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        candidate = (float(result.fun), np.asarray(result.x, dtype=float), bool(result.success))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No optimization starts")
    return logits_to_weights(best[1], dataset.m, policy), best[0], best[2]


def optimization_starts(dataset: Any, policy: str, seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    proportional = base.proportional_weights(dataset)
    starts = [weights_to_logits(np.stack([proportional, proportional]), policy)]
    best_observed = dataset.weights[int(np.argmin(dataset.target))]
    starts.append(weights_to_logits(best_observed, policy))
    for concentration in (0.25, 1.0, 4.0):
        for _ in range(math.ceil((count - len(starts)) / 3)):
            if policy == "single_phase":
                sample = rng.dirichlet(np.full(dataset.m, concentration))
                weights = np.stack([sample, sample])
            else:
                weights = np.stack(
                    [rng.dirichlet(np.full(dataset.m, concentration)), rng.dirichlet(np.full(dataset.m, concentration))]
                )
            starts.append(weights_to_logits(weights, policy))
            if len(starts) >= count:
                return starts
    return starts[:count]


def support_distance(dataset: Any, weights: np.ndarray) -> float:
    fit = dataset.weights.reshape(dataset.n, -1)
    scale = np.maximum(np.std(fit, axis=0), 1e-3)
    distance = np.linalg.norm((fit - weights.reshape(1, -1)) / scale, axis=1)
    return float(np.min(distance))


def diagnostics(dataset: Any, weights: np.ndarray) -> dict[str, float]:
    exposure0 = weights[0] * dataset.c0
    exposure1 = weights[1] * dataset.c1
    total = exposure0 + exposure1
    phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate_weight = phase_fraction * weights[0] + (1.0 - phase_fraction) * weights[1]
    return {
        "max_bucket_weight": float(np.max(weights)),
        "max_simulated_epochs": float(np.max(total)),
        "mean_simulated_epochs": float(np.mean(total)),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "aggregate_hhi": float(np.sum(np.square(aggregate_weight))),
        "fit_support_distance": support_distance(dataset, weights),
    }


def main() -> None:
    args = parse_args()
    for path in (args.source_metrics, args.link_metrics, args.phase_metrics):
        gate.assert_sealed_absent(path)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    phase_metrics = pd.read_csv(args.phase_metrics)
    optimum_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for dataset_id in (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9):
        dataset = base.load_dataset(dataset_id)
        all_indices = np.arange(dataset.n)
        models = fitted_models(dataset_id, dataset, source, links, phase_metrics, all_indices)
        full_optima: dict[tuple[str, str], np.ndarray] = {}
        for fitted in models:
            for policy in POLICIES:
                weights, predicted, converged = optimize(
                    fitted,
                    dataset,
                    policy,
                    optimization_starts(dataset, policy, 20260717, 12),
                )
                full_optima[(fitted.name, policy)] = weights
                record = {
                    "dataset": dataset_id.value,
                    "model": fitted.name,
                    "policy": policy,
                    "predicted_bpb": predicted,
                    "optimizer_converged": converged,
                    **diagnostics(dataset, weights),
                }
                optimum_rows.append(record)
                exposure = weights[0] * dataset.c0 + weights[1] * dataset.c1
                for domain, phase0, phase1, epochs in zip(
                    dataset.domains, weights[0], weights[1], exposure, strict=True
                ):
                    weight_rows.append(
                        {
                            "dataset": dataset_id.value,
                            "model": fitted.name,
                            "policy": policy,
                            "domain": domain,
                            "phase0_weight": phase0,
                            "phase1_weight": phase1,
                            "simulated_epochs": epochs,
                        }
                    )

        for seed in range(args.bootstrap_repeats):
            train, _test = base.split_indices(dataset, dataset_id, all_indices, seed)[seed % base.N_SPLITS]
            for fitted in fitted_models(dataset_id, dataset, source, links, phase_metrics, train):
                policy = "two_phase"
                reference = full_optima[(fitted.name, policy)]
                starts = [
                    weights_to_logits(reference, policy),
                    *optimization_starts(dataset, policy, seed, 3),
                ]
                weights, predicted, converged = optimize(fitted, dataset, policy, starts)
                stability_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "model": fitted.name,
                        "seed": seed,
                        "predicted_bpb": predicted,
                        "optimizer_converged": converged,
                        "tv_from_full_optimum": float(0.25 * np.abs(weights - reference).sum()),
                        **diagnostics(dataset, weights),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optima = pd.DataFrame(optimum_rows)
    weights = pd.DataFrame(weight_rows)
    stability = pd.DataFrame(stability_rows)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    weights.to_csv(args.output_dir / "raw_optimum_weights.csv", index=False)
    stability.to_csv(args.output_dir / "raw_optimum_stability.csv", index=False)
    summary = stability.groupby(["dataset", "model"], as_index=False).agg(
        median_tv_from_full=("tv_from_full_optimum", "median"),
        p90_tv_from_full=("tv_from_full_optimum", lambda values: float(np.quantile(values, 0.9))),
        median_max_epoch=("max_simulated_epochs", "median"),
        p90_max_epoch=("max_simulated_epochs", lambda values: float(np.quantile(values, 0.9))),
        median_support_distance=("fit_support_distance", "median"),
        convergence_rate=("optimizer_converged", "mean"),
    )
    summary.to_csv(args.output_dir / "raw_optimum_stability_summary.csv", index=False)
    figure = px.scatter(
        optima,
        x="max_simulated_epochs",
        y="predicted_bpb",
        color="model",
        symbol="policy",
        facet_col="dataset",
        size="fit_support_distance",
        hover_data=["max_bucket_weight", "phase_total_variation", "aggregate_hhi", "fit_support_distance"],
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Unregularized surrogate optima: predicted value versus exposure pathology",
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(args.output_dir / "raw_optimum_audit.html", include_plotlyjs="cdn")
    (args.output_dir / "report.md").write_text(
        "# Raw-optimum audit\n\n"
        "No deployment penalty, KL term, trust region, or heldout calibration is applied.\n\n"
        "## Full-fit optima\n\n"
        + optima.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Fit-fold stability\n\n"
        + summary.to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )
    print(optima.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
