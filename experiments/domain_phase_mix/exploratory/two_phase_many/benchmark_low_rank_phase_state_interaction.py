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
"""Add a low-rank cross-phase interaction between learned bucket states.

Separate heads model phase responses additively. This benchmark represents the
smallest non-diagonal state transition:

    L = L_sep + sum_j (s0 @ u_j) (s1 @ v_j),

where ``s0`` and ``s1`` are bounded DSP learned-state features. Rank one adds
two coefficients per bucket yet can express cross-bucket transfer or
interference between early and late phase states. Alternating constrained ridge
keeps the separate-head coefficients nonnegative and the interaction signed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "low_rank_phase_state_interaction_20260710"
BASE_L2 = 0.1
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Config:
    """Bilinear rank and symmetric factor ridge."""

    rank: int
    interaction_l2: float

    @property
    def name(self) -> str:
        if self.rank == 0:
            return "separate_heads"
        return f"phase_state_rank_{self.rank}_l2_{self.interaction_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Separate phase bowls plus a low-rank state interaction."""

    config: Config
    mu0: np.ndarray
    mu1: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    intercept: float
    base_coef: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.mu0)

    @property
    def parameter_count(self) -> int:
        return (4 + 2 * self.config.rank) * self.num_domains + 1


def states(exposure: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return bounded learned-state coordinates."""
    return -np.expm1(-np.maximum(exposure * rho[None, :], 0.0))


def interaction_prediction(state0: np.ndarray, state1: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate a rank-r bilinear state interaction."""
    if left.size == 0:
        return np.zeros(len(state0), dtype=float)
    return np.sum((state0 @ left.T) * (state1 @ right.T), axis=1)


def solve_linear_block(
    base_design: np.ndarray,
    state: np.ndarray,
    multiplier: np.ndarray,
    target: np.ndarray,
    interaction_l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit nonnegative base heads and one signed bilinear factor block."""
    rank = multiplier.shape[1]
    interaction = np.hstack([state * multiplier[:, component : component + 1] for component in range(rank)])
    design = np.hstack([base_design, interaction])
    center = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = design - center
    centered_target = target - target_mean
    penalties = np.concatenate(
        [
            np.full(base_design.shape[1], np.sqrt(BASE_L2), dtype=float),
            np.full(interaction.shape[1], np.sqrt(interaction_l2), dtype=float),
        ]
    )
    augmented_design = np.vstack([centered_design, np.diag(penalties)])
    augmented_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    lower = np.concatenate([np.zeros(base_design.shape[1]), np.full(interaction.shape[1], -np.inf)])
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(lower, np.full(design.shape[1], np.inf)),
        method="trf",
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not result.success:
        raise RuntimeError(f"Bilinear block fit failed: {result.message}")
    coef = np.asarray(result.x, dtype=float)
    intercept = target_mean - float((center @ coef).item())
    factors = coef[base_design.shape[1] :].reshape(rank, state.shape[1])
    return intercept, coef[: base_design.shape[1]], factors


def balanced_factors(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve bilinear scale ambiguity by equalizing factor norms."""
    out_left = left.copy()
    out_right = right.copy()
    for component in range(len(left)):
        left_norm = max(float(np.linalg.norm(out_left[component])), 1e-12)
        right_norm = max(float(np.linalg.norm(out_right[component])), 1e-12)
        scale = np.sqrt(right_norm / left_norm)
        out_left[component] *= scale
        out_right[component] /= scale
    return out_left, out_right


def fit_from_start(
    base_design: np.ndarray,
    state0: np.ndarray,
    state1: np.ndarray,
    target: np.ndarray,
    config: Config,
    initial_right: np.ndarray,
    max_iterations: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Alternate factor blocks until prediction error stabilizes."""
    right = initial_right.copy()
    left = np.zeros_like(right)
    intercept = float(target.mean())
    base_coef = np.zeros(base_design.shape[1], dtype=float)
    previous = np.inf
    for _iteration in range(max_iterations):
        intercept, base_coef, left = solve_linear_block(
            base_design,
            state0,
            state1 @ right.T,
            target,
            config.interaction_l2,
        )
        _ignored_intercept, _ignored_base, right = solve_linear_block(
            base_design,
            state1,
            state0 @ left.T,
            target,
            config.interaction_l2,
        )
        left, right = balanced_factors(left, right)
        prediction = intercept + base_design @ base_coef + interaction_prediction(state0, state1, left, right)
        objective = float(
            np.mean((prediction - target) ** 2)
            + config.interaction_l2 * (np.sum(left**2) + np.sum(right**2)) / len(target)
        )
        if np.isfinite(previous) and previous - objective <= 1e-9 * max(previous, 1.0):
            break
        previous = objective
    interaction = interaction_prediction(state0, state1, left, right)
    intercept, base_coef = pooled.fit_raw_nnls(base_design, target - interaction, BASE_L2)
    prediction = intercept + base_design @ base_coef + interaction
    objective = float(
        np.mean((prediction - target) ** 2) + config.interaction_l2 * (np.sum(left**2) + np.sum(right**2)) / len(target)
    )
    return intercept, base_coef, left, right, objective


def initial_factors(rank: int, num_domains: int, seed: int) -> np.ndarray:
    """Build a deterministic orthonormal interaction start."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(num_domains, rank))
    q, _r = np.linalg.qr(matrix)
    return q[:, :rank].T


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    *,
    max_iterations: int,
    starts: int,
) -> FittedModel:
    """Fit separate heads and the best deterministic bilinear start."""
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    target = dataset.y[indices]
    mu0 = pooled.selected_mu(exposure0, target)
    mu1 = pooled.selected_mu(exposure1, target)
    rho0 = recency.channel_base(exposure0).rho
    rho1 = recency.channel_base(exposure1).rho
    base_design = np.hstack([pooled.bowl_design(exposure0, mu0), pooled.bowl_design(exposure1, mu1)])
    state0 = states(exposure0, rho0)
    state1 = states(exposure1, rho1)
    if config.rank == 0:
        intercept, base_coef = pooled.fit_raw_nnls(base_design, target, BASE_L2)
        return FittedModel(
            config=config,
            mu0=mu0,
            mu1=mu1,
            rho0=rho0,
            rho1=rho1,
            intercept=intercept,
            base_coef=base_coef,
            left=np.empty((0, dataset.m)),
            right=np.empty((0, dataset.m)),
        )
    candidates = []
    for seed in range(starts):
        candidates.append(
            fit_from_start(
                base_design,
                state0,
                state1,
                target,
                config,
                initial_factors(config.rank, dataset.m, seed),
                max_iterations,
            )
        )
    intercept, base_coef, left, right, _objective = min(candidates, key=lambda candidate: candidate[-1])
    return FittedModel(
        config=config,
        mu0=mu0,
        mu1=mu1,
        rho0=rho0,
        rho1=rho1,
        intercept=intercept,
        base_coef=base_coef,
        left=left,
        right=right,
    )


def predict(model: FittedModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    """Predict held-out rows."""
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    base_design = np.hstack(
        [
            pooled.bowl_design(exposure0, model.mu0),
            pooled.bowl_design(exposure1, model.mu1),
        ]
    )
    return np.asarray(
        model.intercept
        + base_design @ model.base_coef
        + interaction_prediction(
            states(exposure0, model.rho0),
            states(exposure1, model.rho1),
            model.left,
            model.right,
        ),
        dtype=float,
    )


def configs(ranks: list[int], l2_values: list[float]) -> list[Config]:
    """Return additive control and low-rank sweep."""
    out = [Config(0, BASE_L2)]
    out.extend(Config(rank, l2) for rank in ranks if rank > 0 for l2 in l2_values)
    return out


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    seeds: list[int],
    n_splits: int,
    *,
    max_iterations: int,
    starts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = centered.folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = fit_model(
                    dataset,
                    train_indices,
                    config,
                    max_iterations=max_iterations,
                    starts=starts,
                )
                predictions[config.name][test_indices] = predict(model, dataset, test_indices)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "left_norm": float(np.linalg.norm(model.left)),
                        "right_norm": float(np.linalg.norm(model.right)),
                    }
                )
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = (4 + 2 * config.rank) * dataset.m + 1
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    *,
    max_iterations: int,
    starts: int,
) -> pd.DataFrame:
    """Measure full-fit response on the dense phase-0-Nemotron slice."""
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    rows = []
    for config in model_configs:
        model = fit_model(dataset, np.arange(dataset.n), config, max_iterations=max_iterations, starts=starts)
        prediction = predict(model, dataset, indices)
        target = dataset.y[indices]
        phase1 = dataset.frame.iloc[indices]["phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": config.name,
                "slice_rows": len(indices),
                "slice_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
                "slice_spearman": float(spearmanr(target, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm comparison."""
    long = summary.melt(
        id_vars=["dataset", "model"],
        value_vars=["oof_rmse_mean", "oof_spearman_mean", "fold_mean_regret_at_1_mean"],
        var_name="metric",
        value_name="value",
    )
    figure = px.bar(
        long,
        x="model",
        y="value",
        color="model",
        facet_row="dataset",
        facet_col="metric",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Low-rank cross-phase learned-state interaction",
    )
    figure.update_layout(showlegend=False, height=1000)
    figure.write_html(output_dir / "crossswarm_cv_comparison.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--ranks", default="1,2")
    parser.add_argument("--interaction-l2-values", default="0.01,0.1,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = configs(pooled.parse_int_list(args.ranks), pooled.parse_float_list(args.interaction_l2_values))
    metric_frames = []
    parameter_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            max_iterations=args.max_iterations,
            starts=args.starts,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = starcoder_slice_summary(
        datasets[centered.STARCODER_NAME],
        model_configs,
        max_iterations=args.max_iterations,
        starts=args.starts,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Low-rank cross-phase learned-state interaction",
        "",
        "The bilinear interaction is the smallest non-diagonal state-transition correction to separate heads.",
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(slices.to_string(index=False))
    print(f"Wrote low-rank phase-state benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
