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
"""Measure fixed-aggregate reorder stability across grouped refits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_anchor_ordering as anchor_order,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_optima as optimum_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_separate_heads_reorder as reorder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_calibrated_cumulative_recency as generalized,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_cumulative_recency_crossswarm as cross,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "generalized_power_reorder_stability_20260710"
CANDIDATE_DIR = pooled.REFERENCE_OUTPUTS / "generalized_power_reorder_panel_20260710" / "mixtures"
POWER_L2 = 0.001
ORDER_KL_VALUES = (0.3, 1.0)


def candidate_name(objective: str, order_kl: float) -> str:
    """Return the materialized candidate key for one objective and order penalty."""
    short = "unch" if objective == "uncheatable" else "t9"
    kl = "0p3" if np.isclose(order_kl, 0.3) else "1"
    return f"genpow_sep_{short}_okl{kl}"


def candidate_weights(objective: str, order_kl: float, domains: list[str]) -> np.ndarray:
    """Load one fixed full-panel candidate in dataset domain order."""
    path = CANDIDATE_DIR / f"{candidate_name(objective, order_kl)}.csv"
    return reorder.weights_from_frame(path, domains)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, treating two zero vectors as identical."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def objective_rows(
    dataset: pooled.Dataset,
    objective: str,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate full-panel candidates and refit-specific optima across folds."""
    config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, POWER_L2)
    incumbent = reorder.weights_from_frame(reorder.separate_heads_path(objective), dataset.domain_names)
    fixed_candidates = {
        order_kl: candidate_weights(objective, order_kl, dataset.domain_names) for order_kl in ORDER_KL_VALUES
    }
    pair_noise_sd = np.sqrt(2.0) * optimum_audit.NOISE_SD[objective]
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold_id, (train_indices, _test_indices) in enumerate(cross.folds_for(dataset, seed, n_splits)):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            model = generalized.fit_model(
                dataset,
                train_indices,
                config,
                generalized.HEAD_L2_BY_DATASET[dataset.name],
                maxiter=maxiter,
                coarse_top_k=1,
            )
            aggregate = model.alpha0 * incumbent[0] + model.alpha1 * incumbent[1]
            incumbent_prediction = float(model.predict(incumbent[None, :, :])[0])
            for order_kl, fixed_candidate in fixed_candidates.items():
                fixed_aggregate = model.alpha0 * fixed_candidate[0] + model.alpha1 * fixed_candidate[1]
                if not np.allclose(fixed_aggregate, aggregate, atol=1e-9):
                    raise ValueError(f"{candidate_name(objective, order_kl)} does not preserve aggregate exposure")
                fixed_prediction = float(model.predict(fixed_candidate[None, :, :])[0])
                refit_candidate = anchor_order.optimize_order(
                    model,
                    aggregate,
                    order_kl,
                    penalty_reference=incumbent,
                )
                refit_prediction = float(model.predict(refit_candidate[None, :, :])[0])
                fixed_delta = (fixed_candidate - incumbent).ravel()
                refit_delta = (refit_candidate - incumbent).ravel()
                fixed_gain = incumbent_prediction - fixed_prediction
                rows.append(
                    {
                        "dataset": dataset.name,
                        "objective": objective,
                        "order_kl_reg": order_kl,
                        "seed": seed,
                        "fold": fold_id,
                        "fixed_candidate_gain": fixed_gain,
                        "fixed_gain_diff_sd": fixed_gain / pair_noise_sd,
                        "refit_candidate_gain": incumbent_prediction - refit_prediction,
                        "refit_to_fixed_tv": float(0.5 * np.abs(refit_candidate - fixed_candidate).sum(axis=1).mean()),
                        "direction_cosine": cosine_similarity(fixed_delta, refit_delta),
                        "fixed_tv_to_incumbent": float(0.5 * np.abs(fixed_candidate - incumbent).sum(axis=1).mean()),
                    }
                )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize gain sign and phase-order stability across refits."""
    rows = []
    for (dataset, objective, order_kl), group in frame.groupby(["dataset", "objective", "order_kl_reg"]):
        rows.append(
            {
                "dataset": dataset,
                "objective": objective,
                "order_kl_reg": order_kl,
                "n_refits": len(group),
                "fixed_gain_mean": group["fixed_candidate_gain"].mean(),
                "fixed_gain_std": group["fixed_candidate_gain"].std(ddof=1),
                "fixed_gain_min": group["fixed_candidate_gain"].min(),
                "fixed_gain_positive_fraction": group["fixed_candidate_gain"].gt(0.0).mean(),
                "fixed_gain_ge_2sd_fraction": group["fixed_gain_diff_sd"].ge(2.0).mean(),
                "refit_gain_mean": group["refit_candidate_gain"].mean(),
                "refit_to_fixed_tv_mean": group["refit_to_fixed_tv"].mean(),
                "direction_cosine_mean": group["direction_cosine"].mean(),
                "direction_cosine_min": group["direction_cosine"].min(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    seeds = pooled.parse_int_list(args.seeds)
    rows = []
    for objective in ("uncheatable", "table9"):
        rows.extend(
            objective_rows(
                datasets[f"300m_{objective}"],
                objective,
                seeds,
                args.n_splits,
                args.maxiter,
            )
        )
    diagnostics = pd.DataFrame(rows)
    summary = summarize(diagnostics)
    diagnostics.to_csv(args.output_dir / "refit_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "refit_summary.csv", index=False)
    report = [
        "# Shared-power fixed-aggregate reorder stability",
        "",
        "Each grouped training-fold refit evaluates the fixed full-panel candidate "
        "and independently reoptimizes phase order at the same aggregate exposure.",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"Wrote reorder stability diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
