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
"""Benchmark second-order centered phase-response residuals.

The first-order centered residual has two scalar ordering coordinates: the
change in late benefit and the change in late overexposure penalty relative to
the aggregate-matched tied schedule. This benchmark adds a positive
semidefinite quadratic correction in those same coordinates. All added terms
remain exactly zero for tied schedules.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import nnls
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "second_order_centered_recency_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class CurvatureKind(StrEnum):
    """Nested Taylor corrections to the first-order centered residual."""

    LINEAR = "linear"
    BENEFIT_ONLY = "benefit_only"
    HEADROOM = "headroom"
    BENEFIT_HEADROOM = "benefit_headroom"
    ISOTROPIC = "isotropic"
    DIAGONAL = "diagonal"
    PSD_DIRECTIONS = "psd_directions"


@dataclass(frozen=True)
class ResidualHead:
    """Scaled nonnegative residual head without an intercept."""

    kind: CurvatureKind
    coef: np.ndarray
    scale: np.ndarray
    l2: float


@dataclass(frozen=True)
class FittedCandidate:
    """Effective-exposure backbone plus centered Taylor correction."""

    backbone: Any
    residual: ResidualHead
    c0: np.ndarray
    c1: np.ndarray
    alpha0: float
    alpha1: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction = centered.coverage.predict(
            self.backbone,
            weights,
            self.alpha0,
            self.alpha1,
        )
        design = residual_design(self.backbone.base, weights, self.c0, self.c1, self.residual.kind)
        return np.asarray(prediction + (design / self.residual.scale) @ self.residual.coef, dtype=float)


def residual_design(
    model: Any,
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    kind: CurvatureKind,
) -> np.ndarray:
    """Build first- and second-order coordinates around the tied schedule."""
    first = centered.residual_design(
        model,
        weights,
        c0,
        c1,
        centered.ResidualKind.TIED,
    )
    if kind is CurvatureKind.BENEFIT_ONLY:
        return first[:, :1]
    if kind in (CurvatureKind.HEADROOM, CurvatureKind.BENEFIT_HEADROOM):
        e0 = weights[:, 0, :] * c0[None, :]
        e1 = weights[:, 1, :] * c1[None, :]
        total = e0 + e1
        tied_e1 = total * (c1 / (c0 + c1))[None, :]
        late_signal, _late_penalty = centered.late_response(model, e1)
        tied_signal, _tied_penalty = centered.late_response(model, tied_e1)
        total_signal, _total_penalty = centered.late_response(model, total)
        headroom = -((late_signal - tied_signal) * (1.0 - total_signal) @ model.benefit_coef)
        base = first if kind is CurvatureKind.HEADROOM else first[:, :1]
        return np.column_stack([base, headroom])
    if kind is CurvatureKind.LINEAR:
        return first
    if kind is CurvatureKind.ISOTROPIC:
        return np.column_stack([first, np.sum(first**2, axis=1)])
    if kind is CurvatureKind.DIAGONAL:
        return np.hstack([first, first**2])
    positive_direction = (first[:, 0] + first[:, 1]) ** 2
    negative_direction = (first[:, 0] - first[:, 1]) ** 2
    return np.column_stack([first, positive_direction, negative_direction])


def fit_residual(
    backbone: Any,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    kind: CurvatureKind,
    l2: float,
    alpha0: float,
    alpha1: float,
) -> ResidualHead:
    """Fit one standardized nonnegative Taylor correction."""
    design = residual_design(
        backbone.base,
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        kind,
    )
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-8)
    scaled = design / scale
    base_prediction = centered.coverage.predict(
        backbone,
        dataset.weights[indices],
        alpha0,
        alpha1,
    )
    target = dataset.y[indices] - base_prediction
    if l2 > 0.0:
        scaled = np.vstack([scaled, np.sqrt(l2) * np.eye(scaled.shape[1])])
        target = np.concatenate([target, np.zeros(scaled.shape[1], dtype=float)])
    coef, _residual = nnls(scaled, target, maxiter=20 * scaled.shape[1])
    return ResidualHead(
        kind=kind,
        coef=np.asarray(coef, dtype=float),
        scale=np.asarray(scale, dtype=float),
        l2=l2,
    )


def model_label(kind: CurvatureKind, l2: float) -> str:
    return f"centered_{kind.value}_l2_{l2:g}"


def benchmark_seed(
    dataset: pooled.Dataset,
    seed: int,
    n_splits: int,
    l2_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate nested curvature families with a shared fold-local backbone."""
    folds = centered.folds_for(dataset, seed, n_splits)
    labels = [model_label(kind, l2) for kind in CurvatureKind for l2 in l2_values]
    predictions = {label: np.zeros(dataset.n, dtype=float) for label in labels}
    parameter_rows: list[dict[str, Any]] = []
    for fold_id, (train_indices, test_indices) in enumerate(folds):
        print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
        alpha0, alpha1 = centered.phase_fractions(dataset)
        backbone = centered.fit_backbone(dataset, train_indices)
        for kind in CurvatureKind:
            for l2 in l2_values:
                residual = fit_residual(
                    backbone,
                    dataset,
                    train_indices,
                    kind,
                    l2,
                    alpha0,
                    alpha1,
                )
                candidate = FittedCandidate(
                    backbone=backbone,
                    residual=residual,
                    c0=dataset.c0,
                    c1=dataset.c1,
                    alpha0=alpha0,
                    alpha1=alpha1,
                )
                label = model_label(kind, l2)
                predictions[label][test_indices] = candidate.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "fold": fold_id,
                        "model": label,
                        "l2": l2,
                        "coef": residual.coef.tolist(),
                        "scale": residual.scale.tolist(),
                        "nonzero_coef": int(np.sum(residual.coef > 1e-12)),
                    }
                )
    metric_rows = []
    for label, prediction in predictions.items():
        row = asdict(pooled.metrics(dataset, label, seed, prediction, folds))
        kind_name = label.removeprefix("centered_").split("_l2_", maxsplit=1)[0]
        extra = {
            CurvatureKind.LINEAR.value: 2,
            CurvatureKind.BENEFIT_ONLY.value: 1,
            CurvatureKind.HEADROOM.value: 3,
            CurvatureKind.BENEFIT_HEADROOM.value: 2,
            CurvatureKind.ISOTROPIC.value: 3,
            CurvatureKind.DIAGONAL.value: 4,
            CurvatureKind.PSD_DIRECTIONS.value: 4,
        }[kind_name]
        row["nominal_param_count"] = 4 * dataset.m + 4 + extra
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def fit_full_candidate(
    dataset: pooled.Dataset,
    kind: CurvatureKind,
    l2: float,
) -> FittedCandidate:
    indices = np.arange(dataset.n)
    alpha0, alpha1 = centered.phase_fractions(dataset)
    backbone = centered.fit_backbone(dataset, indices)
    residual = fit_residual(backbone, dataset, indices, kind, l2, alpha0, alpha1)
    return FittedCandidate(
        backbone=backbone,
        residual=residual,
        c0=dataset.c0,
        c1=dataset.c1,
        alpha0=alpha0,
        alpha1=alpha1,
    )


def starcoder_slice_metrics(
    dataset: pooled.Dataset,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate full-fit response shape on the dense StarCoder slice."""
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    rows = []
    for row in selected.itertuples():
        kind_name = str(row.model).removeprefix("centered_").split("_l2_", maxsplit=1)[0]
        l2 = float(str(row.model).rsplit("_", maxsplit=1)[-1])
        candidate = fit_full_candidate(dataset, CurvatureKind(kind_name), l2)
        prediction = candidate.predict(dataset.weights[mask])
        targets = dataset.y[mask]
        phase1 = dataset.frame.loc[mask, "phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": row.model,
                "slice_rows": int(mask.sum()),
                "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "slice_spearman": float(spearmanr(targets, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=(f"{centered.STARCODER_NAME},300m_uncheatable," "300m_table9,production_uncheatable"),
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--l2-values", default="0,0.01,0.1,1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    dataset_names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = pooled.parse_int_list(args.seeds)
    l2_values = pooled.parse_float_list(args.l2_values)
    metric_frames = []
    parameter_frames = []
    for dataset_name in dataset_names:
        dataset = datasets[dataset_name]
        for seed in seeds:
            metrics, parameters = benchmark_seed(dataset, seed, args.n_splits, l2_values)
            metric_frames.append(metrics)
            parameter_frames.append(parameters)
    metrics = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    selected = summary.loc[summary.groupby("dataset")["oof_rmse_mean"].idxmin()].copy()
    slices = pd.DataFrame()
    if centered.STARCODER_NAME in dataset_names:
        starcoder_selected = selected.loc[selected["dataset"].eq(centered.STARCODER_NAME)]
        slices = starcoder_slice_metrics(datasets[centered.STARCODER_NAME], starcoder_selected)
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    selected.to_csv(args.output_dir / "selected_configs.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameter_diagnostics.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    figure = px.scatter(
        summary,
        x="oof_rmse_mean",
        y="oof_spearman_mean",
        color="model",
        facet_col="dataset",
        hover_data=["nominal_param_count", "fold_mean_regret_at_1_mean", "low_tail_rmse_mean"],
        title="First- versus second-order centered phase response",
    )
    figure.write_html(
        args.output_dir / "cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# Second-order centered phase-response benchmark",
        "",
        "All residual terms are zero for aggregate-matched tied schedules. Quadratic variants add positive "
        "curvature in the two first-order ordering coordinates.",
        "",
        "## Selected configurations",
        "",
        selected.to_markdown(index=False),
        "",
        "## All configurations",
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False) if not slices.empty else "Not evaluated.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(selected.to_string(index=False))
    if not slices.empty:
        print(slices.to_string(index=False))
    print(f"Wrote second-order benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
