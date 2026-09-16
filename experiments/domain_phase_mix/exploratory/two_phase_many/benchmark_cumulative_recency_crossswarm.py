# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

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
"""Benchmark cumulative-learning plus recency DSP across three swarms.

This is the cross-swarm guardrail for the model introduced by
``surrogate_search/benchmark_cumulative_recency_starcoder.py``. It compares:

* cumulative exposure only; and
* cumulative exposure plus a phase-1 recency residual.

The nonlinear response shapes and linear heads are refit inside every reported
CV fold. A separate fixed-shape screen chooses the linear-head ridge penalty;
screening metrics are persisted but are not reported as final model quality.

The 300M benchmark uses all matched train and one-phase rows (569 rows) with
correspondence-grouped folds. Two-phase intervention rows remain an external
test set. Production-swarm Uncheatable uses the 840-row production panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "cumulative_recency_crossswarm_20260710"
REFERENCE_300M = pooled.REFERENCE_OUTPUTS / "mechanistic_phase_backbone_300m_multiseed_20260710/cv_summary.csv"
REFERENCE_PRODUCTION = (
    pooled.REFERENCE_OUTPUTS / "mechanistic_phase_backbone_production_multiseed_20260710/cv_summary.csv"
)
REFERENCE_EXTERNAL = (
    pooled.REFERENCE_OUTPUTS
    / "mechanistic_phase_backbone_300m_multiseed_20260710/external_two_phase_heldout_summary.csv"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
SHAPE_BOUND = 3.0


class ModelKind(StrEnum):
    """Nested cumulative-exposure model variants."""

    CUMULATIVE_ONLY = "cumulative_only"
    CUMULATIVE_RECENCY = "cumulative_recency"


@dataclass(frozen=True)
class FittedModel:
    """One fitted cumulative-exposure model."""

    kind: ModelKind
    cumulative_base: recency.ChannelBase
    recency_base: recency.ChannelBase | None
    shape_offsets: np.ndarray
    intercept: float
    coef: np.ndarray
    l2: float
    c0: np.ndarray
    c1: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.c0)

    @property
    def parameter_count(self) -> int:
        if self.kind is ModelKind.CUMULATIVE_ONLY:
            return 2 * self.num_domains + 3
        return 4 * self.num_domains + 5

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.kind,
            self.cumulative_base,
            self.recency_base,
            self.shape_offsets,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def fit_head(design: np.ndarray, targets: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    """Fit a centered nonnegative ridge head with a dimension-aware limit."""
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_targets = np.concatenate([centered_targets, np.zeros(design.shape[1], dtype=float)])
    coef, _residual = nnls(
        centered_design,
        centered_targets,
        maxiter=20 * design.shape[1],
    )
    intercept = target_mean - float((design_mean @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    kind: ModelKind,
    cumulative_base: recency.ChannelBase,
    recency_base: recency.ChannelBase | None,
    shape_offsets: np.ndarray,
) -> np.ndarray:
    """Build cumulative and optional recency response features."""
    cumulative, late = recency.exposures(weights, c0, c1)
    cumulative_rho, cumulative_tau = recency.shifted_shape(cumulative_base, shape_offsets[0], shape_offsets[1])
    cumulative_benefit, cumulative_penalty = recency.channel_features(cumulative, cumulative_rho, cumulative_tau)
    cumulative_design = np.hstack([-cumulative_benefit, cumulative_penalty])
    if kind is ModelKind.CUMULATIVE_ONLY:
        return cumulative_design
    if recency_base is None:
        raise ValueError("The recency model requires a recency channel base")
    late_rho, late_tau = recency.shifted_shape(recency_base, shape_offsets[2], shape_offsets[3])
    late_benefit, late_penalty = recency.channel_features(late, late_rho, late_tau)
    return np.hstack([cumulative_design, -late_benefit, late_penalty])


def shape_starts(kind: ModelKind) -> tuple[np.ndarray, ...]:
    """Return deterministic global-shape starts for one model kind."""
    if kind is ModelKind.CUMULATIVE_RECENCY:
        return recency.shape_starts()
    starts = {(float(start[0]), float(start[1])) for start in recency.shape_starts()}
    return tuple(np.asarray(start, dtype=float) for start in sorted(starts))


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    kind: ModelKind,
    l2: float,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    """Fit one model by profiling the nonnegative linear head."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    cumulative, late = recency.exposures(weights, dataset.c0, dataset.c1)
    cumulative_base = recency.channel_base(cumulative)
    recency_base = recency.channel_base(late) if kind is ModelKind.CUMULATIVE_RECENCY else None

    def objective(shape_offsets: np.ndarray) -> float:
        design = design_matrix(
            weights,
            dataset.c0,
            dataset.c1,
            kind,
            cumulative_base,
            recency_base,
            np.asarray(shape_offsets, dtype=float),
        )
        intercept, coef = fit_head(design, targets, l2)
        prediction = intercept + design @ coef
        return float(np.sqrt(np.mean((prediction - targets) ** 2)))

    starts = shape_starts(kind)
    scored_starts = sorted(((objective(start), start) for start in starts), key=lambda item: item[0])
    selected_starts = [start for _score, start in scored_starts[:coarse_top_k]]
    bounds = [(-SHAPE_BOUND, SHAPE_BOUND)] * len(selected_starts[0])
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 30},
        )
        for start in selected_starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    shape_offsets = np.asarray(best.x, dtype=float)
    design = design_matrix(
        weights,
        dataset.c0,
        dataset.c1,
        kind,
        cumulative_base,
        recency_base,
        shape_offsets,
    )
    intercept, coef = fit_head(design, targets, l2)
    return FittedModel(
        kind=kind,
        cumulative_base=cumulative_base,
        recency_base=recency_base,
        shape_offsets=shape_offsets,
        intercept=intercept,
        coef=coef,
        l2=l2,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
    )


def folds_for(dataset: pooled.Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame.columns:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    return pooled.dataset_folds(dataset, seed, n_splits)


def screen_l2(
    dataset: pooled.Dataset,
    kind: ModelKind,
    l2_values: list[float],
    seeds: list[int],
    n_splits: int,
    *,
    shape_l2: float,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[float, pd.DataFrame]:
    """Choose head regularization using a fixed full-data shape anchor."""
    all_indices = np.arange(dataset.n)
    anchor = fit_model(
        dataset,
        all_indices,
        kind,
        shape_l2,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    full_design = design_matrix(
        dataset.weights,
        dataset.c0,
        dataset.c1,
        kind,
        anchor.cumulative_base,
        anchor.recency_base,
        anchor.shape_offsets,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = folds_for(dataset, seed, n_splits)
        predictions = {l2: np.zeros(dataset.n, dtype=float) for l2 in l2_values}
        for train_indices, test_indices in folds:
            for l2 in l2_values:
                intercept, coef = fit_head(full_design[train_indices], dataset.y[train_indices], l2)
                predictions[l2][test_indices] = intercept + full_design[test_indices] @ coef
        for l2 in l2_values:
            metric = pooled.metrics(dataset, f"{kind.value}_screen", seed, predictions[l2], folds)
            row = asdict(metric)
            row["l2"] = l2
            row["screen_shape_l2"] = shape_l2
            rows.append(row)
    frame = pd.DataFrame(rows)
    selected_l2 = float(frame.groupby("l2")["oof_rmse"].mean().idxmin())
    return selected_l2, frame


def benchmark_selected(
    dataset: pooled.Dataset,
    kind: ModelKind,
    l2: float,
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit outer-fold evaluation for one selected configuration."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = folds_for(dataset, seed, n_splits)
        prediction = np.zeros(dataset.n, dtype=float)
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(
                f"{dataset.name}/{kind.value}: seed={seed} " f"fold={fold_id + 1}/{n_splits} l2={l2:g}",
                flush=True,
            )
            model = fit_model(
                dataset,
                train_indices,
                kind,
                l2,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            prediction[test_indices] = model.predict(dataset.weights[test_indices])
            parameter_rows.append(
                {
                    "dataset": dataset.name,
                    "model": kind.value,
                    "seed": seed,
                    "fold": fold_id,
                    "l2": l2,
                    "shape_offsets": json.dumps(model.shape_offsets.tolist()),
                    "coef_norm": float(np.linalg.norm(model.coef)),
                    "coef_max": float(np.max(model.coef)),
                    "nonzero_coef": int(np.sum(model.coef > 1e-10)),
                }
            )
        row = asdict(pooled.metrics(dataset, kind.value, seed, prediction, folds))
        row["nominal_param_count"] = 2 * dataset.m + 3 if kind is ModelKind.CUMULATIVE_ONLY else 4 * dataset.m + 5
        row["selected_l2"] = l2
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def load_datasets() -> tuple[dict[str, pooled.Dataset], dict[str, pooled.Dataset]]:
    """Load joint 300M fit panels, external interventions, and production."""
    frame = pd.read_csv(joint.PACKET)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, joint.ONE_PHASE_SOURCE, domains)
    datasets: dict[str, pooled.Dataset] = {}
    external: dict[str, pooled.Dataset] = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        datasets[f"300m_{objective}"] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        external[f"300m_{objective}"] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
    production = pooled.load_production_dataset()
    datasets[production.name] = production
    return datasets, external


def external_evaluation(
    dataset: pooled.Dataset,
    external: pooled.Dataset,
    kind: ModelKind,
    l2: float,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> dict[str, Any]:
    model = fit_model(
        dataset,
        np.arange(dataset.n),
        kind,
        l2,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    row = joint.external_metrics(kind.value, external.y, model.predict(external.weights))
    row["dataset"] = dataset.name
    row["external_rows"] = external.n
    row["selected_l2"] = l2
    return row


def checkpoint_paths(output_dir: Path, dataset_name: str, kind: ModelKind) -> dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}__{kind.value}"
    return {
        "selection": checkpoint_dir / f"{stem}__selection.json",
        "screen": checkpoint_dir / f"{stem}__screen.csv",
        "metrics": checkpoint_dir / f"{stem}__metrics.csv",
        "parameters": checkpoint_dir / f"{stem}__parameters.csv",
        "external": checkpoint_dir / f"{stem}__external.json",
    }


def checkpoint_complete(paths: dict[str, Path], has_external: bool) -> bool:
    required = [paths["selection"], paths["screen"], paths["metrics"], paths["parameters"]]
    if has_external:
        required.append(paths["external"])
    return all(path.exists() for path in required)


def reference_cv_rows() -> pd.DataFrame:
    """Load established effective-exposure and current-frontier CV rows."""
    frames = [pd.read_csv(REFERENCE_300M), pd.read_csv(REFERENCE_PRODUCTION)]
    reference = pd.concat(frames, ignore_index=True)
    return reference.loc[reference["model"].isin(["effective_exposure", "split_saturation_penalty_geometry"])].copy()


def write_plot(comparison: pd.DataFrame, output_dir: Path) -> None:
    long = comparison.melt(
        id_vars=["dataset", "model"],
        value_vars=[
            "oof_rmse_mean",
            "oof_spearman_mean",
            "fold_mean_regret_at_1_mean",
            "lower_tail_optimism_mean",
        ],
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
        title="Cumulative plus recency DSP: cross-swarm guardrail",
    )
    figure.update_layout(showlegend=False, height=950)
    figure.update_xaxes(tickangle=-25)
    figure.write_html(
        output_dir / "crossswarm_cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def report_text(
    selected_l2: pd.DataFrame,
    comparison: pd.DataFrame,
    external: pd.DataFrame,
) -> str:
    verdict = []
    for dataset_name in (
        "300m_uncheatable",
        "300m_table9",
        "production_uncheatable",
    ):
        recency_row = comparison.loc[
            comparison["dataset"].eq(dataset_name) & comparison["model"].eq(ModelKind.CUMULATIVE_RECENCY.value)
        ].iloc[0]
        baseline_row = comparison.loc[
            comparison["dataset"].eq(dataset_name) & comparison["model"].eq("effective_exposure")
        ].iloc[0]
        rmse_change = float(recency_row["oof_rmse_mean"]) / float(baseline_row["oof_rmse_mean"]) - 1.0
        verdict.append(
            f"- **{dataset_name}:** RMSE {rmse_change:+.1%} vs effective exposure; "
            f"Spearman {float(recency_row['oof_spearman_mean']):.3f} vs "
            f"{float(baseline_row['oof_spearman_mean']):.3f}; fold Regret@1 "
            f"{float(recency_row['fold_mean_regret_at_1_mean']):.4f} vs "
            f"{float(baseline_row['fold_mean_regret_at_1_mean']):.4f}."
        )
    lines = [
        "# Cumulative-learning plus recency DSP: cross-swarm guardrail",
        "",
        "## Protocol",
        "",
        "- 300M CV uses the 569-row joint train + matched one-phase panel and correspondence-grouped folds.",
        "- Production CV uses all 840 completed production-swarm rows.",
        "- The fixed-shape L2 screen is used only for hyperparameter choice. Every reported CV fold refits nonlinear shapes and the linear head using training rows only.",
        "- The 300M external evaluation uses two-phase intervention rows excluded from fitting and hyperparameter screening.",
        "- Established effective-exposure and split-plus-geometry rows come from the matched July 10 benchmark with the same datasets, folds, and seeds.",
        "",
        "## Verdict",
        "",
        *verdict,
        "- The recency channel is a large improvement over cumulative-only on all three datasets and improves both 300M objectives over effective exposure on OOF RMSE and Spearman.",
        "- It **fails the production guardrail**: production RMSE, rank correlation, and selection regret are substantially worse than effective exposure. The form is therefore diagnostic evidence, not a replacement model.",
        "- On external 300M interventions it selects the observed best row for both objectives. It improves all reported Uncheatable metrics over effective exposure; for Table-9 it improves rank, Regret@1, and tail diagnostics but has slightly worse global RMSE.",
        "- The likely scaling failure is the deterministic per-domain timescale: only four global shape shifts are learned, which is too rigid across 168 heterogeneous production buckets. A next model should hierarchically learn or pool domain response timescales while retaining the cumulative-plus-recency decomposition.",
        "",
        "## Selected regularization",
        "",
        selected_l2.to_markdown(index=False),
        "",
        "## Refit CV comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## External two-phase interventions",
        "",
        external.to_markdown(index=False),
        "",
        "## Interpretation rule",
        "",
        "The recency term is retained only if it improves the cumulative-only ablation without materially regressing established effective-exposure DSP on RMSE, rank correlation, selection regret, or lower-tail optimism. External intervention performance is the stronger 300M test because those rows never enter fitting or L2 selection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default="300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--l2-values", default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--shape-l2", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, external_datasets = load_datasets()
    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    seeds = pooled.parse_int_list(args.seeds)
    l2_values = pooled.parse_float_list(args.l2_values)

    screen_frames = []
    cv_frames = []
    parameter_frames = []
    external_rows = []
    selected_rows = []
    for dataset_name in selected_names:
        dataset = datasets[dataset_name]
        for kind in ModelKind:
            paths = checkpoint_paths(args.output_dir, dataset.name, kind)
            has_external = dataset.name in external_datasets
            if checkpoint_complete(paths, has_external):
                print(f"Loading checkpoint {dataset.name}/{kind.value}", flush=True)
                selection = json.loads(paths["selection"].read_text())
                selected_l2 = float(selection["selected_l2"])
                screen = pd.read_csv(paths["screen"])
                metrics = pd.read_csv(paths["metrics"])
                parameters = pd.read_csv(paths["parameters"])
                external_row = json.loads(paths["external"].read_text()) if has_external else None
            else:
                print(f"Screening {dataset.name}/{kind.value}", flush=True)
                selected_l2, screen = screen_l2(
                    dataset,
                    kind,
                    l2_values,
                    seeds,
                    args.n_splits,
                    shape_l2=args.shape_l2,
                    maxiter=args.maxiter,
                    coarse_top_k=args.coarse_top_k,
                )
                screen["dataset"] = dataset.name
                screen["model"] = kind.value
                metrics, parameters = benchmark_selected(
                    dataset,
                    kind,
                    selected_l2,
                    seeds,
                    args.n_splits,
                    maxiter=args.maxiter,
                    coarse_top_k=args.coarse_top_k,
                )
                external_row = (
                    external_evaluation(
                        dataset,
                        external_datasets[dataset.name],
                        kind,
                        selected_l2,
                        maxiter=args.maxiter,
                        coarse_top_k=args.coarse_top_k,
                    )
                    if has_external
                    else None
                )
                paths["selection"].write_text(
                    json.dumps(
                        {
                            "dataset": dataset.name,
                            "model": kind.value,
                            "selected_l2": selected_l2,
                        },
                        indent=2,
                    )
                )
                screen.to_csv(paths["screen"], index=False)
                metrics.to_csv(paths["metrics"], index=False)
                parameters.to_csv(paths["parameters"], index=False)
                if external_row is not None:
                    paths["external"].write_text(json.dumps(external_row, indent=2))
            screen_frames.append(screen)
            selected_rows.append({"dataset": dataset.name, "model": kind.value, "selected_l2": selected_l2})
            cv_frames.append(metrics)
            parameter_frames.append(parameters)
            if external_row is not None:
                external_rows.append(external_row)

    screen = pd.concat(screen_frames, ignore_index=True)
    raw = pd.concat(cv_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    summary = pooled.summarize(raw)
    comparison = pd.concat([summary, reference_cv_rows()], ignore_index=True, sort=False)
    external = pd.DataFrame(external_rows)
    reference_external = pd.read_csv(REFERENCE_EXTERNAL)
    reference_external = reference_external.loc[
        reference_external["model"].isin(["effective_exposure", "split_saturation_penalty_geometry"])
    ]
    external_comparison = pd.concat([external, reference_external], ignore_index=True, sort=False)

    screen.to_csv(args.output_dir / "l2_screen_by_seed.csv", index=False)
    selected.to_csv(args.output_dir / "selected_l2.csv", index=False)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameter_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "cv_comparison_with_references.csv", index=False)
    external_comparison.to_csv(args.output_dir / "external_two_phase_comparison.csv", index=False)
    write_plot(comparison, args.output_dir)
    (args.output_dir / "report.md").write_text(report_text(selected, comparison, external_comparison))
    print(comparison.to_string(index=False))
    print(external_comparison.to_string(index=False))
    print(f"Wrote cross-swarm guardrail artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
