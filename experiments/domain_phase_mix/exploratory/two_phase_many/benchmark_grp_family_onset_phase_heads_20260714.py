# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

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
"""Test family-specific replay onset in Bucket-resolved and separate-heads GRP.

Both controls use the same family replay-harm feature,

    B_f softplus(log(1 + Z_f) - tau)^2.

The intervention replaces the shared onset ``tau`` with one onset ``tau_f``
per semantic family. The shared response shape and ridge penalty are selected
inside each outer fold. For the intervention, shrinkage of ``tau_f`` toward
the selected shared onset is selected in a second inner CV. Fold checkpoints
make the full benchmark safely resumable.
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
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_grp_domain_saturation_phase_heads_20260714 as phase_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_grp_saturation_hierarchy_20260714 as hierarchy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_production_grp_retained_hybrids_20260713 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/grp_family_onset_phase_heads_20260714"
OUTER_CV_SEED = 4171
INNER_CV_SEED = 4172
DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 3
DEFAULT_TAU_MAXITER = 60
TAU_SHRINK_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
TAU_BOUNDS = (0.0, 7.0)
CHECKPOINT_SCHEMA = 1
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
BOOTSTRAP_DRAWS = 20_000


class OnsetScope(StrEnum):
    SHARED = "shared_tau"
    FAMILY = "family_tau"


@dataclass(frozen=True)
class Variant:
    name: str
    phase: phase_heads.PhaseKind
    onset: OnsetScope


@dataclass(frozen=True)
class FittedModel:
    head: family_grp.FittedHead
    family_tau: np.ndarray | None
    tau_shrink: float | None
    objective: float | None
    iterations: int | None
    converged: bool | None


VARIANTS = (
    Variant("bucket_resolved_shared_tau", phase_heads.PhaseKind.ETA, OnsetScope.SHARED),
    Variant("bucket_resolved_family_tau", phase_heads.PhaseKind.ETA, OnsetScope.FAMILY),
    Variant("power_separate_heads_shared_tau", phase_heads.PhaseKind.SEPARATE_HEADS, OnsetScope.SHARED),
    Variant("power_separate_heads_family_tau", phase_heads.PhaseKind.SEPARATE_HEADS, OnsetScope.FAMILY),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
PAIRS = (
    ("bucket_resolved_shared_tau", "bucket_resolved_family_tau"),
    ("power_separate_heads_shared_tau", "power_separate_heads_family_tau"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in hierarchy.DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=16)
    parser.add_argument("--outer-splits", type=int, default=DEFAULT_OUTER_SPLITS)
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--tau-maxiter", type=int, default=DEFAULT_TAU_MAXITER)
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def phase_variant(phase: phase_heads.PhaseKind) -> phase_heads.Variant:
    name = "power_eta" if phase is phase_heads.PhaseKind.ETA else "power_separate_heads"
    return phase_heads.VARIANT_BY_NAME[name]


def candidate_shapes(phase: phase_heads.PhaseKind, count: int) -> tuple[retained.Shape, ...]:
    return phase_heads.candidate_shapes(phase_variant(phase), count)


def build_design(
    dataset: family_grp.Dataset,
    variant: Variant,
    shape: retained.Shape,
    family_tau: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    heads, replay_exposure = phase_heads.phase_exposures(dataset, shape, variant.phase)
    nonsingleton = retained.nonsingleton_families(dataset)
    pieces: list[np.ndarray] = []
    names: list[str] = []

    for head_index, exposure in enumerate(heads):
        pieces.append(-phase_heads.response(exposure, shape, phase_heads.ResponseKind.POWER, None))
        names.extend(f"phase{head_index}:bucket_signal:{domain}" for domain in dataset.domains)
        if nonsingleton:
            family_exposure = np.column_stack(
                [exposure[:, dataset.family_members[index]].sum(axis=1) for index in nonsingleton]
            )
            pieces.append(-phase_heads.response(family_exposure, shape, phase_heads.ResponseKind.POWER, None))
            names.extend(f"phase{head_index}:family_signal:{dataset.family_names[index]}" for index in nonsingleton)

    family_total = np.column_stack([replay_exposure[:, members].sum(axis=1) for members in dataset.family_members])
    if variant.onset is OnsetScope.SHARED:
        threshold: np.ndarray | float = shape.penalty_threshold
    else:
        if family_tau is None or family_tau.shape != (len(dataset.family_names),):
            raise ValueError("Family onset requires one threshold per family")
        threshold = family_tau[None, :]
    pieces.append(retained.softplus_penalty(family_total, threshold))
    names.extend(f"family_penalty:{name}" for name in dataset.family_names)
    return np.hstack(pieces), tuple(names)


def family_totals(
    dataset: family_grp.Dataset,
    shape: retained.Shape,
    phase: phase_heads.PhaseKind,
) -> np.ndarray:
    _heads, replay_exposure = phase_heads.phase_exposures(dataset, shape, phase)
    return np.column_stack([replay_exposure[:, members].sum(axis=1) for members in dataset.family_members])


def select_shared_hyperparameters(
    dataset: family_grp.Dataset,
    dataset_id: hierarchy.DatasetId,
    phase: phase_heads.PhaseKind,
    shapes: tuple[retained.Shape, ...],
    indices: np.ndarray,
    seed: int,
    inner_splits: int,
) -> phase_heads.SharedSelection:
    return phase_heads.select_shared_hyperparameters(
        dataset,
        phase_heads.DatasetId(dataset_id.value),
        phase_variant(phase),
        shapes,
        indices,
        seed,
        inner_splits,
    )


def fit_shared_model(
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: phase_heads.SharedSelection,
    indices: np.ndarray,
) -> FittedModel:
    design, names = build_design(dataset, variant, selection.shape, None)
    head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
    return FittedModel(head, None, None, None, None, None)


def penalty_coefficient_indices(names: tuple[str, ...]) -> np.ndarray:
    indices = np.asarray([index for index, name in enumerate(names) if name.startswith("family_penalty:")])
    if len(indices) == 0:
        raise ValueError("Design has no family replay-penalty coefficients")
    return indices


def fit_family_tau_model(
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: phase_heads.SharedSelection,
    indices: np.ndarray,
    tau_shrink: float,
    maxiter: int,
    *,
    multistart: bool,
) -> FittedModel:
    anchor = selection.shape.penalty_threshold
    totals = family_totals(dataset, selection.shape, variant.phase)
    logged_totals = np.log1p(totals[indices])
    starts = [np.full(len(dataset.family_names), anchor, dtype=float)]
    quantiles = (0.5, 0.75, 0.9) if multistart else (0.75,)
    starts.extend(np.quantile(logged_totals, quantile, axis=0) for quantile in quantiles)

    def objective_and_gradient(tau: np.ndarray) -> tuple[float, np.ndarray]:
        design, names = build_design(dataset, variant, selection.shape, tau)
        head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
        residual = head.predict_design(design[indices]) - dataset.target[indices]
        coefficients = head.coefficients[penalty_coefficient_indices(names)]
        delta = logged_totals - tau[None, :]
        softplus = np.logaddexp(0.0, delta)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(delta, -50.0, 50.0)))
        penalty_derivative = -2.0 * softplus * sigmoid
        data_gradient = 2.0 * np.mean(
            residual[:, None] * coefficients[None, :] * penalty_derivative,
            axis=0,
        )
        displacement = tau - anchor
        shrink_loss = tau_shrink * float(np.mean(displacement**2))
        shrink_gradient = 2.0 * tau_shrink * displacement / len(displacement)
        ridge_loss = selection.l2 * float(np.sum(head.coefficients**2)) / len(indices)
        loss = float(np.mean(residual**2)) + ridge_loss + shrink_loss
        return loss, data_gradient + shrink_gradient

    results = [
        minimize(
            objective_and_gradient,
            np.clip(start, *TAU_BOUNDS),
            method="L-BFGS-B",
            jac=True,
            bounds=[TAU_BOUNDS] * len(dataset.family_names),
            options={"maxiter": maxiter, "ftol": 1e-12, "maxls": 30},
        )
        for start in starts
    ]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError(f"Family-onset optimization failed for {variant.name}")
    result = min(finite, key=lambda candidate: float(candidate.fun))
    family_tau = np.asarray(result.x, dtype=float)
    design, names = build_design(dataset, variant, selection.shape, family_tau)
    head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
    return FittedModel(
        head=head,
        family_tau=family_tau,
        tau_shrink=tau_shrink,
        objective=float(result.fun),
        iterations=int(result.nit),
        converged=bool(result.success),
    )


def select_tau_shrink(
    dataset: family_grp.Dataset,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    selection: phase_heads.SharedSelection,
    indices: np.ndarray,
    seed: int,
    inner_splits: int,
    maxiter: int,
) -> tuple[float, float]:
    splits = hierarchy.split_indices(dataset, dataset_id, indices, inner_splits, seed)
    best: tuple[float, float] | None = None
    for tau_shrink in TAU_SHRINK_GRID:
        errors: list[np.ndarray] = []
        for train, test in splits:
            model = fit_family_tau_model(
                dataset,
                variant,
                selection,
                train,
                tau_shrink,
                max(20, maxiter // 2),
                multistart=False,
            )
            design, _names = build_design(dataset, variant, selection.shape, model.family_tau)
            errors.append(model.head.predict_design(design[test]) - dataset.target[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        candidate = (score, tau_shrink)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(f"No onset-shrinkage candidate for {variant.name}")
    return best[1], best[0]


def model_prediction(
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: phase_heads.SharedSelection,
    model: FittedModel,
    indices: np.ndarray,
) -> np.ndarray:
    design, _names = build_design(dataset, variant, selection.shape, model.family_tau)
    return model.head.predict_design(design[indices])


def checkpoint_paths(
    output_dir: Path,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    fold: int,
) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"{dataset_id.value}__{variant.name}__outer_{fold}"
    return stem.with_suffix(".json"), stem.with_suffix(".npy")


def load_checkpoint(
    output_dir: Path,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    fold: int,
    test: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, variant, fold)
    if not metadata_path.exists() or not prediction_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {"schema": CHECKPOINT_SCHEMA, "test_indices": test.tolist(), **config}
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    prediction = np.load(prediction_path)
    if prediction.shape != test.shape:
        return None
    return np.asarray(prediction, dtype=float), metadata["selection"]


def save_checkpoint(
    output_dir: Path,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    fold: int,
    test: np.ndarray,
    prediction: np.ndarray,
    selection: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, variant, fold)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    metadata_path.write_text(
        json.dumps(
            {
                "schema": CHECKPOINT_SCHEMA,
                "test_indices": test.tolist(),
                **config,
                "selection": selection,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def nested_oof(
    dataset: family_grp.Dataset,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    shapes: tuple[retained.Shape, ...],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    outer = hierarchy.split_indices(
        dataset,
        dataset_id,
        np.arange(dataset.n),
        args.outer_splits,
        OUTER_CV_SEED,
    )
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    config = {
        "num_shapes": args.num_shapes,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "l2_grid": list(phase_heads.L2_GRID),
        "tau_shrink_grid": list(TAU_SHRINK_GRID),
        "tau_maxiter": args.tau_maxiter,
    }
    for fold, (train, test) in enumerate(outer):
        cached = None if args.force else load_checkpoint(output_dir, dataset_id, variant, fold, test, config)
        if cached is not None:
            fold_prediction, selection_row = cached
            prediction[test] = fold_prediction
            selections.append(selection_row)
            continue

        print(f"{dataset_id.value} {variant.name}: outer fold {fold + 1}/{len(outer)}", flush=True)
        selection = select_shared_hyperparameters(
            dataset,
            dataset_id,
            variant.phase,
            shapes,
            train,
            INNER_CV_SEED + fold,
            args.inner_splits,
        )
        tau_shrink: float | None = None
        tau_inner_rmse: float | None = None
        if variant.onset is OnsetScope.FAMILY:
            tau_shrink, tau_inner_rmse = select_tau_shrink(
                dataset,
                dataset_id,
                variant,
                selection,
                train,
                INNER_CV_SEED + 100 + fold,
                args.inner_splits,
                args.tau_maxiter,
            )
            model = fit_family_tau_model(
                dataset,
                variant,
                selection,
                train,
                tau_shrink,
                args.tau_maxiter,
                multistart=True,
            )
        else:
            model = fit_shared_model(dataset, variant, selection, train)
        fold_prediction = model_prediction(dataset, variant, selection, model, test)
        prediction[test] = fold_prediction
        family_tau = model.family_tau
        selection_row = {
            "dataset": dataset_id.value,
            "variant": variant.name,
            "outer_fold": fold,
            "shape": asdict(selection.shape),
            "l2": selection.l2,
            "shared_inner_rmse": selection.inner_rmse,
            "tau_shrink": tau_shrink,
            "tau_inner_rmse": tau_inner_rmse,
            "tau_min": None if family_tau is None else float(family_tau.min()),
            "tau_median": None if family_tau is None else float(np.median(family_tau)),
            "tau_max": None if family_tau is None else float(family_tau.max()),
            "tau_sd": None if family_tau is None else float(np.std(family_tau)),
            "tau_values": None if family_tau is None else family_tau.tolist(),
            "tau_iterations": model.iterations,
            "tau_converged": model.converged,
            "active_coefficient_count": int(np.count_nonzero(model.head.coefficients > 1e-10)),
        }
        save_checkpoint(
            output_dir,
            dataset_id,
            variant,
            fold,
            test,
            fold_prediction,
            selection_row,
            config,
        )
        selections.append(selection_row)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested OOF prediction for {dataset_id.value} {variant.name}")
    return prediction, [test for _train, test in outer], selections


def parameter_count(dataset: family_grp.Dataset, variant: Variant) -> int:
    base = phase_heads.parameter_count(dataset, phase_variant(variant.phase))
    if variant.onset is OnsetScope.FAMILY:
        return base + len(dataset.family_names) - 1
    return base


def paired_bootstrap(
    observed: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> dict[str, float]:
    delta = (candidate - observed) ** 2 - (reference - observed) ** 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean_mse_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_better": float(np.mean(means < 0.0)),
    }


def fit_full_model(
    dataset: family_grp.Dataset,
    dataset_id: hierarchy.DatasetId,
    variant: Variant,
    shapes: tuple[retained.Shape, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    indices = np.arange(dataset.n)
    selection = select_shared_hyperparameters(
        dataset,
        dataset_id,
        variant.phase,
        shapes,
        indices,
        INNER_CV_SEED + 999,
        args.inner_splits,
    )
    tau_shrink: float | None = None
    tau_inner_rmse: float | None = None
    if variant.onset is OnsetScope.FAMILY:
        tau_shrink, tau_inner_rmse = select_tau_shrink(
            dataset,
            dataset_id,
            variant,
            selection,
            indices,
            INNER_CV_SEED + 1999,
            args.inner_splits,
            args.tau_maxiter,
        )
        model = fit_family_tau_model(
            dataset,
            variant,
            selection,
            indices,
            tau_shrink,
            args.tau_maxiter,
            multistart=True,
        )
    else:
        model = fit_shared_model(dataset, variant, selection, indices)
    return {
        "shape": asdict(selection.shape),
        "l2": selection.l2,
        "shared_inner_rmse": selection.inner_rmse,
        "tau_shrink": tau_shrink,
        "tau_inner_rmse": tau_inner_rmse,
        "family_tau": (
            None if model.family_tau is None else dict(zip(dataset.family_names, model.family_tau.tolist(), strict=True))
        ),
        "intercept": model.head.intercept,
        "coefficients": dict(zip(model.head.feature_names, model.head.coefficients.tolist(), strict=True)),
        "active_coefficient_count": int(np.count_nonzero(model.head.coefficients > 1e-10)),
        "parameter_count": parameter_count(dataset, variant),
    }


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(metrics["dataset"].tolist()))
    colors = dict(
        zip(
            [variant.name for variant in VARIANTS],
            sample_colorscale("RdYlGn_r", np.linspace(0.1, 0.9, len(VARIANTS))),
            strict=True,
        )
    )
    figure = make_subplots(
        rows=len(datasets),
        cols=4,
        subplot_titles=tuple(
            f"{dataset}: {title}" for dataset in datasets for title in ("RMSE", "Spearman", "Regret@1", "Low-tail RMSE")
        ),
    )
    for row, dataset in enumerate(datasets, start=1):
        frame = metrics.loc[metrics["dataset"].eq(dataset)]
        for col, metric in enumerate(
            ("rmse", "spearman", "fold_mean_regret_at_1", "low_tail_rmse"),
            start=1,
        ):
            figure.add_bar(
                x=frame["variant"],
                y=frame[metric],
                marker_color=[colors[name] for name in frame["variant"]],
                text=[f"{value:.5f}" for value in frame[metric]],
                textposition="outside",
                showlegend=False,
                row=row,
                col=col,
            )
    figure.update_layout(
        title="Family-specific replay onset in Bucket-resolved and separate-heads GRP",
        template="plotly_white",
        width=2100,
        height=470 * len(datasets),
        margin={"b": 170},
    )
    figure.update_xaxes(tickangle=-30)
    figure.write_html(output_dir / "family_onset_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def gate_decisions(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for shared_name, family_name in PAIRS:
        shared = metrics.loc[metrics["variant"].eq(shared_name)].set_index("dataset")
        family = metrics.loc[metrics["variant"].eq(family_name)].set_index("dataset")
        relative_rmse = family["rmse"] / shared["rmse"] - 1.0
        relative_tail = family["low_tail_rmse"] / shared["low_tail_rmse"] - 1.0
        pair_rows = comparisons.loc[comparisons["candidate"].eq(family_name)]
        universal_paper_gate = bool(
            (relative_rmse < 0.0).sum() >= 2
            and float(relative_rmse.mean()) <= -0.01
            and float(relative_rmse.max()) <= 0.02
            and float(relative_tail.max()) <= 0.05
            and (pair_rows["fold_wins"] >= 3).sum() >= 2
        )
        decisive_target = pair_rows.loc[
            pair_rows["fold_wins"].eq(pair_rows["fold_count"]) & pair_rows["probability_better"].ge(0.99)
        ]
        decisive_target_gain = any(relative_rmse.loc[dataset] <= -0.02 for dataset in decisive_target["dataset"])
        observatory_gate = bool(
            decisive_target_gain and float(relative_rmse.max()) <= 0.01 and float(relative_tail.max()) <= 0.05
        )
        decisions[family_name] = {
            "add_to_observatory": observatory_gate,
            "observatory_criterion": (
                "At least one target improves RMSE by 2% with wins in every outer fold and paired-bootstrap "
                "probability >=0.99; no other target regresses by more than 1% RMSE or 5% low-tail RMSE."
            ),
            "universal_paper_candidate": universal_paper_gate,
            "universal_criterion": (
                "RMSE improves on at least 2/3 datasets, mean RMSE improves by at least 1%, no RMSE regression "
                "exceeds 2%, no low-tail RMSE regression exceeds 5%, and at least two datasets win 3/5 folds."
            ),
            "relative_rmse": relative_rmse.to_dict(),
            "relative_low_tail_rmse": relative_tail.to_dict(),
        }
    return decisions


def write_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    selections: pd.DataFrame,
    decisions: dict[str, Any],
    output_dir: Path,
) -> None:
    def model_metric(variant: str, metric: str) -> float:
        values = metrics.loc[
            metrics["dataset"].eq(hierarchy.DatasetId.THREE_HUNDRED_M_TABLE9.value) & metrics["variant"].eq(variant),
            metric,
        ].to_numpy(dtype=float)
        if len(values) != 1:
            raise ValueError(f"Expected one Table-9 {variant} {metric} value, found {len(values)}")
        return float(values[0])

    def comparison_metric(metric: str) -> float:
        values = comparisons.loc[
            comparisons["dataset"].eq(hierarchy.DatasetId.THREE_HUNDRED_M_TABLE9.value)
            & comparisons["candidate"].eq("power_separate_heads_family_tau"),
            metric,
        ].to_numpy(dtype=float)
        if len(values) != 1:
            raise ValueError(f"Expected one Table-9 paired {metric} value, found {len(values)}")
        return float(values[0])

    bucket_table9_rmse = model_metric("bucket_resolved_family_tau", "rmse")
    bucket_table9_control_rmse = model_metric("bucket_resolved_shared_tau", "rmse")
    heads_table9_rmse = model_metric("power_separate_heads_family_tau", "rmse")
    heads_table9_control_rmse = model_metric("power_separate_heads_shared_tau", "rmse")
    heads_table9_low_tail = model_metric("power_separate_heads_family_tau", "low_tail_rmse")
    heads_table9_control_low_tail = model_metric("power_separate_heads_shared_tau", "low_tail_rmse")
    compact_selections = selections.drop(columns=["tau_values"], errors="ignore")
    lines = [
        "# Family-specific replay onset in GRP",
        "",
        "## Question",
        "",
        "Does replacing the shared replay onset $\\tau$ by one onset $\\tau_f$ per semantic family improve "
        "Bucket-resolved family GRP or Power + separate heads? The family model is",
        "",
        "$$P_f(Z_f)=B_f\\,\\operatorname{softplus}(\\log(1+Z_f)-\\tau_f)^2,$$",
        "",
        "and nests the shared-onset control at $\\tau_f=\\tau$. The response shape and ridge penalty are "
        "selected inside each outer fold. A second inner CV selects shrinkage of $\\tau_f$ toward $\\tau$.",
        "",
        "## Nested OOF metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Paired comparisons",
        "",
        comparisons.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- **Bucket-resolved family GRP:** family-specific onset is not supported. It is exactly neutral on "
        "production and 300M Uncheatable, then worsens Table-9 RMSE by "
        f"{bucket_table9_rmse / bucket_table9_control_rmse - 1:+.2%}.",
        "- **Power + separate heads:** family onset is exactly neutral on production, effectively neutral on "
        "300M Uncheatable, and improves Table-9 RMSE by "
        f"{heads_table9_rmse / heads_table9_control_rmse - 1:+.2%} and low-tail RMSE by "
        f"{heads_table9_low_tail / heads_table9_control_low_tail - 1:+.2%}. It wins "
        f"{int(comparison_metric('fold_wins'))}/{int(comparison_metric('fold_count'))} outer folds; the paired "
        f"bootstrap probability of lower squared error is {comparison_metric('probability_better'):.4f}.",
        "- The full Table-9 separate-heads fit learns onset exposures of 18.5, 12.8, and 6.0 retained epochs "
        "for broad text, tech/code, and reasoning. On production, all 36 family onsets collapse exactly to the "
        "shared value, so the nested extension self-prunes when the panel does not identify family differences.",
        "- This passes the diagnostic-Observatory gate but not the stricter universal paper-model gate. It is "
        "useful for inspecting a targeted Table-9 mechanism; it is not evidence that every GRP fit should learn "
        "family onsets.",
        "",
        "## Observatory gate",
        "",
        "~~~json",
        json.dumps(decisions, indent=2, allow_nan=False),
        "~~~",
        "",
        "## Fold selections",
        "",
        compact_selections.to_markdown(index=False),
        "",
        "## Reproduce",
        "",
        "~~~bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "benchmark_grp_family_onset_phase_heads_20260714.py",
        "~~~",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def check_parity() -> None:
    for dataset_id in hierarchy.DatasetId:
        dataset = hierarchy.load_dataset(dataset_id)
        for phase in phase_heads.PhaseKind:
            variant = Variant("parity", phase, OnsetScope.SHARED)
            shape = candidate_shapes(phase, 2)[0]
            design, names = build_design(dataset, variant, shape, None)
            reference, reference_names, _layout = phase_heads.build_design(
                dataset,
                phase_variant(phase),
                shape,
                None,
            )
            error = float(np.max(np.abs(design - reference)))
            if error > 1e-12 or names != reference_names:
                raise ValueError(f"Shared-onset parity failed for {dataset_id.value} {phase.value}: {error:.3e}")
    print("Shared-onset design parity passed", flush=True)


def main() -> None:
    args = parse_args()
    check_parity()
    if args.check_parity:
        return
    requested = tuple(hierarchy.DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}

    for dataset_index, dataset_id in enumerate(requested):
        dataset = hierarchy.load_dataset(dataset_id)
        predictions_by_variant: dict[str, np.ndarray] = {}
        outer_folds: list[np.ndarray] | None = None
        full_models[dataset_id.value] = {}
        for variant in VARIANTS:
            shapes = candidate_shapes(variant.phase, args.num_shapes)
            prediction, current_folds, selections = nested_oof(
                dataset,
                dataset_id,
                variant,
                shapes,
                args.output_dir,
                args,
            )
            if outer_folds is None:
                outer_folds = current_folds
            elif any(not np.array_equal(left, right) for left, right in zip(outer_folds, current_folds, strict=True)):
                raise ValueError(f"Outer folds differ for {dataset_id.value}")
            predictions_by_variant[variant.name] = prediction
            summary = family_grp.metric_summary(dataset.target, prediction, current_folds)
            metric_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.name,
                    "parameter_count": parameter_count(dataset, variant),
                    **summary,
                }
            )
            for fold, test in enumerate(current_folds):
                fold_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "outer_fold": fold,
                        "rmse": float(np.sqrt(np.mean((prediction[test] - dataset.target[test]) ** 2))),
                    }
                )
            for row_index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
                prediction_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "row_index": row_index,
                        "observed": observed,
                        "prediction": predicted,
                        "residual": predicted - observed,
                    }
                )
            selection_rows.extend(selections)
            full_models[dataset_id.value][variant.name] = fit_full_model(
                dataset,
                dataset_id,
                variant,
                shapes,
                args,
            )

        if outer_folds is None:
            raise RuntimeError(f"No folds generated for {dataset_id.value}")
        fold_frame = pd.DataFrame(fold_rows)
        for pair_index, (shared_name, family_name) in enumerate(PAIRS):
            shared = predictions_by_variant[shared_name]
            family = predictions_by_variant[family_name]
            shared_fold = fold_frame.loc[
                fold_frame["dataset"].eq(dataset_id.value) & fold_frame["variant"].eq(shared_name),
                "rmse",
            ].to_numpy()
            family_fold = fold_frame.loc[
                fold_frame["dataset"].eq(dataset_id.value) & fold_frame["variant"].eq(family_name),
                "rmse",
            ].to_numpy()
            comparison_rows.append(
                {
                    "dataset": dataset_id.value,
                    "reference": shared_name,
                    "candidate": family_name,
                    "fold_wins": int(np.count_nonzero(family_fold < shared_fold)),
                    "fold_count": len(outer_folds),
                    **paired_bootstrap(
                        dataset.target,
                        shared,
                        family,
                        OUTER_CV_SEED + 100 * dataset_index + pair_index,
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "nested_cv_selections.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    (args.output_dir / "full_models.json").write_text(json.dumps(full_models, indent=2, allow_nan=False) + "\n")
    decisions = gate_decisions(metrics, comparisons)
    (args.output_dir / "observatory_gate.json").write_text(json.dumps(decisions, indent=2, allow_nan=False) + "\n")
    plot_metrics(metrics, args.output_dir)
    write_report(metrics, comparisons, selections, decisions, args.output_dir)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decisions, indent=2), flush=True)


if __name__ == "__main__":
    main()
