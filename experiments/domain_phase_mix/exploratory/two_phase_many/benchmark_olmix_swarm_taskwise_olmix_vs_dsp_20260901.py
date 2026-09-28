# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///

"""Compare paper-faithful taskwise OLMix with single-phase DSP models.

The expensive fits are restartable at task-fold granularity. Every estimator
uses the same mixture-blocked outer folds. OLMix follows the default paper
estimator by fitting one positive log-linear law per atomic metric and averaging
the 42 task predictions. Canonical DSP retains its established aggregate head
and refines its nonlinear geometry with an active-set implicit gradient through
the ridge-NNLS head.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_single_phase_dsp_20260901 as incumbent,
)
from experiments.domain_phase_mix.olmix_loglinear_fit import fit_olmix_loglinear_model  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_taskwise_olmix_vs_dsp_20260901"
DEFAULT_INPUT_DIR = incumbent.DEFAULT_INPUT_DIR
INCUMBENT_OUTPUT_DIR = incumbent.DEFAULT_OUTPUT_DIR
TASKWISE_MODELS = ("olmix_taskwise_raw", "olmix_taskwise_log_epoch")
CANONICAL_MODELS = ("dsp_canonical_macro", "dsp_canonical_macro_coarse")
FITTABLE_MODELS = (*TASKWISE_MODELS, *CANONICAL_MODELS)
REFERENCE_MODELS = ("linear_epoch_log_link", "dsp_benefit_log_link")
ALL_MODELS = (*TASKWISE_MODELS, *CANONICAL_MODELS, *REFERENCE_MODELS)
MODEL_LABELS = {
    "olmix_taskwise_raw": "Vanilla OLMix (taskwise raw weights)",
    "olmix_taskwise_log_epoch": "OLMix + log epochs (taskwise)",
    "dsp_canonical_macro": "Canonical DSP (aggregate)",
    "dsp_canonical_macro_coarse": "Canonical DSP, coarse screen (aggregate)",
    "dsp_benefit_log_link": "Shared-rate DSP (taskwise)",
    "linear_epoch_log_link": "Linear log-epoch head (taskwise diagnostic)",
}


@dataclasses.dataclass(frozen=True)
class Fold:
    pool: str
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-repeats", type=int, default=incumbent.OUTER_REPEATS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--models", nargs="+", choices=FITTABLE_MODELS, default=list(FITTABLE_MODELS))
    parser.add_argument("--olmix-starts", type=int, default=48)
    parser.add_argument("--canonical-maxiter", type=int, default=36)
    parser.add_argument("--canonical-restarts", type=int, default=2)
    parser.add_argument("--canonical-coarse-starts", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


def folds_for_pool(pool: incumbent.Pool, repeats: int) -> tuple[Fold, ...]:
    rows = np.arange(len(pool.runs))
    folds: list[Fold] = []
    for repeat in range(repeats):
        labels = incumbent.block_labels(
            pool.weights,
            incumbent.OUTER_FOLDS,
            incumbent.FOLD_SEED + 100 * repeat,
        )
        for fold in range(incumbent.OUTER_FOLDS):
            folds.append(
                Fold(
                    pool=pool.name,
                    repeat=repeat,
                    fold=fold,
                    train=rows[labels != fold],
                    test=rows[labels == fold],
                )
            )
    return tuple(folds)


def model_inputs(pool: incumbent.Pool, model: str) -> np.ndarray:
    if model == "olmix_taskwise_raw":
        return pool.weights
    if model == "olmix_taskwise_log_epoch":
        return np.log1p(pool.exposures)
    raise ValueError(f"No taskwise OLMix inputs for {model}")


def shard_path(
    output_dir: Path,
    *,
    pool: str,
    model: str,
    repeat: int,
    fold: int,
    task: int | None,
) -> Path:
    suffix = f"task_{task:02d}.npz" if task is not None else "fold.npz"
    return output_dir / "shards" / pool / model / f"repeat_{repeat:02d}" / f"fold_{fold:02d}" / suffix


def atomic_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def valid_prediction_shard(path: Path, test: np.ndarray, task: int | None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as payload:
            if not np.array_equal(payload["test"], test):
                return False
            prediction = np.asarray(payload["prediction"], dtype=float)
            if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
                return False
            if task is not None and int(payload["task"].item()) != task:
                return False
    except (KeyError, OSError, ValueError):
        return False
    return True


def fit_olmix_task_shard(
    pool: incumbent.Pool,
    split: Fold,
    model: str,
    task: int,
    starts: int,
    path: Path,
) -> str:
    if valid_prediction_shard(path, split.test, task):
        return "cached"
    inputs = model_inputs(pool, model)
    fit = fit_olmix_loglinear_model(
        inputs[split.train],
        pool.outcomes[split.train, task],
        seed=0,
        n_starts=starts,
    )
    prediction = fit.predict(inputs[split.test])
    atomic_save(
        path,
        test=split.test,
        prediction=prediction,
        task=np.asarray(task),
        huber_loss=np.asarray(fit.huber_loss),
    )
    return "fitted"


def coarse_canonical_fit(
    exposure: np.ndarray,
    response: np.ndarray,
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    seed: int,
    starts: int,
) -> tuple[np.ndarray, float, int]:
    """Select a fixed canonical geometry without continuous optimization."""
    if starts < 1:
        raise ValueError("canonical coarse starts must be positive")
    rung = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    bounds = dsp_ladder.rung_bounds(rung, exposure.shape[1])
    lows = np.asarray([low for low, _ in bounds])
    highs = np.asarray([high for _, high in bounds])
    generator = np.random.default_rng(20_260_824 + seed)
    candidates = [0.5 * (lows + highs)]
    candidates.extend(generator.uniform(lows, highs) for _ in range(starts - 1))

    best_start = -1
    best_objective = np.inf
    best_vector: np.ndarray | None = None
    for start_id, vector in enumerate(candidates):
        objective = 0.0
        for train, validation in inner_folds:
            design = dsp_ladder.rung_design(exposure[train], vector, rung, exposure.shape[1])
            intercept, coefficients = dsp_ladder.solve_head(design, response[train], ())
            validation_design = dsp_ladder.rung_design(exposure[validation], vector, rung, exposure.shape[1])
            residual = intercept + validation_design @ coefficients - response[validation]
            objective += float(residual @ residual)
        if objective < best_objective:
            best_start = start_id
            best_objective = objective
            best_vector = vector

    assert best_vector is not None
    return best_vector, best_objective, best_start


def canonical_prediction(
    pool: incumbent.Pool,
    split: Fold,
    *,
    model: str,
    maxiter: int,
    restarts: int,
    coarse_starts: int,
) -> tuple[np.ndarray, int, float]:
    exposure = pool.exposures[split.train]
    response = pool.outcomes[split.train].mean(axis=1)
    inner_labels = incumbent.block_labels(
        pool.weights[split.train],
        incumbent.INNER_FOLDS,
        incumbent.FOLD_SEED + 1000 * split.repeat + split.fold,
    )
    inner_folds = tuple(
        (np.flatnonzero(inner_labels != fold), np.flatnonzero(inner_labels == fold))
        for fold in range(incumbent.INNER_FOLDS)
    )
    rung = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    seed = 100 * split.repeat + split.fold
    if model == "dsp_canonical_macro_coarse":
        vector, objective, start_id = coarse_canonical_fit(
            exposure,
            response,
            inner_folds,
            seed=seed,
            starts=coarse_starts,
        )
        intercept, coefficients = dsp_ladder.solve_head(
            dsp_ladder.rung_design(exposure, vector, rung, len(pool.buckets)), response, ()
        )
    elif model == "dsp_canonical_macro":
        vector, intercept, coefficients = dsp_ladder.fit_rung(
            exposure,
            response,
            rung,
            inner_folds,
            (),
            seed=seed,
            maxiter=maxiter,
            restarts=restarts,
        )
        start_id = -1
        objective = float("nan")
    else:
        raise ValueError(f"Unknown canonical DSP model: {model}")
    design = dsp_ladder.rung_design(pool.exposures[split.test], vector, rung, len(pool.buckets))
    prediction = np.asarray(intercept + design @ coefficients, dtype=float)
    return prediction, start_id, objective


def fit_canonical_shard(
    pool: incumbent.Pool,
    split: Fold,
    *,
    model: str,
    maxiter: int,
    restarts: int,
    coarse_starts: int,
    path: Path,
) -> str:
    if valid_prediction_shard(path, split.test, None):
        return "cached"
    prediction, start_id, objective = canonical_prediction(
        pool,
        split,
        model=model,
        maxiter=maxiter,
        restarts=restarts,
        coarse_starts=coarse_starts,
    )
    atomic_save(
        path,
        test=split.test,
        prediction=prediction,
        selected_coarse_start=np.asarray(start_id),
        inner_objective=np.asarray(objective),
    )
    return "fitted"


def fit_requested_models(
    pools: tuple[incumbent.Pool, ...],
    *,
    output_dir: Path,
    repeats: int,
    models: tuple[str, ...],
    workers: int,
    olmix_starts: int,
    canonical_maxiter: int,
    canonical_restarts: int,
    canonical_coarse_starts: int,
) -> None:
    olmix_jobs = []
    canonical_jobs = []
    for pool in pools:
        for split in folds_for_pool(pool, repeats):
            for model in models:
                if model in TASKWISE_MODELS:
                    for task in range(len(pool.tasks)):
                        path = shard_path(
                            output_dir,
                            pool=pool.name,
                            model=model,
                            repeat=split.repeat,
                            fold=split.fold,
                            task=task,
                        )
                        if not valid_prediction_shard(path, split.test, task):
                            olmix_jobs.append(
                                delayed(fit_olmix_task_shard)(pool, split, model, task, olmix_starts, path)
                            )
                elif model in CANONICAL_MODELS:
                    path = shard_path(
                        output_dir,
                        pool=pool.name,
                        model=model,
                        repeat=split.repeat,
                        fold=split.fold,
                        task=None,
                    )
                    if not valid_prediction_shard(path, split.test, None):
                        canonical_jobs.append(
                            delayed(fit_canonical_shard)(
                                pool,
                                split,
                                model=model,
                                maxiter=canonical_maxiter,
                                restarts=canonical_restarts,
                                coarse_starts=canonical_coarse_starts,
                                path=path,
                            )
                        )
    jobs = [*olmix_jobs, *canonical_jobs]
    print(
        f"Pending shards: {len(olmix_jobs)} taskwise OLMix, {len(canonical_jobs)} canonical DSP; " f"workers={workers}",
        flush=True,
    )
    if not jobs:
        return
    with parallel_config(backend="loky", inner_max_num_threads=1):
        Parallel(n_jobs=workers, verbose=10)(jobs)


def load_taskwise_fold(
    pool: incumbent.Pool,
    split: Fold,
    model: str,
    output_dir: Path,
) -> np.ndarray | None:
    task_predictions = []
    for task in range(len(pool.tasks)):
        path = shard_path(
            output_dir,
            pool=pool.name,
            model=model,
            repeat=split.repeat,
            fold=split.fold,
            task=task,
        )
        if not valid_prediction_shard(path, split.test, task):
            return None
        with np.load(path) as payload:
            task_predictions.append(np.asarray(payload["prediction"], dtype=float))
    return np.column_stack(task_predictions).mean(axis=1)


def load_canonical_fold(
    pool: incumbent.Pool,
    split: Fold,
    model: str,
    output_dir: Path,
) -> np.ndarray | None:
    path = shard_path(
        output_dir,
        pool=pool.name,
        model=model,
        repeat=split.repeat,
        fold=split.fold,
        task=None,
    )
    if not valid_prediction_shard(path, split.test, None):
        return None
    with np.load(path) as payload:
        return np.asarray(payload["prediction"], dtype=float)


def prediction_rows(
    pool: incumbent.Pool,
    split: Fold,
    model: str,
    prediction: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    actual = pool.outcomes.mean(axis=1)
    scores = incumbent.fold_scores(actual[split.test], prediction)
    fold_row: dict[str, object] = {
        "pool": pool.name,
        "variant": model,
        "repeat": split.repeat,
        "fold": split.fold,
        "test_rows": len(split.test),
        **scores,
    }
    rows = [
        {
            "pool": pool.name,
            "variant": model,
            "repeat": split.repeat,
            "fold": split.fold,
            "run": pool.runs[row],
            "index": int(row),
            "observed_macro_bpb": actual[row],
            "predicted_macro_bpb": prediction[local],
        }
        for local, row in enumerate(split.test)
    ]
    return rows, fold_row


def compile_fitted_predictions(
    pools: tuple[incumbent.Pool, ...],
    *,
    output_dir: Path,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_output: list[dict[str, object]] = []
    fold_output: list[dict[str, object]] = []
    for pool in pools:
        for split in folds_for_pool(pool, repeats):
            for model in TASKWISE_MODELS:
                prediction = load_taskwise_fold(pool, split, model, output_dir)
                if prediction is not None:
                    rows, fold_row = prediction_rows(pool, split, model, prediction)
                    prediction_output.extend(rows)
                    fold_output.append(fold_row)
            for model in CANONICAL_MODELS:
                prediction = load_canonical_fold(pool, split, model, output_dir)
                if prediction is not None:
                    rows, fold_row = prediction_rows(pool, split, model, prediction)
                    prediction_output.extend(rows)
                    fold_output.append(fold_row)
    return pd.DataFrame(prediction_output), pd.DataFrame(fold_output)


def load_reference_predictions(repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(INCUMBENT_OUTPUT_DIR / "predictions.csv")
    folds = pd.read_csv(INCUMBENT_OUTPUT_DIR / "fold_metrics.csv")
    predictions = predictions[predictions.variant.isin(REFERENCE_MODELS) & predictions.repeat.lt(repeats)].copy()
    folds = folds[folds.variant.isin(REFERENCE_MODELS) & folds.repeat.lt(repeats)].copy()
    return predictions, folds


def aggregate_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, variant), group in predictions.groupby(["pool", "variant"], sort=False):
        repeat_rows = []
        for repeat, repeated in group.groupby("repeat"):
            score = incumbent.fold_scores(
                repeated.observed_macro_bpb.to_numpy(float),
                repeated.predicted_macro_bpb.to_numpy(float),
            )
            repeat_rows.append({"repeat": int(repeat), **score})
        repeat_frame = pd.DataFrame(repeat_rows)
        fold_group = folds[(folds.pool == pool) & (folds.variant == variant)]
        row: dict[str, object] = {
            "pool": pool,
            "variant": variant,
            "completed_repeats": int(repeat_frame.repeat.nunique()),
            "completed_folds": len(fold_group),
        }
        for metric in ("rmse", "mae", "spearman", "calibration_slope"):
            row[metric] = float(repeat_frame[metric].mean())
            row[f"{metric}_repeat_sd"] = (
                float(repeat_frame[metric].std(ddof=1)) if len(repeat_frame) > 1 else float("nan")
            )
        row["mean_fold_selection_regret"] = float(fold_group.selection_regret.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pool", "rmse", "mean_fold_selection_regret"])


def corrected_contrasts(folds: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rows = []
    for pool, group in folds.groupby("pool"):
        pivot = group.pivot(index=["repeat", "fold"], columns="variant", values="rmse")
        variants = [variant for variant in ALL_MODELS if variant in pivot.columns]
        for challenger in variants:
            for comparator in variants:
                if challenger >= comparator:
                    continue
                paired = pivot[[challenger, comparator]].dropna()
                if len(paired) < 2:
                    continue
                difference = paired[challenger] - paired[comparator]
                complete_repeats = max(1, min(repeats, int(paired.index.get_level_values("repeat").nunique())))
                factor = 1.0 / (incumbent.OUTER_FOLDS * complete_repeats) + 1.0 / (incumbent.OUTER_FOLDS - 1.0)
                se = float(np.sqrt(factor * difference.var(ddof=1)))
                critical = float(stats.t.ppf(0.975, len(difference) - 1))
                mean = float(difference.mean())
                rows.append(
                    {
                        "pool": pool,
                        "comparison": f"{challenger}_minus_{comparator}",
                        "paired_folds": len(difference),
                        "mean_rmse_difference": mean,
                        "corrected_se": se,
                        "ci_low": mean - critical * se,
                        "ci_high": mean + critical * se,
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    requested_repeats: int,
    olmix_starts: int,
    canonical_maxiter: int,
    canonical_restarts: int,
    canonical_coarse_starts: int,
) -> None:
    lines = [
        "# Taskwise OLMix versus single-phase DSP",
        "",
        "This benchmark corrects the earlier scalar-macro OLMix baseline. Vanilla OLMix and its log-epoch "
        "transform fit one independent positive log-linear law to each of 42 atomic BPB metrics, then average "
        "their held-out predictions. All rows use the same K-means mixture-blocked outer folds.",
        "",
        "Canonical DSP retains the programme's established aggregate target and four parameters per bucket: "
        "benefit amplitude, saturation rate, overexposure amplitude, and overexposure threshold. The shared-rate "
        "DSP row is the prespecified best practical DSP package from the preceding matched benchmark: taskwise "
        "positive heads with one nested-selected saturation rate. The linear log-epoch row is included only as a "
        "diagnostic. Head granularity is therefore part of each estimator package, not an experimentally isolated "
        "difference.",
        "",
        "## Results",
        "",
        "| pool | model | repeats | RMSE | Spearman | mean fold regret |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.pool} | {MODEL_LABELS.get(row.variant, row.variant)} | {row.completed_repeats} | "
            f"{row.rmse:.5f} | {row.spearman:.3f} | {row.mean_fold_selection_regret:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Requested outer validation: {requested_repeats} repeats x {incumbent.OUTER_FOLDS} folds.",
            f"- OLMix: {olmix_starts} deterministic multistarts per task-fold fit; Huber delta "
            f"{incumbent.DEFAULT_HUBER_DELTA if hasattr(incumbent, 'DEFAULT_HUBER_DELTA') else 0.02:g}.",
            f"- Canonical DSP: {canonical_restarts} starts, at most {canonical_maxiter} L-BFGS-B iterations, "
            f"an active-set implicit ridge-NNLS Jacobian, and {incumbent.INNER_FOLDS}-fold blocked inner selection.",
            f"- Coarse canonical DSP: {canonical_coarse_starts} deterministic candidate geometries and "
            f"{incumbent.INNER_FOLDS}-fold blocked inner selection; no continuous optimizer.",
            "- Selection regret is the observed BPB gap between the held-fold row predicted best and the actual "
            "held-fold minimum; lower is better.",
            "",
            "## Corrected RMSE contrasts",
            "",
            contrasts.to_markdown(index=False, floatfmt=".6f") if not contrasts.empty else "No paired contrasts yet.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 1 <= args.outer_repeats <= incumbent.OUTER_REPEATS:
        raise ValueError(f"outer repeats must be in [1, {incumbent.OUTER_REPEATS}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pools = tuple(incumbent.load_pool(args.input_dir, name) for name in incumbent.POOLS)
    if not args.compile_only:
        fit_requested_models(
            pools,
            output_dir=args.output_dir,
            repeats=args.outer_repeats,
            models=tuple(args.models),
            workers=args.workers,
            olmix_starts=args.olmix_starts,
            canonical_maxiter=args.canonical_maxiter,
            canonical_restarts=args.canonical_restarts,
            canonical_coarse_starts=args.canonical_coarse_starts,
        )
    fitted_predictions, fitted_folds = compile_fitted_predictions(
        pools,
        output_dir=args.output_dir,
        repeats=args.outer_repeats,
    )
    reference_predictions, reference_folds = load_reference_predictions(args.outer_repeats)
    predictions = pd.concat([fitted_predictions, reference_predictions], ignore_index=True)
    folds = pd.concat([fitted_folds, reference_folds], ignore_index=True)
    aggregate = aggregate_metrics(predictions, folds)
    contrasts = corrected_contrasts(folds, args.outer_repeats)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    contrasts.to_csv(args.output_dir / "corrected_contrasts.csv", index=False)
    protocol = {
        "outer_folds": incumbent.OUTER_FOLDS,
        "outer_repeats_requested": args.outer_repeats,
        "fold_geometry": "KMeans on square-root mixture weights",
        "models": list(ALL_MODELS),
        "olmix_starts": args.olmix_starts,
        "olmix_fit_granularity": "one law per atomic task",
        "canonical_dsp_fit_granularity": "aggregate macro BPB",
        "canonical_gradient": "active-set implicit differentiation through the ridge-NNLS head",
        "canonical_maxiter": args.canonical_maxiter,
        "canonical_restarts": args.canonical_restarts,
        "canonical_coarse_starts": args.canonical_coarse_starts,
        "inputs": {pool.name: pool.input_hashes for pool in pools},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    write_report(
        args.output_dir,
        aggregate,
        contrasts,
        requested_repeats=args.outer_repeats,
        olmix_starts=args.olmix_starts,
        canonical_maxiter=args.canonical_maxiter,
        canonical_restarts=args.canonical_restarts,
        canonical_coarse_starts=args.canonical_coarse_starts,
    )
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
