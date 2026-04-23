# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "wandb"]
# ///
"""Repeat the two-phase Feature-Bayes-Linear validation loop under boundary-aligned WSD."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rigging.filesystem import marin_prefix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "harbor" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "fray" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "haliax" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "levanter" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "iris" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "zephyr" / "src"))

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.utils import create_cache_tokenizer_step
from scipy.optimize import minimize
from scipy.stats import spearmanr

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    GENERAL_MODELS,
    DatasetSpec,
    _huber_delta,
)
from experiments.domain_phase_mix.exploratory.plot_starcoder_optima_validation import (
    _fetch_actual_metric_map,
)
from experiments.domain_phase_mix.starcoder_metadata import TWO_PHASE_STARCODER, load_starcoder_dataset
from experiments.domain_phase_mix.static_batch_selection import (
    replay_proposals_to_observed,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_two_phase_wsd_boundary_aligned_experiment,
)
from fray.cluster import ResourceConfig

OUTPUT_ROOT = SCRIPT_DIR / "starcoder_wsd_boundary_aligned_repeat_outputs"
PLOTS_DIRNAME = "plots"
PLOT_SCRIPT = SCRIPT_DIR / "plot_starcoder_optima_validation.py"

DATASET_NAME = TWO_PHASE_STARCODER.name
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"
OBJECTIVE_MODEL = "DS-RE-CEQ"
OBJECTIVE_MODE = "retrospective"
SOURCE_POLICY = "feature_bayes_linear_observed"
DEFAULT_POLICY = "feature_bayes_linear_wsd_boundary_aligned"
FULL_SUBSET_GRID = (4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96)
DOMAIN_NAMES = ("nemotron_full", "starcoder")

PROXY_RUN_ID_BASE = 96_000
VALIDATION_RUN_ID_BASE = 97_000
DEFAULT_DATA_SEED = 0

DEFAULT_DSRE_FIT_SEEDS = (0, 1, 2)
DEFAULT_DSRE_RESTARTS = 4
DEFAULT_DSRE_MAXITER = 300
DEFAULT_OPT_SEARCH_POINTS = 4096
DEFAULT_OPT_RESTARTS = 8
DEFAULT_OPT_MAXITER = 200


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(14, max(1, cpu_count - 2))


def _parse_subset_sizes(text: str, *, max_size: int) -> tuple[int, ...]:
    raw = FULL_SUBSET_GRID if not text.strip() else tuple(int(part) for part in text.split(",") if part.strip())
    return tuple(sorted({size for size in raw if 0 < size <= max_size}))


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "output_dir": output_dir,
        "plots_dir": output_dir / PLOTS_DIRNAME,
        "proxy_selection_csv": output_dir / "proxy_selection_plan.csv",
        "proxy_launch_plan_json": output_dir / "proxy_launch_plan.json",
        "proxy_results_csv": output_dir / "proxy_results.csv",
        "fit_status_json": output_dir / "fit_status.json",
        "model_scores_csv": output_dir / "model_scores.csv",
        "curve_points_csv": output_dir / "curve_points.csv",
        "predicted_optima_csv": output_dir / "predicted_optima.csv",
        "predicted_optima_jsonl": output_dir / "predicted_optima.jsonl",
        "validation_launch_plan_json": output_dir / f"{DATASET_NAME}_validation_launch_plan.json",
    }


def _output_tag(output_dir: Path) -> str:
    return output_dir.name.replace("_", "-")[:16]


def _proxy_name_prefix(output_dir: Path) -> str:
    return f"t2s-wsdb-proxy-{_output_tag(output_dir)}"


def _validation_name_prefix(output_dir: Path) -> str:
    return f"t2s-wsdb-val-{_output_tag(output_dir)}"


def _region_local_marin_path(default_path: str) -> str:
    current_prefix = marin_prefix().rstrip("/")
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


@contextmanager
def _executor_cli_context():
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


def _load_source_bundle(row_limit: int | None = None) -> tuple[DatasetSpec, pd.DataFrame]:
    spec, frame = load_starcoder_dataset(TWO_PHASE_STARCODER, target_col=OBJECTIVE_METRIC)
    if row_limit is None or row_limit >= spec.R:
        return spec, frame
    indices = np.arange(row_limit, dtype=int)
    return spec.subset(indices), frame.iloc[:row_limit].reset_index(drop=True)


def _phase_weight_config_from_row(row: pd.Series, *, run_id: int, spec: DatasetSpec) -> WeightConfig:
    phase_weights = {
        phase_name: {domain_name: float(row[f"{phase_name}_{domain_name}"]) for domain_name in spec.domain_names}
        for phase_name in spec.phase_names
    }
    return WeightConfig(run_id=run_id, phase_weights=phase_weights)


def _phase_column_payload(config: WeightConfig, spec: DatasetSpec) -> dict[str, float]:
    return {
        f"{phase_name}_{domain_name}": float(config.phase_weights[phase_name][domain_name])
        for phase_name in spec.phase_names
        for domain_name in spec.domain_names
    }


def _included_subset_sizes(rank: int, subset_sizes: tuple[int, ...]) -> list[int]:
    return [int(size) for size in subset_sizes if size >= rank]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_proxy_launch_plan(output_dir: Path) -> dict[str, Any]:
    return json.loads(_paths(output_dir)["proxy_launch_plan_json"].read_text())


def _plan_proxy(
    *,
    output_dir: Path,
    policy: str,
    subset_sizes: tuple[int, ...],
    data_seed: int,
    row_limit: int | None,
) -> None:
    spec, frame = _load_source_bundle(row_limit=row_limit)
    max_k = max(subset_sizes)
    selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=max_k, seed=0)
    prefix = _proxy_name_prefix(output_dir)

    rows: list[dict[str, Any]] = []
    for rank, source_idx in enumerate(selection.selected_indices, start=1):
        source_row = frame.iloc[int(source_idx)]
        weight_config = _phase_weight_config_from_row(source_row, run_id=PROXY_RUN_ID_BASE + rank, spec=spec)
        rows.append(
            {
                "rank": rank,
                "source_idx": int(source_idx),
                "source_run_id": int(source_row["run_id"]) if "run_id" in source_row else int(source_idx),
                "source_wandb_run_id": str(source_row["wandb_run_id"]) if "wandb_run_id" in source_row else "",
                "selector_seed": 0,
                "included_subset_sizes": _included_subset_sizes(rank, subset_sizes),
                "run_name": f"fbl_wsdba_rank{rank:03d}",
                "weight_config": weight_config.to_dict(),
                **_phase_column_payload(weight_config, spec),
            }
        )

    payload = {
        "dataset": DATASET_NAME,
        "source_policy": SOURCE_POLICY,
        "policy": policy,
        "subset_sizes": list(subset_sizes),
        "name_prefix": prefix,
        "data_seed": data_seed,
        "n_runs": len(rows),
        "runs": rows,
    }
    paths = _paths(output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["plots_dir"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["proxy_launch_plan_json"], payload)

    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["included_subset_sizes"] = json.dumps(row["included_subset_sizes"])
        flat["weight_config"] = json.dumps(row["weight_config"], sort_keys=True)
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(paths["proxy_selection_csv"], index=False)


def _launch_proxy(
    *,
    output_dir: Path,
    data_seed: int,
    dry_run_launches: bool,
) -> None:
    plan = _load_proxy_launch_plan(output_dir)
    if int(plan["data_seed"]) != data_seed:
        raise ValueError(f"Proxy data_seed mismatch: plan={plan['data_seed']} cli={data_seed}")

    if dry_run_launches:
        return

    tokenizer_cache_base = _region_local_marin_path(TOKENIZER_CACHE_BASE)
    eval_datasets_cache_path = _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    experiment = create_two_phase_wsd_boundary_aligned_experiment(
        name=str(plan["name_prefix"]),
        eval_datasets_cache_path=eval_datasets_cache_path,
        resources=ResourceConfig.with_tpu("v5p-8"),
    )
    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=str(plan["name_prefix"]),
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=list(EVAL_TASKS),
        gcs_path=eval_datasets_cache_path,
        name_prefix=str(plan["name_prefix"]),
    )

    configs = [WeightConfig.from_dict(run["weight_config"]) for run in plan["runs"]]
    weight_configs_step = experiment.create_weight_configs_step(
        configs=configs,
        summary={
            "kind": "two_phase_wsd_boundary_aligned_proxy_repeat",
            "source_policy": plan["source_policy"],
            "policy": plan["policy"],
            "subset_sizes": plan["subset_sizes"],
        },
        seed=data_seed,
        name_prefix=str(plan["name_prefix"]),
    )
    training_steps = [
        experiment.create_training_step(
            WeightConfig.from_dict(run["weight_config"]),
            name_prefix=str(plan["name_prefix"]),
            run_name=str(run["run_name"]),
            data_seed=data_seed,
        )
        for run in plan["runs"]
    ]
    with _executor_cli_context():
        executor_main(
            ExecutorMainConfig(max_concurrent=len(training_steps) + 3),
            steps=[cache_tokenizer_step, cache_eval_datasets_step, weight_configs_step, *training_steps],
            description=f"{plan['name_prefix']}: WSD boundary-aligned proxy sweep",
        )


def _collect_proxy_results(
    *,
    output_dir: Path,
    row_limit: int | None,
    wandb_entity: str,
    wandb_project: str,
    skip_wandb: bool,
    collect_from_source_observed: bool,
) -> None:
    spec, frame = _load_source_bundle(row_limit=row_limit)
    del spec
    plan = _load_proxy_launch_plan(output_dir)
    run_names = [f"{plan['name_prefix']}/{run['run_name']}" for run in plan["runs"]]
    actual_metric_map = (
        {}
        if skip_wandb
        else _fetch_actual_metric_map(
            dataset=DATASET_NAME,
            run_names=run_names,
            metric_key=OBJECTIVE_METRIC,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
    )

    rows: list[dict[str, Any]] = []
    for run in plan["runs"]:
        full_run_name = f"{plan['name_prefix']}/{run['run_name']}"
        actual = dict(actual_metric_map.get(full_run_name, {}))
        source_row = frame.iloc[int(run["source_idx"])]
        if actual.get("actual_bpb") is None and collect_from_source_observed:
            actual["actual_bpb"] = float(source_row[OBJECTIVE_METRIC])
            actual["wandb_state"] = "mock_source_observed"
            actual["wandb_url"] = ""
            actual["wandb_run_name"] = full_run_name

        rows.append(
            {
                "dataset": DATASET_NAME,
                "policy": str(plan["policy"]),
                "rank": int(run["rank"]),
                "selector_seed": int(run["selector_seed"]),
                "source_idx": int(run["source_idx"]),
                "source_run_id": int(run["source_run_id"]),
                "source_wandb_run_id": str(run["source_wandb_run_id"]),
                "included_subset_sizes": json.dumps(run["included_subset_sizes"]),
                "run_name": str(run["run_name"]),
                "full_run_name": full_run_name,
                OBJECTIVE_METRIC: actual.get("actual_bpb"),
                "actual_bpb": actual.get("actual_bpb"),
                "wandb_state": actual.get("wandb_state"),
                "wandb_url": actual.get("wandb_url"),
                "wandb_run_name": actual.get("wandb_run_name", full_run_name),
                **{
                    f"{phase_name}_{domain_name}": float(run["weight_config"]["phase_weights"][phase_name][domain_name])
                    for phase_name in TWO_PHASE_STARCODER.phase_names
                    for domain_name in DOMAIN_NAMES
                },
            }
        )
    pd.DataFrame(rows).sort_values("rank").to_csv(_paths(output_dir)["proxy_results_csv"], index=False)


def _successful_proxy_rows(proxy_results: pd.DataFrame) -> pd.DataFrame:
    return proxy_results[proxy_results["actual_bpb"].notna()].sort_values("rank").reset_index(drop=True)


def _contiguous_prefix_length(proxy_results: pd.DataFrame) -> int:
    contiguous = 0
    for row in proxy_results.sort_values("rank").itertuples(index=False):
        if int(row.rank) != contiguous + 1 or pd.isna(row.actual_bpb):
            break
        contiguous += 1
    return contiguous


def _proxy_results_to_spec(base_spec: DatasetSpec, proxy_results: pd.DataFrame) -> DatasetSpec:
    weights = np.zeros((len(proxy_results), base_spec.N, base_spec.M), dtype=float)
    for row_idx, row in enumerate(proxy_results.itertuples(index=False)):
        for phase_idx, phase_name in enumerate(base_spec.phase_names):
            for domain_idx, domain_name in enumerate(base_spec.domain_names):
                weights[row_idx, phase_idx, domain_idx] = float(getattr(row, f"{phase_name}_{domain_name}"))
    return DatasetSpec(
        weights=weights,
        y=proxy_results["actual_bpb"].to_numpy(dtype=float),
        epoch_multipliers=np.asarray(base_spec.epoch_multipliers, dtype=float),
        domain_names=list(base_spec.domain_names),
        phase_names=list(base_spec.phase_names),
        small_domains=list(base_spec.small_domains or []),
        name=f"{base_spec.name}_wsd_boundary_aligned_proxy",
    )


def _sample_simplex_points(rng: np.random.Generator, n_points: int, n_dims: int) -> np.ndarray:
    raw = rng.exponential(1.0, size=(n_points, n_dims))
    return raw / raw.sum(axis=1, keepdims=True)


def _phase_weights_from_point(point: np.ndarray, spec: DatasetSpec) -> dict[str, dict[str, float]]:
    return {
        spec.phase_names[phase_idx]: {
            spec.domain_names[domain_idx]: float(point[phase_idx, domain_idx]) for domain_idx in range(spec.M)
        }
        for phase_idx in range(spec.N)
    }


def _optimize_predicted_optimum(
    predict_fn,
    spec: DatasetSpec,
    *,
    search_seed: int,
    search_points: int,
    opt_restarts: int,
    opt_maxiter: int,
) -> tuple[dict[str, dict[str, float]], float, int | None, float | None]:
    rng = np.random.default_rng(search_seed)
    points = np.zeros((search_points, spec.N, spec.M), dtype=float)
    for phase_idx in range(spec.N):
        points[:, phase_idx, :] = _sample_simplex_points(rng, search_points, spec.M)

    pred = np.asarray(predict_fn(points), dtype=float)
    finite_mask = np.isfinite(pred)
    if not finite_mask.any():
        phase_weights = _phase_weights_from_point(points[0], spec)
        return phase_weights, float("inf"), None, None

    finite_indices = np.flatnonzero(finite_mask)
    best_idx = int(finite_indices[np.argmin(pred[finite_mask])])
    best_point = points[best_idx].copy()
    best_pred = float(pred[best_idx])

    if spec.M == 2:
        small_domain_idx = spec.small_domains[0] if spec.small_domains else 1
        other_domain_idx = 1 - small_domain_idx

        def point_from_small_weights(small_weights: np.ndarray) -> np.ndarray:
            point = np.zeros((spec.N, spec.M), dtype=float)
            point[:, small_domain_idx] = small_weights
            point[:, other_domain_idx] = 1.0 - small_weights
            return point

        def objective(small_weights: np.ndarray) -> float:
            clipped = np.clip(np.asarray(small_weights, dtype=float), 0.0, 1.0)
            point = point_from_small_weights(clipped)
            values = np.asarray(predict_fn(point[None, :, :]), dtype=float)
            if values.shape != (1,) or not np.isfinite(values[0]):
                return float("inf")
            return float(values[0])

        starts = [best_point[:, small_domain_idx]]
        starts.extend(rng.uniform(0.0, 1.0, size=(max(opt_restarts - 1, 0), spec.N)))
        for x0 in starts:
            try:
                result = minimize(
                    objective,
                    x0=np.asarray(x0, dtype=float),
                    method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * spec.N,
                    options={"maxiter": opt_maxiter},
                )
            except Exception:
                continue
            candidate_pred = objective(result.x)
            if np.isfinite(candidate_pred) and candidate_pred < best_pred:
                best_pred = candidate_pred
                best_point = point_from_small_weights(result.x)

    replay = replay_proposals_to_observed(best_point[None, :, :], spec.weights)
    nearest_rank = int(replay.selected_indices[0]) if replay.selected_indices else None
    nearest_distance = replay.mean_distance if nearest_rank is not None else None
    return _phase_weights_from_point(best_point, spec), best_pred, nearest_rank, nearest_distance


def _dsre_model():
    return next(model for model in GENERAL_MODELS if model.name == OBJECTIVE_MODEL)


def _median_numeric(rows: list[dict[str, float]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _evaluate_subset_prefix(
    *,
    full_spec: DatasetSpec,
    proxy_results: pd.DataFrame,
    subset_size: int,
    policy: str,
    dsre_fit_seeds: tuple[int, ...],
    dsre_restarts: int,
    dsre_maxiter: int,
    opt_search_points: int,
    opt_restarts: int,
    opt_maxiter: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    huber_delta = _huber_delta(full_spec.y)
    model = _dsre_model()
    train_spec = full_spec.subset(np.arange(subset_size, dtype=int))
    selected_source_indices = proxy_results["source_idx"].iloc[:subset_size].astype(int).tolist()

    per_fit_rows: list[dict[str, float]] = []
    per_fit_optima: list[tuple[int, dict[str, dict[str, float]], float, int | None, float | None]] = []

    for fit_seed in dsre_fit_seeds:
        try:
            predict_fn, _ = model.fit_fn(train_spec, seed=fit_seed, n_restarts=dsre_restarts, maxiter=dsre_maxiter)
            preds = np.asarray(predict_fn(full_spec.weights), dtype=float)
            if preds.shape != (full_spec.R,) or not np.isfinite(preds).all():
                raise ValueError(f"Invalid predictions: {preds.shape}")

            residuals = full_spec.y - preds
            chosen_idx = int(np.argmin(preds))
            best_idx = int(np.argmin(full_spec.y))
            top5 = np.argsort(preds)[: min(5, len(preds))]
            true_ranks = np.argsort(np.argsort(full_spec.y)) + 1
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((full_spec.y - np.mean(full_spec.y)) ** 2))
            optimum = _optimize_predicted_optimum(
                predict_fn,
                full_spec,
                search_seed=fit_seed + subset_size * 1_000,
                search_points=opt_search_points,
                opt_restarts=opt_restarts,
                opt_maxiter=opt_maxiter,
            )
            per_fit_rows.append(
                {
                    "fit_seed": float(fit_seed),
                    "success": 1.0,
                    "regret@1": float(full_spec.y[chosen_idx] - full_spec.y[best_idx]),
                    "regret@5": float(np.min(full_spec.y[top5]) - full_spec.y[best_idx]),
                    "chosen_true_rank": float(true_ranks[chosen_idx]),
                    "Spearman": float(spearmanr(full_spec.y, preds)[0]),
                    "R²": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    "Huber": float(
                        np.mean(
                            np.where(
                                np.abs(residuals) <= huber_delta,
                                0.5 * residuals**2,
                                huber_delta * (np.abs(residuals) - 0.5 * huber_delta),
                            )
                        )
                    ),
                    "RMSE": float(np.sqrt(np.mean(residuals**2))),
                }
            )
            per_fit_optima.append((fit_seed, *optimum))
        except Exception:
            per_fit_rows.append(
                {
                    "fit_seed": float(fit_seed),
                    "success": 0.0,
                    "regret@1": float("inf"),
                    "regret@5": float("inf"),
                    "chosen_true_rank": float("inf"),
                    "Spearman": float("nan"),
                    "R²": float("nan"),
                    "Huber": float("inf"),
                    "RMSE": float("inf"),
                }
            )

    success_rows = [row for row in per_fit_rows if row["success"] > 0]
    score_row = {
        "dataset": DATASET_NAME,
        "mode": OBJECTIVE_MODE,
        "policy": policy,
        "subset_size": subset_size,
        "selector_seed": 0,
        "evaluation_model": OBJECTIVE_MODEL,
        "success_rate": float(np.mean([row["success"] for row in per_fit_rows])),
        "n_records": 1,
        "selected_indices": json.dumps(selected_source_indices),
    }
    if not success_rows:
        score_row.update(
            {
                "R²_median": float("nan"),
                "R²_q25": float("nan"),
                "R²_q75": float("nan"),
                "regret@1_median": float("inf"),
                "regret@1_q25": float("inf"),
                "regret@1_q75": float("inf"),
                "Huber_median": float("inf"),
                "Huber_q25": float("inf"),
                "Huber_q75": float("inf"),
                "RMSE_median": float("inf"),
                "RMSE_q25": float("inf"),
                "RMSE_q75": float("inf"),
                "regret@5_median": float("inf"),
                "regret@5_q25": float("inf"),
                "regret@5_q75": float("inf"),
                "chosen_true_rank_median": float("inf"),
                "chosen_true_rank_q25": float("inf"),
                "chosen_true_rank_q75": float("inf"),
                "Spearman_median": float("nan"),
                "Spearman_q25": float("nan"),
                "Spearman_q75": float("nan"),
            }
        )
        return score_row, None

    median_regret = _median_numeric(success_rows, "regret@1")
    representative = min(
        success_rows,
        key=lambda row: (abs(float(row["regret@1"]) - median_regret), int(row["fit_seed"])),
    )
    representative_fit_seed = int(representative["fit_seed"])
    representative_optimum = next(optimum for optimum in per_fit_optima if optimum[0] == representative_fit_seed)
    phase_weights, predicted_objective, nearest_rank, nearest_distance = representative_optimum[1:]
    nearest_source_idx = int(proxy_results.iloc[nearest_rank]["source_idx"]) if nearest_rank is not None else np.nan

    score_row.update(
        {
            "R²_median": _median_numeric(success_rows, "R²"),
            "R²_q25": _median_numeric(success_rows, "R²"),
            "R²_q75": _median_numeric(success_rows, "R²"),
            "regret@1_median": _median_numeric(success_rows, "regret@1"),
            "regret@1_q25": _median_numeric(success_rows, "regret@1"),
            "regret@1_q75": _median_numeric(success_rows, "regret@1"),
            "Huber_median": _median_numeric(success_rows, "Huber"),
            "Huber_q25": _median_numeric(success_rows, "Huber"),
            "Huber_q75": _median_numeric(success_rows, "Huber"),
            "RMSE_median": _median_numeric(success_rows, "RMSE"),
            "RMSE_q25": _median_numeric(success_rows, "RMSE"),
            "RMSE_q75": _median_numeric(success_rows, "RMSE"),
            "regret@5_median": _median_numeric(success_rows, "regret@5"),
            "regret@5_q25": _median_numeric(success_rows, "regret@5"),
            "regret@5_q75": _median_numeric(success_rows, "regret@5"),
            "chosen_true_rank_median": _median_numeric(success_rows, "chosen_true_rank"),
            "chosen_true_rank_q25": _median_numeric(success_rows, "chosen_true_rank"),
            "chosen_true_rank_q75": _median_numeric(success_rows, "chosen_true_rank"),
            "Spearman_median": _median_numeric(success_rows, "Spearman"),
            "Spearman_q25": _median_numeric(success_rows, "Spearman"),
            "Spearman_q75": _median_numeric(success_rows, "Spearman"),
        }
    )
    optimum_row = {
        "dataset": DATASET_NAME,
        "mode": OBJECTIVE_MODE,
        "policy": policy,
        "subset_size": subset_size,
        "selector_seed": 0,
        "evaluation_model": OBJECTIVE_MODEL,
        "representative_fit_seed": representative_fit_seed,
        "predicted_objective": float(predicted_objective),
        "nearest_observed_idx": nearest_source_idx,
        "nearest_observed_distance": nearest_distance,
        "selected_indices": json.dumps(selected_source_indices),
        **{
            f"{phase_name}_{domain_name}": float(phase_weights[phase_name][domain_name])
            for phase_name in full_spec.phase_names
            for domain_name in full_spec.domain_names
        },
    }
    return score_row, optimum_row


def _evaluate_subset_prefix_from_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _evaluate_subset_prefix(**kwargs)


def _write_predicted_optima_jsonl(predicted_optima: pd.DataFrame, output_path: Path) -> None:
    records = predicted_optima.to_dict(orient="records")
    with output_path.open("w") as handle:
        for run_id, record in enumerate(records):
            phase_names = sorted(
                {"_".join(key.split("_", 2)[:2]) for key in record if key.startswith("phase_") and pd.notna(record[key])}
            )
            phase_weights = {
                phase_name: {
                    domain_name: float(record[f"{phase_name}_{domain_name}"])
                    for domain_name in DOMAIN_NAMES
                    if f"{phase_name}_{domain_name}" in record and pd.notna(record[f"{phase_name}_{domain_name}"])
                }
                for phase_name in phase_names
            }
            payload = {
                "dataset": record["dataset"],
                "mode": record["mode"],
                "policy": record["policy"],
                "subset_size": int(record["subset_size"]),
                "selector_seed": int(record["selector_seed"]),
                "evaluation_model": record["evaluation_model"],
                "weight_config": WeightConfig(run_id=run_id, phase_weights=phase_weights).to_dict(),
                "predicted_objective": float(record["predicted_objective"]),
                "nearest_observed_idx": (
                    None if pd.isna(record["nearest_observed_idx"]) else int(record["nearest_observed_idx"])
                ),
                "nearest_observed_distance": (
                    None if pd.isna(record["nearest_observed_distance"]) else float(record["nearest_observed_distance"])
                ),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _fit_predict(
    *,
    output_dir: Path,
    policy: str,
    subset_sizes: tuple[int, ...],
    row_limit: int | None,
    workers: int,
    dsre_fit_seeds: tuple[int, ...],
    dsre_restarts: int,
    dsre_maxiter: int,
    opt_search_points: int,
    opt_restarts: int,
    opt_maxiter: int,
) -> None:
    base_spec, _ = _load_source_bundle(row_limit=row_limit)
    proxy_results = pd.read_csv(_paths(output_dir)["proxy_results_csv"]).sort_values("rank").reset_index(drop=True)
    successful_rows = _successful_proxy_rows(proxy_results)
    contiguous_prefix = _contiguous_prefix_length(proxy_results)
    ready_subset_sizes = tuple(size for size in subset_sizes if size <= contiguous_prefix)
    fit_status = {
        "dataset": DATASET_NAME,
        "policy": policy,
        "proxy_successful_rows": len(successful_rows),
        "contiguous_prefix_length": contiguous_prefix,
        "requested_subset_sizes": list(subset_sizes),
        "ready_subset_sizes": list(ready_subset_sizes),
        "pending_subset_sizes": [int(size) for size in subset_sizes if size > contiguous_prefix],
    }
    _write_json(_paths(output_dir)["fit_status_json"], fit_status)

    if not ready_subset_sizes:
        pd.DataFrame().to_csv(_paths(output_dir)["model_scores_csv"], index=False)
        pd.DataFrame().to_csv(_paths(output_dir)["curve_points_csv"], index=False)
        pd.DataFrame().to_csv(_paths(output_dir)["predicted_optima_csv"], index=False)
        _paths(output_dir)["predicted_optima_jsonl"].write_text("")
        return

    full_spec = _proxy_results_to_spec(base_spec, successful_rows)
    task_kwargs = [
        {
            "full_spec": full_spec,
            "proxy_results": successful_rows,
            "subset_size": subset_size,
            "policy": policy,
            "dsre_fit_seeds": dsre_fit_seeds,
            "dsre_restarts": dsre_restarts,
            "dsre_maxiter": dsre_maxiter,
            "opt_search_points": opt_search_points,
            "opt_restarts": opt_restarts,
            "opt_maxiter": opt_maxiter,
        }
        for subset_size in ready_subset_sizes
    ]

    if workers > 1 and len(task_kwargs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(task_kwargs))) as executor:
            task_results = list(executor.map(_evaluate_subset_prefix_from_kwargs, task_kwargs))
    else:
        task_results = [_evaluate_subset_prefix(**kwargs) for kwargs in task_kwargs]

    model_score_rows: list[dict[str, Any]] = []
    predicted_optimum_rows: list[dict[str, Any]] = []
    for score_row, optimum_row in task_results:
        model_score_rows.append(score_row)
        if optimum_row is not None:
            predicted_optimum_rows.append(optimum_row)

    model_scores = pd.DataFrame(model_score_rows).sort_values("subset_size").reset_index(drop=True)
    predicted_optima = pd.DataFrame(predicted_optimum_rows).sort_values("subset_size").reset_index(drop=True)
    model_scores.to_csv(_paths(output_dir)["model_scores_csv"], index=False)
    model_scores.to_csv(_paths(output_dir)["curve_points_csv"], index=False)
    predicted_optima.to_csv(_paths(output_dir)["predicted_optima_csv"], index=False)
    _write_predicted_optima_jsonl(predicted_optima, _paths(output_dir)["predicted_optima_jsonl"])


def _load_predicted_optima_records(output_dir: Path, *, policy: str) -> list[dict[str, Any]]:
    records = []
    path = _paths(output_dir)["predicted_optima_jsonl"]
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record["dataset"] == DATASET_NAME
                and record["policy"] == policy
                and record["evaluation_model"] == OBJECTIVE_MODEL
            ):
                records.append(record)
    records.sort(key=lambda row: int(row["subset_size"]))
    return records


def _launch_validation(
    *,
    output_dir: Path,
    policy: str,
    data_seed: int,
    dry_run_launches: bool,
) -> None:
    records = _load_predicted_optima_records(output_dir, policy=policy)
    prefix = _validation_name_prefix(output_dir)
    runs = []
    for record in records:
        subset_size = int(record["subset_size"])
        source = WeightConfig.from_dict(record["weight_config"])
        weight_config = WeightConfig(
            run_id=VALIDATION_RUN_ID_BASE + subset_size,
            phase_weights=source.phase_weights,
        )
        runs.append(
            {
                "subset_size": subset_size,
                "selector_seed": int(record["selector_seed"]),
                "run_name": f"fbl_wsdba_k{subset_size:03d}_optimum",
                "weight_config": weight_config.to_dict(),
            }
        )

    payload = {
        "dataset": DATASET_NAME,
        "benchmark_output_dir": str(output_dir),
        "policy": policy,
        "name_prefix": prefix,
        "n_runs": len(runs),
        "runs": runs,
    }
    _write_json(_paths(output_dir)["validation_launch_plan_json"], payload)
    if dry_run_launches or not runs:
        return

    tokenizer_cache_base = _region_local_marin_path(TOKENIZER_CACHE_BASE)
    eval_datasets_cache_path = _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    experiment = create_two_phase_wsd_boundary_aligned_experiment(
        name=prefix,
        eval_datasets_cache_path=eval_datasets_cache_path,
        resources=ResourceConfig.with_tpu("v5p-8"),
    )
    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=list(EVAL_TASKS),
        gcs_path=eval_datasets_cache_path,
        name_prefix=prefix,
    )
    configs = [WeightConfig.from_dict(run["weight_config"]) for run in runs]
    weight_configs_step = experiment.create_weight_configs_step(
        configs=configs,
        summary={
            "source": "two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_repeat",
            "dataset": DATASET_NAME,
            "policy": policy,
            "subset_sizes": [int(run["subset_size"]) for run in runs],
            "benchmark_output_dir": str(output_dir),
        },
        seed=data_seed,
        name_prefix=prefix,
    )
    training_steps = [
        experiment.create_training_step(
            WeightConfig.from_dict(run["weight_config"]),
            name_prefix=prefix,
            run_name=str(run["run_name"]),
            data_seed=data_seed,
        )
        for run in runs
    ]
    with _executor_cli_context():
        executor_main(
            ExecutorMainConfig(max_concurrent=len(training_steps) + 3),
            steps=[cache_tokenizer_step, cache_eval_datasets_step, weight_configs_step, *training_steps],
            description=f"{prefix}: WSD boundary-aligned optimum validation runs ({policy})",
        )


def _plot_results(
    *,
    output_dir: Path,
    policy: str,
    wandb_entity: str,
    wandb_project: str,
    skip_wandb: bool,
) -> None:
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--benchmark-output-dir",
        str(output_dir),
        "--dataset",
        DATASET_NAME,
        "--policy",
        policy,
        "--wandb-entity",
        wandb_entity,
        "--wandb-project",
        wandb_project,
        "--observed-reference-csv",
        str(_paths(output_dir)["proxy_results_csv"]),
    ]
    if skip_wandb:
        command.append("--skip-wandb")
    subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("plan_proxy", "launch_proxy", "collect_proxy", "fit_predict", "launch_validation", "plot", "all"),
        required=True,
    )
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--subset-sizes", type=str, default="")
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--wandb-entity", type=str, default="marin-community")
    parser.add_argument("--wandb-project", type=str, default="marin")
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--dry-run-launches", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--collect-from-source-observed", action="store_true")
    parser.add_argument("--dsre-fit-seeds", type=int, default=len(DEFAULT_DSRE_FIT_SEEDS))
    parser.add_argument("--dsre-restarts", type=int, default=DEFAULT_DSRE_RESTARTS)
    parser.add_argument("--dsre-maxiter", type=int, default=DEFAULT_DSRE_MAXITER)
    parser.add_argument("--opt-search-points", type=int, default=DEFAULT_OPT_SEARCH_POINTS)
    parser.add_argument("--opt-restarts", type=int, default=DEFAULT_OPT_RESTARTS)
    parser.add_argument("--opt-maxiter", type=int, default=DEFAULT_OPT_MAXITER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_spec, _ = _load_source_bundle(row_limit=args.row_limit)
    subset_sizes = _parse_subset_sizes(args.subset_sizes, max_size=source_spec.R)
    if not subset_sizes:
        raise ValueError("No subset sizes remain after clipping to the available source rows")

    output_dir = args.output_dir.resolve()
    paths = _paths(output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["plots_dir"].mkdir(parents=True, exist_ok=True)

    if args.stage in {"plan_proxy", "all"}:
        _plan_proxy(
            output_dir=output_dir,
            policy=args.policy,
            subset_sizes=subset_sizes,
            data_seed=args.data_seed,
            row_limit=args.row_limit,
        )
    if args.stage in {"launch_proxy", "all"}:
        _launch_proxy(
            output_dir=output_dir,
            data_seed=args.data_seed,
            dry_run_launches=args.dry_run_launches,
        )
    if args.stage in {"collect_proxy", "all"}:
        _collect_proxy_results(
            output_dir=output_dir,
            row_limit=args.row_limit,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            skip_wandb=args.skip_wandb,
            collect_from_source_observed=args.collect_from_source_observed,
        )
    if args.stage in {"fit_predict", "all"}:
        _fit_predict(
            output_dir=output_dir,
            policy=args.policy,
            subset_sizes=subset_sizes,
            row_limit=args.row_limit,
            workers=max(1, args.workers),
            dsre_fit_seeds=tuple(range(max(1, args.dsre_fit_seeds))),
            dsre_restarts=max(1, args.dsre_restarts),
            dsre_maxiter=max(1, args.dsre_maxiter),
            opt_search_points=max(64, args.opt_search_points),
            opt_restarts=max(1, args.opt_restarts),
            opt_maxiter=max(1, args.opt_maxiter),
        )
    if args.stage in {"launch_validation", "all"}:
        _launch_validation(
            output_dir=output_dir,
            policy=args.policy,
            data_seed=args.data_seed,
            dry_run_launches=args.dry_run_launches,
        )
    if args.stage in {"plot", "all"}:
        _plot_results(
            output_dir=output_dir,
            policy=args.policy,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            skip_wandb=args.skip_wandb,
        )


if __name__ == "__main__":
    main()
