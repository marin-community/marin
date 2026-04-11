# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Benchmark generic static selectors on StarCoder subset fitting and replay.

This script evaluates:
- retrospective observed-pool subset selection over a fixed sample-size grid
- prospective replay policies at the recommended k for each dataset

Artifacts:
- selection_records.csv
- model_scores.csv
- curve_points.csv
- selector_summary.csv
- predicted_optima.csv
- predicted_optima.jsonl
- plots/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "levanter" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "iris" / "src"))

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    GENERAL_MODELS,
    DatasetSpec,
    _huber_delta,
)
from experiments.domain_phase_mix.starcoder_metadata import (
    THREE_PHASE_STARCODER,
    TWO_PHASE_STARCODER,
    load_starcoder_dataset,
)
from experiments.domain_phase_mix.static_batch_selection import (
    prospective_d_optimal_selection,
    prospective_generic_selection,
    replay_proposals_to_observed,
    retrospective_d_optimal_selection,
    retrospective_generic_selection,
    weight_configs_to_tensor,
)
from experiments.domain_phase_mix.three_phase_starcoder_experiment import create_three_phase_experiment
from experiments.domain_phase_mix.two_phase_starcoder_experiment import create_two_phase_experiment

OUTPUT_ROOT = SCRIPT_DIR / "starcoder_generic_selector_outputs"
PLOTS_DIRNAME = "plots"
SUBSET_SCRIPT = SCRIPT_DIR / "starcoder_subset_selection.py"
SUBSET_OUTPUT_ROOT = SCRIPT_DIR / "starcoder_subset_selection_outputs"
TWO_PHASE_ORACLE_BANK_DIR = SUBSET_OUTPUT_ROOT / "two_phase_oracle_k10_top100_proc"
THREE_PHASE_ORACLE_BANK_DIR = SUBSET_OUTPUT_ROOT / "three_phase_oracle_k16_longrun_proc2_20260308"

COMMITTEE_MODELS = ("DS-RE-CEQ", "CES-Overfit", "BayesLogQuad(w-e)")
PLOT_MODEL = "DS-RE-CEQ"
CURVE_POLICIES = (
    "random_observed",
    "feature_maximin_observed",
    "feature_dpp_observed",
    "feature_bayes_linear_observed",
)
SUBSET_SIZE_GRID = (4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128)
RECOMMENDED_K = {
    TWO_PHASE_STARCODER.name: 10,
    THREE_PHASE_STARCODER.name: 16,
}
RANDOM_BOOTSTRAP_SEEDS = 64
RETRO_DOPT_SEEDS = 8
PROSPECTIVE_SEEDS = 16
PROSPECTIVE_POOL_SIZE = 2048
DSRE_FIT_SEEDS = (0, 1, 2)
DSRE_RESTARTS = 4
DSRE_MAXITER = 300
OPT_SEARCH_POINTS = 4096
OPT_RESTARTS = 8
OPT_MAXITER = 200
ORACLE_BANK_WORKERS = 14

POLICY_LABELS = {
    "random_observed": "Random",
    "doptimal_observed": "DS-RE-CEQ D-opt",
    "feature_maximin_observed": "Feature Maximin",
    "feature_dpp_observed": "Feature DPP",
    "feature_bayes_linear_observed": "Feature Bayes Linear",
    "sampler_replay": "Sampler Replay",
    "doptimal_replay": "DS-RE-CEQ D-opt Replay",
    "feature_maximin_replay": "Feature Maximin Replay",
    "feature_dpp_replay": "Feature DPP Replay",
    "feature_bayes_linear_replay": "Feature Bayes Linear Replay",
}
POLICY_COLORS = {
    "random_observed": "#6b7280",
    "feature_maximin_observed": "#c2410c",
    "feature_dpp_observed": "#0f766e",
    "feature_bayes_linear_observed": "#1d4ed8",
}
DATASET_LINESTYLES = {
    TWO_PHASE_STARCODER.name: "-",
    THREE_PHASE_STARCODER.name: "--",
}
DATASET_LABELS = {
    TWO_PHASE_STARCODER.name: "2-phase",
    THREE_PHASE_STARCODER.name: "3-phase",
}


@dataclass(frozen=True)
class SelectionRecord:
    dataset: str
    mode: str
    policy: str
    subset_size: int
    selector_seed: int
    selected_indices: tuple[int, ...]
    selected_configs_json: str
    replay_mean_distance: float
    replay_max_distance: float
    diagnostics_json: str


@dataclass(frozen=True)
class PredictedOptimum:
    phase_weights: dict[str, dict[str, float]]
    predicted_objective: float
    nearest_observed_idx: int | None
    nearest_observed_distance: float | None


def _model_map() -> dict[str, Any]:
    return {model.name: model for model in GENERAL_MODELS}


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(14, max(1, cpu_count - 2))


def _stable_seed(*parts: int) -> int:
    seed = 0
    for part in parts:
        seed = (seed * 1315423911 + int(part) + 0x9E3779B9) & 0xFFFFFFFF
    return int(seed)


def _subset_sizes_for_spec(spec: DatasetSpec, override: tuple[int, ...] | None = None) -> tuple[int, ...]:
    sizes = override or SUBSET_SIZE_GRID
    clipped = sorted({int(size) for size in sizes if 0 < int(size) <= spec.R})
    return tuple(clipped)


def _slice_spec(spec: DatasetSpec, frame: pd.DataFrame, row_limit: int | None) -> tuple[DatasetSpec, pd.DataFrame]:
    if row_limit is None or row_limit >= spec.R:
        return spec, frame
    indices = np.arange(row_limit, dtype=int)
    return spec.subset(indices), frame.iloc[:row_limit].reset_index(drop=True)


def _flatten_phase_weights(phase_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{phase_name}_{domain_name}": float(weight)
        for phase_name, domains in phase_weights.items()
        for domain_name, weight in domains.items()
    }


def _weight_config_json(configs: list[WeightConfig]) -> str:
    return json.dumps([config.to_dict() for config in configs], sort_keys=True)


def _record_to_dict(record: SelectionRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["selected_indices"] = json.dumps(list(record.selected_indices))
    payload["selected_configs_json"] = record.selected_configs_json
    payload.update(json.loads(record.diagnostics_json))
    return payload


def _selection_from_indices(
    *,
    dataset: str,
    mode: str,
    policy: str,
    subset_size: int,
    selector_seed: int,
    selected_indices: Sequence[int],
    selected_configs_json: str = "[]",
    replay_mean_distance: float = 0.0,
    replay_max_distance: float = 0.0,
    diagnostics: dict[str, float] | None = None,
) -> SelectionRecord:
    return SelectionRecord(
        dataset=dataset,
        mode=mode,
        policy=policy,
        subset_size=subset_size,
        selector_seed=selector_seed,
        selected_indices=tuple(int(idx) for idx in selected_indices),
        selected_configs_json=selected_configs_json,
        replay_mean_distance=float(replay_mean_distance),
        replay_max_distance=float(replay_max_distance),
        diagnostics_json=json.dumps(diagnostics or {}, sort_keys=True),
    )


def _load_dataset_bundle(dataset_name: str, row_limit: int | None = None) -> tuple[DatasetSpec, pd.DataFrame, Any]:
    if dataset_name == TWO_PHASE_STARCODER.name:
        spec, frame = load_starcoder_dataset(TWO_PHASE_STARCODER)
        experiment = create_two_phase_experiment()
    elif dataset_name == THREE_PHASE_STARCODER.name:
        spec, frame = load_starcoder_dataset(THREE_PHASE_STARCODER)
        experiment = create_three_phase_experiment()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    sliced_spec, sliced_frame = _slice_spec(spec, frame, row_limit)
    return sliced_spec, sliced_frame, experiment


def _create_experiment_for_dataset(dataset_name: str) -> Any:
    if dataset_name == TWO_PHASE_STARCODER.name:
        return create_two_phase_experiment()
    if dataset_name == THREE_PHASE_STARCODER.name:
        return create_three_phase_experiment()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def ensure_two_phase_oracle_bank(*, workers: int, skip: bool = False) -> Path | None:
    """Ensure the canonical two-phase oracle bank with top_search_candidates exists."""
    if skip:
        return None
    target = TWO_PHASE_ORACLE_BANK_DIR / "top_search_candidates.csv"
    if target.exists():
        return TWO_PHASE_ORACLE_BANK_DIR

    TWO_PHASE_ORACLE_BANK_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SUBSET_SCRIPT),
        "--output-dir",
        str(TWO_PHASE_ORACLE_BANK_DIR),
        "--datasets",
        TWO_PHASE_STARCODER.name,
        "--k-values",
        "10",
        "--workers",
        str(workers),
        "--search-random-subsets",
        "96",
        "--search-kcenter-subsets",
        "12",
        "--search-dopt-subsets",
        "12",
        "--search-top-subsets",
        "12",
        "--search-top-final",
        "24",
        "--hill-climb-max-accepted",
        "20",
        "--hill-climb-swap-candidates",
        "12",
        "--search-restarts",
        "4",
        "--search-maxiter",
        "300",
        "--final-restarts",
        "8",
        "--final-maxiter",
        "500",
        "--final-seeds",
        "3",
        "--save-top-search-n",
        "100",
        "--parallel-backend",
        "process",
        "--skip-deployable",
    ]
    print("Running two-phase oracle-bank refresh:", " ".join(command))
    subprocess.run(command, check=True)
    return TWO_PHASE_ORACLE_BANK_DIR


def _random_subset_record(spec: DatasetSpec, dataset: str, subset_size: int, selector_seed: int) -> SelectionRecord:
    dataset_id = 0 if dataset == TWO_PHASE_STARCODER.name else 1
    rng = np.random.default_rng(_stable_seed(17, subset_size, selector_seed, dataset_id))
    selected = tuple(sorted(int(idx) for idx in rng.choice(spec.R, size=subset_size, replace=False)))
    return _selection_from_indices(
        dataset=dataset,
        mode="retrospective",
        policy="random_observed",
        subset_size=subset_size,
        selector_seed=selector_seed,
        selected_indices=selected,
    )


def _generate_retrospective_records(
    spec: DatasetSpec,
    *,
    dataset: str,
    subset_sizes: Sequence[int],
    random_bootstrap_seeds: int,
    retrospective_dopt_seeds: int,
    workers: int,
) -> list[SelectionRecord]:
    payloads = [
        (spec, dataset, int(subset_size), random_bootstrap_seeds, retrospective_dopt_seeds)
        for subset_size in subset_sizes
    ]
    if workers <= 1 or len(payloads) <= 1:
        result_groups = [_retrospective_subset_records(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            result_groups = list(executor.map(_retrospective_subset_records, payloads))

    records: list[SelectionRecord] = []
    for group in result_groups:
        records.extend(group)
    return records


def _retrospective_subset_records(
    payload: tuple[DatasetSpec, str, int, int, int],
) -> list[SelectionRecord]:
    spec, dataset, subset_size, random_bootstrap_seeds, retrospective_dopt_seeds = payload
    records: list[SelectionRecord] = []
    recommended_k = RECOMMENDED_K[dataset]
    for selector_seed in range(random_bootstrap_seeds):
        records.append(_random_subset_record(spec, dataset, subset_size, selector_seed))

    for policy, method in (
        ("feature_maximin_observed", "feature_maximin"),
        ("feature_dpp_observed", "feature_dpp"),
        ("feature_bayes_linear_observed", "feature_bayes_linear"),
    ):
        selection = retrospective_generic_selection(spec, method=method, k=subset_size, seed=0)
        records.append(
            _selection_from_indices(
                dataset=dataset,
                mode="retrospective",
                policy=policy,
                subset_size=subset_size,
                selector_seed=0,
                selected_indices=selection.selected_indices,
                diagnostics=selection.diagnostics,
            )
        )

    if subset_size == recommended_k:
        for selector_seed in range(retrospective_dopt_seeds):
            selection = retrospective_d_optimal_selection(spec, k=subset_size, seed=selector_seed)
            records.append(
                _selection_from_indices(
                    dataset=dataset,
                    mode="retrospective",
                    policy="doptimal_observed",
                    subset_size=subset_size,
                    selector_seed=selector_seed,
                    selected_indices=selection.selected_indices,
                    diagnostics=selection.diagnostics,
                )
            )
    return records


def _generate_prospective_records(
    spec: DatasetSpec,
    experiment: Any,
    *,
    dataset: str,
    prospective_seeds: int,
    pool_size: int,
    workers: int,
) -> list[SelectionRecord]:
    del experiment

    payloads = [(spec, dataset, selector_seed, pool_size) for selector_seed in range(prospective_seeds)]
    if workers <= 1 or len(payloads) <= 1:
        result_groups = [_prospective_seed_records(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            result_groups = list(executor.map(_prospective_seed_records, payloads))

    records: list[SelectionRecord] = []
    for group in result_groups:
        records.extend(group)
    return records


def _proposals_to_record(
    *,
    dataset: str,
    policy: str,
    subset_size: int,
    selector_seed: int,
    proposals: list[WeightConfig],
    spec: DatasetSpec,
    diagnostics: dict[str, float] | None = None,
) -> SelectionRecord:
    proposal_weights = weight_configs_to_tensor(
        proposals,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    replay = replay_proposals_to_observed(proposal_weights, spec.weights)
    return _selection_from_indices(
        dataset=dataset,
        mode="prospective",
        policy=policy,
        subset_size=subset_size,
        selector_seed=selector_seed,
        selected_indices=replay.selected_indices,
        selected_configs_json=_weight_config_json(proposals),
        replay_mean_distance=replay.mean_distance,
        replay_max_distance=replay.max_distance,
        diagnostics=diagnostics,
    )


def _prospective_seed_records(payload: tuple[DatasetSpec, str, int, int]) -> list[SelectionRecord]:
    spec, dataset, selector_seed, pool_size = payload
    experiment = _create_experiment_for_dataset(dataset)
    subset_size = RECOMMENDED_K[dataset]
    records: list[SelectionRecord] = []

    sampler = experiment.create_weight_sampler(seed=selector_seed)
    sampler_configs = sampler.sample_n_configs(subset_size, deduplicate=True)
    records.append(
        _proposals_to_record(
            dataset=dataset,
            policy="sampler_replay",
            subset_size=subset_size,
            selector_seed=selector_seed,
            proposals=sampler_configs,
            spec=spec,
        )
    )

    dopt_configs, dopt_selection = prospective_d_optimal_selection(
        spec,
        experiment,
        n_select=subset_size,
        seed=selector_seed,
        pool_size=pool_size,
    )
    records.append(
        _proposals_to_record(
            dataset=dataset,
            policy="doptimal_replay",
            subset_size=subset_size,
            selector_seed=selector_seed,
            proposals=dopt_configs,
            spec=spec,
            diagnostics=dopt_selection.diagnostics,
        )
    )

    for policy, method in (
        ("feature_maximin_replay", "feature_maximin"),
        ("feature_dpp_replay", "feature_dpp"),
        ("feature_bayes_linear_replay", "feature_bayes_linear"),
    ):
        configs, selection = prospective_generic_selection(
            spec,
            experiment,
            method=method,
            n_select=subset_size,
            seed=selector_seed,
            pool_size=pool_size,
        )
        records.append(
            _proposals_to_record(
                dataset=dataset,
                policy=policy,
                subset_size=subset_size,
                selector_seed=selector_seed,
                proposals=configs,
                spec=spec,
                diagnostics=selection.diagnostics,
            )
        )
    return records


def _safe_huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    return float(np.mean(np.where(abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta))))


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


def _fit_predict_function(model_name: str, train_spec: DatasetSpec, fit_seed: int):
    model = _model_map()[model_name]
    if model_name == "DS-RE-CEQ":
        return model.fit_fn(train_spec, seed=fit_seed, n_restarts=DSRE_RESTARTS, maxiter=DSRE_MAXITER)
    return model.fit_fn(train_spec)


def _optimize_predicted_optimum(
    predict_fn,
    spec: DatasetSpec,
    *,
    search_seed: int,
) -> PredictedOptimum:
    rng = np.random.default_rng(search_seed)
    points = np.zeros((OPT_SEARCH_POINTS, spec.N, spec.M), dtype=float)
    for phase_idx in range(spec.N):
        points[:, phase_idx, :] = _sample_simplex_points(rng, OPT_SEARCH_POINTS, spec.M)

    pred = np.asarray(predict_fn(points), dtype=float)
    finite_mask = np.isfinite(pred)
    if not finite_mask.any():
        return PredictedOptimum(
            phase_weights=_phase_weights_from_point(points[0], spec),
            predicted_objective=float("inf"),
            nearest_observed_idx=None,
            nearest_observed_distance=None,
        )

    finite_indices = np.flatnonzero(finite_mask)
    best_idx = int(finite_indices[np.argmin(pred[finite_mask])])
    best_point = points[best_idx].copy()
    best_pred = float(pred[best_idx])

    if spec.M == 2:
        small_domain_idx = spec.small_domains[0] if spec.small_domains else 1
        other_domain_idx = 1 - small_domain_idx

        def _point_from_small_weights(small_weights: np.ndarray) -> np.ndarray:
            point = np.zeros((spec.N, spec.M), dtype=float)
            point[:, small_domain_idx] = small_weights
            point[:, other_domain_idx] = 1.0 - small_weights
            return point

        def _objective(small_weights: np.ndarray) -> float:
            clipped = np.clip(np.asarray(small_weights, dtype=float), 0.0, 1.0)
            point = _point_from_small_weights(clipped)
            values = np.asarray(predict_fn(point[None, :, :]), dtype=float)
            if values.shape != (1,) or not np.isfinite(values[0]):
                return float("inf")
            return float(values[0])

        starts = [best_point[:, small_domain_idx]]
        starts.extend(rng.uniform(0.0, 1.0, size=(max(OPT_RESTARTS - 1, 0), spec.N)))
        for x0 in starts:
            try:
                result = minimize(
                    _objective,
                    x0=np.asarray(x0, dtype=float),
                    method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * spec.N,
                    options={"maxiter": OPT_MAXITER},
                )
            except Exception:
                continue
            candidate_pred = _objective(result.x)
            if np.isfinite(candidate_pred) and candidate_pred < best_pred:
                best_pred = candidate_pred
                best_point = _point_from_small_weights(result.x)

    replay = replay_proposals_to_observed(best_point[None, :, :], spec.weights)
    nearest_idx = int(replay.selected_indices[0]) if replay.selected_indices else None
    return PredictedOptimum(
        phase_weights=_phase_weights_from_point(best_point, spec),
        predicted_objective=best_pred,
        nearest_observed_idx=nearest_idx,
        nearest_observed_distance=replay.mean_distance if nearest_idx is not None else None,
    )


def _failure_row(
    record: SelectionRecord,
    model_name: str,
    fit_replicates: int,
    runtime_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    score_row = {
        "dataset": record.dataset,
        "mode": record.mode,
        "policy": record.policy,
        "subset_size": record.subset_size,
        "selector_seed": record.selector_seed,
        "evaluation_model": model_name,
        "success": 0.0,
        "success_rate": 0.0,
        "n_fit_replicates": fit_replicates,
        "regret@1": float("inf"),
        "regret@5": float("inf"),
        "chosen_true_rank": float("inf"),
        "Spearman": float("nan"),
        "R²": float("nan"),
        "Huber": float("inf"),
        "RMSE": float("inf"),
        "runtime_s": runtime_s,
        "representative_fit_seed": -1,
        "selected_indices": json.dumps(list(record.selected_indices)),
    }
    optimum_row = {
        "dataset": record.dataset,
        "mode": record.mode,
        "policy": record.policy,
        "subset_size": record.subset_size,
        "selector_seed": record.selector_seed,
        "evaluation_model": model_name,
        "representative_fit_seed": -1,
        "predicted_objective": float("inf"),
        "nearest_observed_idx": np.nan,
        "nearest_observed_distance": np.nan,
        "selected_indices": json.dumps(list(record.selected_indices)),
    }
    return score_row, optimum_row


def _median_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _evaluate_selection_record(
    payload: tuple[DatasetSpec, SelectionRecord, str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, record, model_name, huber_delta = payload
    start = time.perf_counter()
    fit_seed_values = DSRE_FIT_SEEDS if model_name == "DS-RE-CEQ" else (0,)
    train_spec = spec.subset(np.array(record.selected_indices, dtype=int))

    per_fit_rows: list[dict[str, Any]] = []
    per_fit_optima: list[tuple[int, PredictedOptimum]] = []
    for fit_seed in fit_seed_values:
        try:
            predict_fn, _ = _fit_predict_function(model_name, train_spec, fit_seed)
            preds = np.asarray(predict_fn(spec.weights), dtype=float)
            if preds.shape != (spec.R,) or not np.isfinite(preds).all():
                raise ValueError(f"Invalid predictions: {preds.shape}")
            residuals = spec.y - preds
            chosen_idx = int(np.argmin(preds))
            best_idx = int(np.argmin(spec.y))
            top5 = np.argsort(preds)[: min(5, len(preds))]
            true_ranks = np.argsort(np.argsort(spec.y)) + 1
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
            optimum = _optimize_predicted_optimum(
                predict_fn,
                spec,
                search_seed=_stable_seed(
                    record.selector_seed,
                    record.subset_size,
                    fit_seed,
                    len(record.selected_indices),
                ),
            )
            per_fit_rows.append(
                {
                    "fit_seed": fit_seed,
                    "success": 1.0,
                    "regret@1": float(spec.y[chosen_idx] - spec.y[best_idx]),
                    "regret@5": float(np.min(spec.y[top5]) - spec.y[best_idx]),
                    "chosen_true_rank": float(true_ranks[chosen_idx]),
                    "Spearman": float(spearmanr(spec.y, preds)[0]),
                    "R²": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    "Huber": _safe_huber(spec.y, preds, huber_delta),
                    "RMSE": float(np.sqrt(np.mean(residuals**2))),
                }
            )
            per_fit_optima.append((fit_seed, optimum))
        except Exception:
            per_fit_rows.append(
                {
                    "fit_seed": fit_seed,
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
    runtime_s = time.perf_counter() - start
    if not success_rows:
        return _failure_row(record, model_name, len(fit_seed_values), runtime_s)

    median_regret = _median_numeric(success_rows, "regret@1")
    representative = min(
        success_rows,
        key=lambda row: (abs(float(row["regret@1"]) - median_regret), int(row["fit_seed"])),
    )
    representative_fit_seed = int(representative["fit_seed"])
    representative_optimum = next(optimum for fit_seed, optimum in per_fit_optima if fit_seed == representative_fit_seed)

    score_row = {
        "dataset": record.dataset,
        "mode": record.mode,
        "policy": record.policy,
        "subset_size": record.subset_size,
        "selector_seed": record.selector_seed,
        "evaluation_model": model_name,
        "success": 1.0,
        "success_rate": float(np.mean([row["success"] for row in per_fit_rows])),
        "n_fit_replicates": len(fit_seed_values),
        "regret@1": _median_numeric(success_rows, "regret@1"),
        "regret@5": _median_numeric(success_rows, "regret@5"),
        "chosen_true_rank": _median_numeric(success_rows, "chosen_true_rank"),
        "Spearman": _median_numeric(success_rows, "Spearman"),
        "R²": _median_numeric(success_rows, "R²"),
        "Huber": _median_numeric(success_rows, "Huber"),
        "RMSE": _median_numeric(success_rows, "RMSE"),
        "runtime_s": runtime_s,
        "representative_fit_seed": representative_fit_seed,
        "selected_indices": json.dumps(list(record.selected_indices)),
    }
    optimum_row = {
        "dataset": record.dataset,
        "mode": record.mode,
        "policy": record.policy,
        "subset_size": record.subset_size,
        "selector_seed": record.selector_seed,
        "evaluation_model": model_name,
        "representative_fit_seed": representative_fit_seed,
        "predicted_objective": representative_optimum.predicted_objective,
        "nearest_observed_idx": representative_optimum.nearest_observed_idx,
        "nearest_observed_distance": representative_optimum.nearest_observed_distance,
        "selected_indices": json.dumps(list(record.selected_indices)),
        **_flatten_phase_weights(representative_optimum.phase_weights),
    }
    return score_row, optimum_row


def _evaluate_records(
    spec: DatasetSpec,
    records: list[SelectionRecord],
    *,
    model_names: Sequence[str],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    huber_delta = float(_huber_delta(spec.y))
    payloads = [(spec, record, model_name, huber_delta) for record in records for model_name in model_names]
    if workers <= 1:
        results = [_evaluate_selection_record(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_evaluate_selection_record, payloads))
    score_rows = [score for score, _ in results]
    optimum_rows = [optimum for _, optimum in results]
    return pd.DataFrame(score_rows), pd.DataFrame(optimum_rows)


def _curve_summary_rows(model_scores: pd.DataFrame) -> pd.DataFrame:
    if model_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "mode", "policy", "evaluation_model", "subset_size"]
    metrics = ("R²", "regret@1", "Huber", "RMSE", "regret@5", "chosen_true_rank", "Spearman")
    for keys, frame in model_scores.groupby(group_cols, sort=True):
        dataset, mode, policy, evaluation_model, subset_size = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "policy": policy,
            "evaluation_model": evaluation_model,
            "subset_size": int(subset_size),
            "n_records": len(frame),
            "success_rate": float(frame["success_rate"].mean()),
        }
        for metric in metrics:
            finite = frame.loc[np.isfinite(frame[metric]), metric].to_numpy(dtype=float)
            if finite.size == 0:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
            else:
                row[f"{metric}_median"] = float(np.median(finite))
                row[f"{metric}_q25"] = float(np.percentile(finite, 25))
                row[f"{metric}_q75"] = float(np.percentile(finite, 75))
        rows.append(row)
    return pd.DataFrame(rows)


def _selector_summary_rows(model_scores: pd.DataFrame) -> pd.DataFrame:
    if model_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (dataset, mode, policy), frame in model_scores.groupby(["dataset", "mode", "policy"], sort=True):
        recommended_k = RECOMMENDED_K.get(dataset)
        frame = frame[frame["subset_size"] == recommended_k]
        if frame.empty:
            continue
        model_medians = frame.groupby("evaluation_model")[["regret@1", "chosen_true_rank"]].median().reset_index()
        dsre = model_medians[model_medians["evaluation_model"] == "DS-RE-CEQ"]
        dsre_regret = float(dsre["regret@1"].iloc[0]) if not dsre.empty else np.nan
        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "policy": policy,
                "subset_size": recommended_k,
                "dsre_median_regret@1": dsre_regret,
                "committee_mean_regret@1": float(model_medians["regret@1"].mean()),
                "committee_mean_true_rank": float(model_medians["chosen_true_rank"].mean()),
                "n_records": int(frame["selector_seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _apply_subset_size_axis(ax: Any, subset_sizes: Sequence[int], *, scale: str) -> None:
    ticks = sorted({int(size) for size in subset_sizes})
    if scale == "linear":
        ax.set_xscale("linear")
    elif scale == "logx":
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        raise ValueError(f"Unsupported subset-size axis scale: {scale}")
    ax.set_xticks(ticks)
    ax.set_xlim(min(ticks), max(ticks))


def _policy_handles(policies: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=POLICY_COLORS.get(policy, "#111827"),
            linewidth=2.0,
            marker="o",
            linestyle="-",
            label=POLICY_LABELS.get(policy, policy),
        )
        for policy in policies
    ]


def _dataset_handles(datasets: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=2.0,
            linestyle=DATASET_LINESTYLES[dataset],
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in datasets
    ]


def _metric_axes(metric: str, *, figsize: tuple[float, float]) -> tuple[plt.Figure, list[plt.Axes]]:
    if metric != "R²":
        fig, ax = plt.subplots(figsize=figsize)
        return fig, [ax]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figsize[0], figsize[1] + 1.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2], "hspace": 0.18},
    )
    return fig, list(axes)


def _style_metric_axes(
    axes: list[plt.Axes],
    *,
    metric: str,
    subset_sizes: Sequence[int | float],
    x_scale: str,
) -> None:
    main_ax = axes[0]
    _apply_subset_size_axis(main_ax, subset_sizes, scale=x_scale)
    main_ax.set_ylabel(metric)
    main_ax.grid(True, alpha=0.3)

    if len(axes) == 1:
        main_ax.set_xlabel("Subset size")
        return

    zoom_ax = axes[1]
    _apply_subset_size_axis(zoom_ax, subset_sizes, scale=x_scale)
    zoom_ax.set_xlabel("Subset size")
    zoom_ax.set_ylabel(f"{metric} zoom")
    zoom_ax.set_ylim(0.0, 1.0)
    zoom_ax.grid(True, alpha=0.3)
    zoom_ax.set_title("Zoomed to 0-1", pad=12)
    main_ax.tick_params(labelbottom=False)


def _plot_dataset_metric(
    curves: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    plots_dir: Path,
    x_scale: str,
) -> None:
    frame = curves[
        (curves["dataset"] == dataset)
        & (curves["mode"] == "retrospective")
        & (curves["evaluation_model"] == PLOT_MODEL)
        & (curves["policy"].isin(CURVE_POLICIES))
    ]
    if frame.empty:
        return

    fig, axes = _metric_axes(metric, figsize=(7.5, 4.8))
    main_ax = axes[0]
    for policy in CURVE_POLICIES:
        policy_frame = frame[frame["policy"] == policy].sort_values("subset_size")
        if policy_frame.empty:
            continue
        x = policy_frame["subset_size"].to_numpy(dtype=float)
        y = policy_frame[f"{metric}_median"].to_numpy(dtype=float)
        color = POLICY_COLORS.get(policy, "#111827")
        label = POLICY_LABELS.get(policy, policy)
        if policy == "random_observed":
            q25 = policy_frame[f"{metric}_q25"].to_numpy(dtype=float)
            q75 = policy_frame[f"{metric}_q75"].to_numpy(dtype=float)
            for ax in axes:
                ax.fill_between(x, q25, q75, color=color, alpha=0.2)
                ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
        else:
            for ax in axes:
                ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
    _style_metric_axes(axes, metric=metric, subset_sizes=frame["subset_size"].unique(), x_scale=x_scale)
    title_suffix = "log-x" if x_scale == "logx" else "linear-x"
    main_ax.set_title(f"{dataset} — {metric} vs subset size ({PLOT_MODEL}, {title_suffix})")
    main_ax.legend(frameon=False)
    if len(axes) == 1:
        fig.tight_layout()
    else:
        fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.22)
    suffix = "_logx" if x_scale == "logx" else ""
    out_path = plots_dir / f"{dataset}_{metric.lower().replace('@', 'at').replace('²', '2')}_comparison{suffix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cross_dataset_overlay(curves: pd.DataFrame, *, metric: str, plots_dir: Path, x_scale: str) -> None:
    frame = curves[
        (curves["mode"] == "retrospective")
        & (curves["evaluation_model"] == PLOT_MODEL)
        & (curves["policy"].isin(CURVE_POLICIES[1:]))
    ]
    if frame.empty:
        return

    fig, axes = _metric_axes(metric, figsize=(7.8, 4.8))
    main_ax = axes[0]
    datasets_present: list[str] = []
    for policy in CURVE_POLICIES[1:]:
        for dataset in (TWO_PHASE_STARCODER.name, THREE_PHASE_STARCODER.name):
            policy_frame = frame[(frame["policy"] == policy) & (frame["dataset"] == dataset)].sort_values("subset_size")
            if policy_frame.empty:
                continue
            if dataset not in datasets_present:
                datasets_present.append(dataset)
            x = policy_frame["subset_size"].to_numpy(dtype=float)
            y = policy_frame[f"{metric}_median"].to_numpy(dtype=float)
            for ax in axes:
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    linestyle=DATASET_LINESTYLES[dataset],
                    color=POLICY_COLORS.get(policy, "#111827"),
                )
    _style_metric_axes(axes, metric=metric, subset_sizes=frame["subset_size"].unique(), x_scale=x_scale)
    title_suffix = "log-x" if x_scale == "logx" else "linear-x"
    main_ax.set_title(f"Static selectors across 2-phase vs 3-phase ({PLOT_MODEL}, {title_suffix})")
    selector_legend = main_ax.legend(
        handles=_policy_handles(CURVE_POLICIES[1:]),
        title="Selector",
        frameon=False,
        loc="upper left",
    )
    main_ax.add_artist(selector_legend)
    main_ax.legend(
        handles=_dataset_handles(datasets_present),
        title="Dataset / line style",
        frameon=False,
        loc="upper right",
    )
    if len(axes) == 1:
        fig.tight_layout()
    else:
        fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.22)
    suffix = "_logx" if x_scale == "logx" else ""
    out_path = plots_dir / f"cross_dataset_{metric.lower().replace('@', 'at').replace('²', '2')}_overlay{suffix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dataset_panel(curves: pd.DataFrame, *, dataset: str, plots_dir: Path, x_scale: str) -> None:
    frame = curves[
        (curves["dataset"] == dataset)
        & (curves["mode"] == "retrospective")
        & (curves["evaluation_model"] == PLOT_MODEL)
        & (curves["policy"].isin(CURVE_POLICIES))
    ]
    if frame.empty:
        return

    fig = plt.figure(figsize=(11.0, 9.2))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.95], hspace=0.35, wspace=0.28)
    axes_by_metric = {
        "R²": [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])],
        "regret@1": [fig.add_subplot(grid[0, 1])],
        "Huber": [fig.add_subplot(grid[1, 1])],
        "RMSE": [fig.add_subplot(grid[2, :])],
    }

    for metric, metric_axes in axes_by_metric.items():
        metric_frame = frame.copy()
        for policy in CURVE_POLICIES:
            policy_frame = metric_frame[metric_frame["policy"] == policy].sort_values("subset_size")
            if policy_frame.empty:
                continue
            x = policy_frame["subset_size"].to_numpy(dtype=float)
            y = policy_frame[f"{metric}_median"].to_numpy(dtype=float)
            color = POLICY_COLORS.get(policy, "#111827")
            if policy == "random_observed":
                q25 = policy_frame[f"{metric}_q25"].to_numpy(dtype=float)
                q75 = policy_frame[f"{metric}_q75"].to_numpy(dtype=float)
                for ax in metric_axes:
                    ax.fill_between(x, q25, q75, color=color, alpha=0.2)
                    ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=POLICY_LABELS.get(policy, policy))
            else:
                for ax in metric_axes:
                    ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=POLICY_LABELS.get(policy, policy))

        _style_metric_axes(
            metric_axes,
            metric=metric,
            subset_sizes=metric_frame["subset_size"].unique(),
            x_scale=x_scale,
        )
        if len(metric_axes) == 1:
            metric_axes[0].set_title(metric)
        else:
            metric_axes[0].set_title("R²")

    handles, labels = axes_by_metric["R²"][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    title_suffix = "log-x" if x_scale == "logx" else "linear-x"
    fig.suptitle(f"{dataset} — random vs static selectors ({PLOT_MODEL}, {title_suffix})", y=0.98)
    fig.subplots_adjust(top=0.93, bottom=0.09)
    suffix = "_logx" if x_scale == "logx" else ""
    out_path = plots_dir / f"{dataset}_panel{suffix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
                    for domain_name in (
                        "nemotron_full",
                        "starcoder",
                    )
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


def _append_logbook(
    output_dir: Path,
    args: argparse.Namespace,
    selection_records: pd.DataFrame,
    model_scores: pd.DataFrame,
) -> None:
    logbook = Path(".agents/logbooks/starcoder_subset_sampling.md")
    logbook.parent.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M")
    summary_lines = [
        f"### {timestamp} - generic selector benchmark implementation smoke",
        "- Hypothesis: generic feature-space selectors can be benchmarked end-to-end "
        "with saved predicted optima and DS-RE-CEQ sample-efficiency plots.",
        f"- Command: {' '.join(sys.argv)}",
        f"- Output dir: {output_dir}",
        f"- Selection rows: {len(selection_records)}",
        f"- Model score rows: {len(model_scores)}",
        f"- Result: emitted benchmark artifacts and plots under {output_dir}.",
        "- Interpretation: harness is runnable; full results depend on the actual benchmark budget used.",
        "- Next action: run the full benchmark configuration and review "
        "selector_summary.csv / predicted_optima exports.",
        "",
    ]
    with logbook.open("a") as handle:
        handle.write("\n".join(summary_lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generic feature-space selectors on StarCoder")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--datasets", type=str, default=f"{TWO_PHASE_STARCODER.name},{THREE_PHASE_STARCODER.name}")
    parser.add_argument("--subset-sizes", type=str, default="")
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--random-bootstrap-seeds", type=int, default=RANDOM_BOOTSTRAP_SEEDS)
    parser.add_argument("--retrospective-dopt-seeds", type=int, default=RETRO_DOPT_SEEDS)
    parser.add_argument("--prospective-seeds", type=int, default=PROSPECTIVE_SEEDS)
    parser.add_argument("--prospective-pool-size", type=int, default=PROSPECTIVE_POOL_SIZE)
    parser.add_argument("--dsre-fit-seeds", type=int, default=len(DSRE_FIT_SEEDS))
    parser.add_argument("--dsre-restarts", type=int, default=DSRE_RESTARTS)
    parser.add_argument("--dsre-maxiter", type=int, default=DSRE_MAXITER)
    parser.add_argument("--opt-search-points", type=int, default=OPT_SEARCH_POINTS)
    parser.add_argument("--opt-restarts", type=int, default=OPT_RESTARTS)
    parser.add_argument("--opt-maxiter", type=int, default=OPT_MAXITER)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--skip-two-phase-oracle-ensure", action="store_true")
    parser.add_argument("--skip-logbook", action="store_true")
    return parser.parse_args()


def main() -> None:
    global DSRE_FIT_SEEDS
    global DSRE_RESTARTS
    global DSRE_MAXITER
    global OPT_SEARCH_POINTS
    global OPT_RESTARTS
    global OPT_MAXITER

    args = _parse_args()
    DSRE_FIT_SEEDS = tuple(range(max(1, args.dsre_fit_seeds)))
    DSRE_RESTARTS = max(1, args.dsre_restarts)
    DSRE_MAXITER = max(1, args.dsre_maxiter)
    OPT_SEARCH_POINTS = max(64, args.opt_search_points)
    OPT_RESTARTS = max(1, args.opt_restarts)
    OPT_MAXITER = max(1, args.opt_maxiter)

    output_dir = args.output_dir
    plots_dir = output_dir / PLOTS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    subset_sizes_override = tuple(int(part) for part in args.subset_sizes.split(",") if part.strip()) or None

    oracle_reference = {
        "three_phase_oracle_bank": str(THREE_PHASE_ORACLE_BANK_DIR),
        "three_phase_oracle_bank_exists": (THREE_PHASE_ORACLE_BANK_DIR / "top_search_candidates.csv").exists(),
        "two_phase_oracle_bank": None,
        "two_phase_oracle_bank_exists": False,
    }
    if args.row_limit is None:
        ensured = ensure_two_phase_oracle_bank(workers=ORACLE_BANK_WORKERS, skip=args.skip_two_phase_oracle_ensure)
        if ensured is not None:
            oracle_reference["two_phase_oracle_bank"] = str(ensured)
            oracle_reference["two_phase_oracle_bank_exists"] = (ensured / "top_search_candidates.csv").exists()
    (output_dir / "oracle_references.json").write_text(json.dumps(oracle_reference, indent=2, sort_keys=True))

    selection_frames: list[pd.DataFrame] = []
    model_score_frames: list[pd.DataFrame] = []
    optimum_frames: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        spec, _frame, experiment = _load_dataset_bundle(dataset_name, row_limit=args.row_limit)
        subset_sizes = _subset_sizes_for_spec(spec, subset_sizes_override)
        print(f"dataset={dataset_name} rows={spec.R} subset_sizes={subset_sizes}")

        retrospective_records = _generate_retrospective_records(
            spec,
            dataset=dataset_name,
            subset_sizes=subset_sizes,
            random_bootstrap_seeds=args.random_bootstrap_seeds,
            retrospective_dopt_seeds=args.retrospective_dopt_seeds,
            workers=args.workers,
        )
        prospective_records = _generate_prospective_records(
            spec,
            experiment,
            dataset=dataset_name,
            prospective_seeds=args.prospective_seeds,
            pool_size=args.prospective_pool_size,
            workers=args.workers,
        )
        records = retrospective_records + prospective_records
        selection_frame = pd.DataFrame([_record_to_dict(record) for record in records])
        selection_frames.append(selection_frame)

        model_scores, predicted_optima = _evaluate_records(
            spec,
            records,
            model_names=COMMITTEE_MODELS,
            workers=args.workers,
        )
        model_score_frames.append(model_scores)
        optimum_frames.append(predicted_optima)

    selection_records = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    model_scores = pd.concat(model_score_frames, ignore_index=True) if model_score_frames else pd.DataFrame()
    predicted_optima = pd.concat(optimum_frames, ignore_index=True) if optimum_frames else pd.DataFrame()
    curve_points = _curve_summary_rows(model_scores)
    selector_summary = _selector_summary_rows(model_scores)

    selection_records.to_csv(output_dir / "selection_records.csv", index=False)
    model_scores.to_csv(output_dir / "model_scores.csv", index=False)
    curve_points.to_csv(output_dir / "curve_points.csv", index=False)
    selector_summary.to_csv(output_dir / "selector_summary.csv", index=False)
    predicted_optima.to_csv(output_dir / "predicted_optima.csv", index=False)
    _write_predicted_optima_jsonl(predicted_optima, output_dir / "predicted_optima.jsonl")

    for dataset_name in dataset_names:
        for metric in ("R²", "regret@1", "Huber", "RMSE"):
            _plot_dataset_metric(
                curve_points,
                dataset=dataset_name,
                metric=metric,
                plots_dir=plots_dir,
                x_scale="linear",
            )
            _plot_dataset_metric(
                curve_points,
                dataset=dataset_name,
                metric=metric,
                plots_dir=plots_dir,
                x_scale="logx",
            )
        _plot_dataset_panel(curve_points, dataset=dataset_name, plots_dir=plots_dir, x_scale="linear")
        _plot_dataset_panel(curve_points, dataset=dataset_name, plots_dir=plots_dir, x_scale="logx")
    for metric in ("R²", "regret@1", "Huber", "RMSE"):
        _plot_cross_dataset_overlay(curve_points, metric=metric, plots_dir=plots_dir, x_scale="linear")
        _plot_cross_dataset_overlay(curve_points, metric=metric, plots_dir=plots_dir, x_scale="logx")

    if not args.skip_logbook:
        _append_logbook(output_dir, args, selection_records, model_scores)

    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
