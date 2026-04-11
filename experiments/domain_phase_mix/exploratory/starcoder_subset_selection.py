# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Oracle subset search and static batch design study for StarCoder datasets."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.domain_phase_mix.static_batch_selection import (
    ReplayMatch,
    compute_subset_diagnostics,
    fit_anchor_artifacts,
    greedy_k_center_indices,
    prospective_d_optimal_selection,
    replay_proposals_to_observed,
    retrospective_d_optimal_selection,
    sampler_replay_selection,
    weight_configs_to_tensor,
)
from experiments.domain_phase_mix.starcoder_metadata import (
    DEFAULT_STARCODER_OBJECTIVE,
    THREE_PHASE_STARCODER,
    TWO_PHASE_STARCODER,
    load_starcoder_dataset,
)
from experiments.domain_phase_mix.three_phase_starcoder_experiment import create_three_phase_experiment
from experiments.domain_phase_mix.two_phase_starcoder_experiment import create_two_phase_experiment
from experiments.domain_phase_mix.exploratory.dsre_ceq_tools import fit_dsre_ceq_artifacts
from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec

OUTPUT_DIR = Path(__file__).resolve().parent / "starcoder_subset_selection_outputs"
SEARCH_RESTARTS = 4
SEARCH_MAXITER = 300
FINAL_RESTARTS = 16
FINAL_MAXITER = 800
FINAL_SEEDS = 5
SEARCH_RANDOM_SUBSETS = 128
SEARCH_KCENTER_SUBSETS = 32
SEARCH_DOPT_SUBSETS = 32
SEARCH_TOP_SUBSETS = 10
SEARCH_TOP_FINAL = 20
HILL_CLIMB_MAX_ACCEPTED = 25
HILL_CLIMB_SWAP_CANDIDATES = 16
DEPLOYABLE_SEEDS = 32
SAVE_TOP_SEARCH_N = 0
EVALUATE_DEPLOYABLE = True
PARALLEL_BACKEND: Literal["thread", "process"] = "process"
RECOMMENDED_SUCCESS = 0.95
RECOMMENDED_TOL = 0.05
MATERIAL_IMPROVEMENT = 0.9

TWO_PHASE_K = (8, 10, 12, 16, 20)
THREE_PHASE_K = (10, 16, 24, 32)


@dataclass(frozen=True)
class SubsetRecord:
    """One subset candidate."""

    dataset_name: str
    k: int
    policy: str
    seed: int
    indices: tuple[int, ...]


def _normalize_indices(indices) -> tuple[int, ...]:
    return tuple(int(idx) for idx in indices)


def _stable_subset_id(indices: tuple[int, ...]) -> str:
    return "-".join(str(idx) for idx in _normalize_indices(indices))


def _score_subset_once(
    spec: DatasetSpec,
    *,
    indices: tuple[int, ...],
    fit_seed: int,
    n_restarts: int,
    maxiter: int,
) -> dict[str, float]:
    start = time.perf_counter()
    try:
        artifacts = fit_dsre_ceq_artifacts(
            spec.subset(np.array(indices, dtype=int)),
            seed=fit_seed,
            n_restarts=n_restarts,
            maxiter=maxiter,
        )
        preds = artifacts.predict_fn(spec.weights)
        chosen_idx = int(np.argmin(preds))
        best_idx = int(np.argmin(spec.y))
        top5 = np.argsort(preds)[: min(5, len(preds))]
        residuals = spec.y - preds
        rmse = float(np.sqrt(np.mean(residuals**2)))
        sp_corr, _ = spearmanr(spec.y, preds)
        true_ranks = np.argsort(np.argsort(spec.y))
        runtime = time.perf_counter() - start
        return {
            "success": 1.0,
            "regret@1": float(spec.y[chosen_idx] - spec.y[best_idx]),
            "regret@5": float(np.min(spec.y[top5]) - spec.y[best_idx]),
            "chosen_true_rank": float(true_ranks[chosen_idx] + 1),
            "rmse": rmse,
            "spearman": float(sp_corr) if np.isfinite(sp_corr) else float("nan"),
            "runtime_s": runtime,
        }
    except Exception:
        return {
            "success": 0.0,
            "regret@1": float("inf"),
            "regret@5": float("inf"),
            "chosen_true_rank": float("inf"),
            "rmse": float("inf"),
            "spearman": float("nan"),
            "runtime_s": time.perf_counter() - start,
        }


def _score_records_parallel(
    spec: DatasetSpec,
    records: list[SubsetRecord],
    *,
    fit_seed: int,
    n_restarts: int,
    maxiter: int,
    workers: int,
) -> list[dict[str, float | int | str]]:
    if not records:
        return []

    if workers <= 1:
        return [_score_record_job((spec, record, fit_seed, n_restarts, maxiter)) for record in records]

    executor_cls = ProcessPoolExecutor if PARALLEL_BACKEND == "process" else ThreadPoolExecutor
    payloads = [(spec, record, fit_seed, n_restarts, maxiter) for record in records]
    with executor_cls(max_workers=max(1, workers)) as pool:
        return list(pool.map(_score_record_job, payloads))


def _make_random_records(spec: DatasetSpec, dataset_name: str, k: int, seed: int) -> list[SubsetRecord]:
    rng = np.random.default_rng(seed)
    records: list[SubsetRecord] = []
    for offset in range(SEARCH_RANDOM_SUBSETS):
        idx = tuple(sorted(int(i) for i in rng.choice(spec.R, size=k, replace=False)))
        records.append(SubsetRecord(dataset_name, k, "random_observed", seed + offset, idx))
    return records


def _make_kcenter_records(spec: DatasetSpec, dataset_name: str, k: int, seed: int) -> list[SubsetRecord]:
    records: list[SubsetRecord] = []
    for offset in range(SEARCH_KCENTER_SUBSETS):
        idx = tuple(sorted(int(i) for i in greedy_k_center_indices(spec.weights, k, seed=seed + offset)))
        records.append(SubsetRecord(dataset_name, k, "kcenter_observed", seed + offset, idx))
    return records


def _make_dopt_records(
    spec: DatasetSpec,
    dataset_name: str,
    k: int,
    seed: int,
    *,
    workers: int,
) -> list[SubsetRecord]:
    if SEARCH_DOPT_SUBSETS <= 0:
        return []

    payloads = [(spec, dataset_name, k, seed + offset) for offset in range(SEARCH_DOPT_SUBSETS)]
    if workers <= 1:
        return [_make_dopt_record_job(payload) for payload in payloads]

    executor_cls = ProcessPoolExecutor if PARALLEL_BACKEND == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=max(1, min(workers, SEARCH_DOPT_SUBSETS))) as pool:
        return list(pool.map(_make_dopt_record_job, payloads))


def _make_dopt_record_job(payload: tuple[DatasetSpec, str, int, int]) -> SubsetRecord:
    spec, dataset_name, k, seed = payload
    selection = retrospective_d_optimal_selection(spec, k=k, seed=seed)
    idx = tuple(sorted(int(i) for i in selection.selected_indices))
    return SubsetRecord(dataset_name, k, "doptimal_observed", seed, idx)


def _rank_candidate_pool(spec: DatasetSpec, seed: int) -> np.ndarray:
    anchor = fit_anchor_artifacts(spec, seed=seed)
    jacobian = anchor.jacobian(spec.weights)
    return np.argsort(-np.sum(jacobian**2, axis=1))


def _hill_climb(
    spec: DatasetSpec,
    *,
    dataset_name: str,
    k: int,
    initial_indices: tuple[int, ...],
    seed: int,
    workers: int,
) -> SubsetRecord:
    rng = np.random.default_rng(seed)
    ranked_pool = _rank_candidate_pool(spec, seed)
    current = initial_indices
    current_score = _score_subset_once(
        spec,
        indices=current,
        fit_seed=seed,
        n_restarts=SEARCH_RESTARTS,
        maxiter=SEARCH_MAXITER,
    )["regret@1"]
    accepted = 0

    while accepted < HILL_CLIMB_MAX_ACCEPTED:
        current_set = set(current)
        preferred = [idx for idx in ranked_pool if idx not in current_set][: HILL_CLIMB_SWAP_CANDIDATES // 2]
        remaining = [idx for idx in range(spec.R) if idx not in current_set]
        if not remaining:
            break
        random_candidates = list(
            int(idx)
            for idx in rng.choice(
                remaining,
                size=min(HILL_CLIMB_SWAP_CANDIDATES - len(preferred), len(remaining)),
                replace=False,
            )
        )
        replacements = list(dict.fromkeys([*preferred, *random_candidates]))
        positions = list(
            int(pos)
            for pos in rng.choice(
                len(current),
                size=min(4, len(current)),
                replace=False,
            )
        )

        proposals: list[SubsetRecord] = []
        for position in positions:
            for candidate_idx in replacements:
                proposal = list(current)
                proposal[position] = candidate_idx
                if len(set(proposal)) != k:
                    continue
                idx = tuple(sorted(int(i) for i in proposal))
                if idx == current:
                    continue
                proposals.append(SubsetRecord(dataset_name, k, "oracle_hill_climb", seed, idx))

        if not proposals:
            break

        scored = _score_records_parallel(
            spec,
            proposals,
            fit_seed=seed,
            n_restarts=SEARCH_RESTARTS,
            maxiter=SEARCH_MAXITER,
            workers=workers,
        )
        best = min(scored, key=lambda row: row["regret@1"])
        if float(best["regret@1"]) < float(current_score):
            current = _normalize_indices(json.loads(str(best["selected_indices"])))
            current_score = float(best["regret@1"])
            accepted += 1
        else:
            break

    return SubsetRecord(dataset_name, k, "oracle_hill_climb", seed, _normalize_indices(current))


def _final_rescore(
    spec: DatasetSpec,
    record: SubsetRecord,
) -> dict[str, float | int | str]:
    scores = [
        _score_subset_once(
            spec,
            indices=record.indices,
            fit_seed=record.seed + offset,
            n_restarts=FINAL_RESTARTS,
            maxiter=FINAL_MAXITER,
        )
        for offset in range(FINAL_SEEDS)
    ]
    frame = pd.DataFrame(scores)
    diagnostics = compute_subset_diagnostics(
        spec.weights[list(record.indices)],
        full_pool_weights=spec.weights,
        epoch_multipliers=np.asarray(spec.epoch_multipliers, dtype=float),
        small_domain_idx=spec.small_domains[0] if spec.small_domains else None,
    )
    return {
        "dataset": record.dataset_name,
        "k": record.k,
        "policy": record.policy,
        "seed": record.seed,
        "subset_id": _stable_subset_id(record.indices),
        "selected_indices": json.dumps(list(_normalize_indices(record.indices))),
        "success_rate": float(frame["success"].mean()),
        "median_regret@1": float(frame["regret@1"].median()),
        "median_regret@5": float(frame["regret@5"].median()),
        "median_true_rank": float(frame["chosen_true_rank"].median()),
        "median_rmse": float(frame["rmse"].median()),
        "median_spearman": float(frame["spearman"].median(skipna=True)),
        "median_runtime_s": float(frame["runtime_s"].median()),
        **diagnostics,
    }


def _score_record_job(
    payload: tuple[DatasetSpec, SubsetRecord, int, int, int],
) -> dict[str, float | int | str]:
    spec, record, fit_seed, n_restarts, maxiter = payload
    metrics = _score_subset_once(
        spec,
        indices=record.indices,
        fit_seed=fit_seed,
        n_restarts=n_restarts,
        maxiter=maxiter,
    )
    return {
        "dataset": record.dataset_name,
        "k": record.k,
        "policy": record.policy,
        "seed": record.seed,
        "subset_id": _stable_subset_id(record.indices),
        "selected_indices": json.dumps(list(_normalize_indices(record.indices))),
        **metrics,
    }


def _final_rescore_job(payload: tuple[DatasetSpec, SubsetRecord]) -> dict[str, float | int | str]:
    spec, record = payload
    return _final_rescore(spec, record)


def _final_rescore_parallel(
    spec: DatasetSpec,
    records: list[SubsetRecord],
    *,
    workers: int,
) -> list[dict[str, float | int | str]]:
    if not records:
        return []
    if workers <= 1:
        return [_final_rescore(spec, record) for record in records]

    executor_cls = ProcessPoolExecutor if PARALLEL_BACKEND == "process" else ThreadPoolExecutor
    payloads = [(spec, record) for record in records]
    with executor_cls(max_workers=max(1, workers)) as pool:
        return list(pool.map(_final_rescore_job, payloads))


def _evaluate_deployable_policy(
    spec: DatasetSpec,
    *,
    dataset_name: str,
    k: int,
    workers: int,
) -> list[dict[str, float | int | str]]:
    experiment = (
        create_two_phase_experiment(name=dataset_name)
        if spec.N == 2
        else create_three_phase_experiment(name=dataset_name)
    )
    rows: list[dict[str, float | int | str]] = []
    for seed in range(DEPLOYABLE_SEEDS):
        dopt_configs, selection = prospective_d_optimal_selection(
            spec,
            experiment,
            n_select=k,
            seed=seed,
            pool_size=max(2048, 64 * k),
        )
        dopt_match = _score_replay_match(
            spec,
            dataset_name=dataset_name,
            k=k,
            policy="deployable_doptimal_replay",
            seed=seed,
            replay=replay_from_configs(spec, dopt_configs),
            workers=workers,
        )
        dopt_match["selection_logdet"] = selection.info_logdet
        rows.append(dopt_match)

        sampler_match = _score_replay_match(
            spec,
            dataset_name=dataset_name,
            k=k,
            policy="sampler_replay",
            seed=seed,
            replay=sampler_replay_selection(spec, experiment, n_select=k, seed=seed),
            workers=workers,
        )
        rows.append(sampler_match)
    return rows


def replay_from_configs(spec: DatasetSpec, configs) -> ReplayMatch:
    proposal_weights = weight_configs_to_tensor(
        configs,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    return replay_proposals_to_observed(proposal_weights, spec.weights)


def _score_replay_match(
    spec: DatasetSpec,
    *,
    dataset_name: str,
    k: int,
    policy: str,
    seed: int,
    replay: ReplayMatch,
    workers: int,
) -> dict[str, float | int | str]:
    record = SubsetRecord(dataset_name, k, policy, seed, _normalize_indices(replay.selected_indices))
    summary = _final_rescore(spec, record)
    summary["replay_mean_distance"] = replay.mean_distance
    summary["replay_max_distance"] = replay.max_distance
    return summary


def _recommended_k(frame: pd.DataFrame) -> dict[str, int]:
    recommendations: dict[str, int] = {}
    for dataset_name, dataset_df in frame.groupby("dataset"):
        best_regret = float(dataset_df["median_regret@1"].min())
        candidates = dataset_df[
            (dataset_df["success_rate"] >= RECOMMENDED_SUCCESS)
            & (dataset_df["median_regret@1"] <= best_regret * (1.0 + RECOMMENDED_TOL))
        ].sort_values(["k", "median_regret@1"])
        if candidates.empty:
            best_row = dataset_df.sort_values(["median_regret@1", "k"]).iloc[0]
            recommendations[dataset_name] = int(best_row["k"])
        else:
            recommendations[dataset_name] = int(candidates.iloc[0]["k"])
    return recommendations


def _search_dataset(
    spec: DatasetSpec,
    *,
    dataset_name: str,
    k_values: tuple[int, ...],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    search_rows: list[dict[str, float | int | str]] = []
    top_search_rows: list[dict[str, float | int | str]] = []
    final_rows: list[dict[str, float | int | str]] = []
    deployable_rows: list[dict[str, float | int | str]] = []

    for k in k_values:
        print(
            f"[study] dataset={dataset_name} k={k} starting "
            f"(random={SEARCH_RANDOM_SUBSETS}, kcenter={SEARCH_KCENTER_SUBSETS}, dopt={SEARCH_DOPT_SUBSETS})",
            flush=True,
        )
        initial_records = [
            *_make_random_records(spec, dataset_name, k, seed=1000 + k),
            *_make_kcenter_records(spec, dataset_name, k, seed=2000 + k),
            *_make_dopt_records(spec, dataset_name, k, seed=3000 + k, workers=workers),
        ]
        scored_initial = _score_records_parallel(
            spec,
            initial_records,
            fit_seed=0,
            n_restarts=SEARCH_RESTARTS,
            maxiter=SEARCH_MAXITER,
            workers=workers,
        )
        search_rows.extend(scored_initial)
        print(f"[study] dataset={dataset_name} k={k} initial_search_done rows={len(scored_initial)}", flush=True)

        ranked = (
            pd.DataFrame(scored_initial)
            .sort_values(["regret@1", "success", "runtime_s"], ascending=[True, False, True])
            .drop_duplicates(subset=["subset_id"])
            .head(SEARCH_TOP_SUBSETS)
        )
        hill_records = [
            _hill_climb(
                spec,
                dataset_name=dataset_name,
                k=k,
                initial_indices=_normalize_indices(json.loads(row["selected_indices"])),
                seed=int(row["seed"]),
                workers=workers,
            )
            for row in ranked.to_dict("records")
        ]
        scored_hill = _score_records_parallel(
            spec,
            hill_records,
            fit_seed=0,
            n_restarts=SEARCH_RESTARTS,
            maxiter=SEARCH_MAXITER,
            workers=workers,
        )
        search_rows.extend(scored_hill)
        print(f"[study] dataset={dataset_name} k={k} hill_climb_done rows={len(scored_hill)}", flush=True)

        combined = pd.concat([pd.DataFrame(scored_initial), pd.DataFrame(scored_hill)], ignore_index=True)
        if SAVE_TOP_SEARCH_N > 0:
            top_search = (
                combined.sort_values(["regret@1", "success", "runtime_s"], ascending=[True, False, True])
                .drop_duplicates(subset=["subset_id"])
                .head(SAVE_TOP_SEARCH_N)
                .copy()
            )
            top_search["search_rank"] = np.arange(1, len(top_search) + 1)
            top_search_rows.extend(top_search.to_dict("records"))
        finalists = (
            combined.sort_values(["regret@1", "success", "runtime_s"], ascending=[True, False, True])
            .drop_duplicates(subset=["subset_id"])
            .head(SEARCH_TOP_FINAL)
        )
        final_records = [
            SubsetRecord(
                dataset_name=dataset_name,
                k=k,
                policy=row["policy"],
                seed=int(row["seed"]),
                indices=_normalize_indices(json.loads(row["selected_indices"])),
            )
            for row in finalists.to_dict("records")
        ]
        final_rows.extend(_final_rescore_parallel(spec, final_records, workers=workers))
        print(
            f"[study] dataset={dataset_name} k={k} final_rescore_done finalists={len(final_records)}",
            flush=True,
        )
        if EVALUATE_DEPLOYABLE:
            deployable_rows.extend(_evaluate_deployable_policy(spec, dataset_name=dataset_name, k=k, workers=workers))
            print(f"[study] dataset={dataset_name} k={k} deployable_eval_done", flush=True)
        else:
            print(f"[study] dataset={dataset_name} k={k} deployable_eval_skipped", flush=True)

    return (
        pd.DataFrame(search_rows),
        pd.DataFrame(top_search_rows),
        pd.DataFrame(final_rows),
        pd.DataFrame(deployable_rows),
    )


def _write_outputs(
    *,
    output_dir: Path,
    search_df: pd.DataFrame,
    top_search_df: pd.DataFrame,
    final_df: pd.DataFrame,
    deployable_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(output_dir / "search_results.csv", index=False)
    top_search_df.to_csv(output_dir / "top_search_candidates.csv", index=False)
    final_df.to_csv(output_dir / "oracle_finalists.csv", index=False)
    deployable_df.to_csv(output_dir / "deployable_policy.csv", index=False)

    if final_df.empty:
        pd.DataFrame().to_csv(output_dir / "oracle_summary.csv", index=False)
        with open(output_dir / "recommendations.json", "w") as f:
            json.dump({"objective_metric": DEFAULT_STARCODER_OBJECTIVE, "recommended_k": {}, "oracle_ceiling": []}, f)
        return

    oracle_ceiling = (
        final_df.sort_values(["dataset", "k", "median_regret@1", "success_rate"])
        .groupby(["dataset", "k"], as_index=False)
        .first()
    )
    recommendations = _recommended_k(oracle_ceiling)
    random_baseline = (
        search_df[search_df["policy"] == "random_observed"]
        .groupby(["dataset", "k"], as_index=False)["regret@1"]
        .median()
        .rename(columns={"regret@1": "random_median_regret@1"})
    )
    summary = oracle_ceiling.merge(random_baseline, on=["dataset", "k"], how="left")
    summary["material_improvement"] = (
        summary["median_regret@1"] <= summary["random_median_regret@1"] * MATERIAL_IMPROVEMENT
    )
    summary.to_csv(output_dir / "oracle_summary.csv", index=False)

    with open(output_dir / "recommendations.json", "w") as f:
        json.dump(
            {
                "objective_metric": DEFAULT_STARCODER_OBJECTIVE,
                "recommended_k": recommendations,
                "oracle_ceiling": summary.to_dict(orient="records"),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _configure_from_args(args: argparse.Namespace) -> None:
    global SEARCH_RANDOM_SUBSETS
    global SEARCH_KCENTER_SUBSETS
    global SEARCH_DOPT_SUBSETS
    global SEARCH_TOP_SUBSETS
    global SEARCH_TOP_FINAL
    global HILL_CLIMB_MAX_ACCEPTED
    global HILL_CLIMB_SWAP_CANDIDATES
    global DEPLOYABLE_SEEDS
    global SEARCH_RESTARTS
    global SEARCH_MAXITER
    global FINAL_RESTARTS
    global FINAL_MAXITER
    global FINAL_SEEDS
    global SAVE_TOP_SEARCH_N
    global EVALUATE_DEPLOYABLE
    global PARALLEL_BACKEND

    SEARCH_RANDOM_SUBSETS = args.search_random_subsets
    SEARCH_KCENTER_SUBSETS = args.search_kcenter_subsets
    SEARCH_DOPT_SUBSETS = args.search_dopt_subsets
    SEARCH_TOP_SUBSETS = args.search_top_subsets
    SEARCH_TOP_FINAL = args.search_top_final
    HILL_CLIMB_MAX_ACCEPTED = args.hill_climb_max_accepted
    HILL_CLIMB_SWAP_CANDIDATES = args.hill_climb_swap_candidates
    DEPLOYABLE_SEEDS = args.deployable_seeds
    SEARCH_RESTARTS = args.search_restarts
    SEARCH_MAXITER = args.search_maxiter
    FINAL_RESTARTS = args.final_restarts
    FINAL_MAXITER = args.final_maxiter
    FINAL_SEEDS = args.final_seeds
    SAVE_TOP_SEARCH_N = args.save_top_search_n
    EVALUATE_DEPLOYABLE = not args.skip_deployable
    PARALLEL_BACKEND = args.parallel_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StarCoder subset-selection study")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_STARCODER_OBJECTIVE)
    parser.add_argument(
        "--datasets",
        type=str,
        default="two_phase_starcoder,three_phase_starcoder",
        help="Comma-separated dataset names",
    )
    parser.add_argument("--k-values", type=str, default="", help="Optional comma-separated override for k values")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))))
    parser.add_argument("--search-random-subsets", type=int, default=SEARCH_RANDOM_SUBSETS)
    parser.add_argument("--search-kcenter-subsets", type=int, default=SEARCH_KCENTER_SUBSETS)
    parser.add_argument("--search-dopt-subsets", type=int, default=SEARCH_DOPT_SUBSETS)
    parser.add_argument("--search-top-subsets", type=int, default=SEARCH_TOP_SUBSETS)
    parser.add_argument("--search-top-final", type=int, default=SEARCH_TOP_FINAL)
    parser.add_argument("--hill-climb-max-accepted", type=int, default=HILL_CLIMB_MAX_ACCEPTED)
    parser.add_argument("--hill-climb-swap-candidates", type=int, default=HILL_CLIMB_SWAP_CANDIDATES)
    parser.add_argument("--deployable-seeds", type=int, default=DEPLOYABLE_SEEDS)
    parser.add_argument("--search-restarts", type=int, default=SEARCH_RESTARTS)
    parser.add_argument("--search-maxiter", type=int, default=SEARCH_MAXITER)
    parser.add_argument("--final-restarts", type=int, default=FINAL_RESTARTS)
    parser.add_argument("--final-maxiter", type=int, default=FINAL_MAXITER)
    parser.add_argument("--final-seeds", type=int, default=FINAL_SEEDS)
    parser.add_argument("--save-top-search-n", type=int, default=SAVE_TOP_SEARCH_N)
    parser.add_argument("--parallel-backend", choices=("thread", "process"), default=PARALLEL_BACKEND)
    parser.add_argument("--skip-deployable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_from_args(args)
    dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}
    override_k_values = _parse_int_list(args.k_values) if args.k_values else ()
    search_frames: list[pd.DataFrame] = []
    top_search_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    deployable_frames: list[pd.DataFrame] = []

    definitions = []
    if TWO_PHASE_STARCODER.name in dataset_names:
        definitions.append((TWO_PHASE_STARCODER, override_k_values or TWO_PHASE_K))
    if THREE_PHASE_STARCODER.name in dataset_names:
        definitions.append((THREE_PHASE_STARCODER, override_k_values or THREE_PHASE_K))

    for definition, k_values in definitions:
        spec, _ = load_starcoder_dataset(definition, target_col=args.objective_metric)
        search_df, top_search_df, final_df, deployable_df = _search_dataset(
            spec,
            dataset_name=definition.name,
            k_values=k_values,
            workers=args.workers,
        )
        search_frames.append(search_df)
        top_search_frames.append(top_search_df)
        final_frames.append(final_df)
        deployable_frames.append(deployable_df)

    _write_outputs(
        output_dir=args.output_dir,
        search_df=pd.concat(search_frames, ignore_index=True) if search_frames else pd.DataFrame(),
        top_search_df=pd.concat(top_search_frames, ignore_index=True) if top_search_frames else pd.DataFrame(),
        final_df=pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame(),
        deployable_df=pd.concat(deployable_frames, ignore_index=True) if deployable_frames else pd.DataFrame(),
    )


if __name__ == "__main__":
    main()
