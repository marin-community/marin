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
"""Export a self-contained data bundle for the 300M mixture-fit debugger.

The exporter deliberately separates three prediction regimes:

* the 280 collapsed fit designs receive grouped out-of-fold predictions;
* checkpoints outside the fit panel receive predictions from a full 280-row fit;
* proportional repeats remain a distinct noise-reference stratum.

Each fold and full-fit prediction is cached independently. Re-running the script
therefore resumes after interruption without refitting completed model/fold pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_original_separate_heads_policy_ablation_300m as separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (  # noqa: E402
    GenericFamilyPacket,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (  # noqa: E402
    build_penalty_calibration_surrogate,
    penalty_calibration_param_keys,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (  # noqa: E402
    PacketData,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PACKET_PATH = SCRIPT_DIR / "reference_outputs/two_phase_solver_gap_collaborator_packet_20260701.zip"
APP_DIR = SCRIPT_DIR / "mixture_fit_debugger"
DEFAULT_OUTPUT_JSON = APP_DIR / "src/generated/dashboard_data.json"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "reference_outputs/mixture_fit_debugger_fit_cache_300m_20260712"
PACKET_ROOT = "two_phase_solver_gap_collaborator_packet_20260701/data"
FIT_FILE = "fit_matrix_collapsed_proportional_300m.csv"
HELDOUT_FILE = "heldout_300m_checkpoint_metrics.csv"
ALL_FILE = "all_300m_checkpoint_metrics.csv"
TARGET_IDS = ("uncheatable", "table9")
MODEL_IDS = ("canonical", "effective_exposure", "effective_exposure_geometry", "separate_heads", "grp")
TARGET_COLUMNS = {
    "uncheatable": "eval_uncheatable_eval_bpb",
    "table9": "table9_macro_bpb",
}
TARGET_LABELS = {
    "uncheatable": "Uncheatable eval BPB",
    "table9": "OLMoBaseEval Table-9 macro BPB",
}
SEPARATE_HEADS_L2 = {"uncheatable": 1.0, "table9": 1.5}
DSP_LINEAR_REG = 0.01
DSP_MAXITER = 16
DSP_COARSE_TOP_K = 3
GRP_VARIANT = "power_family_penalty"
# Preserve the old regularized GRP's fitted shape and retune only its ridge strength
# on the current target. L2=0 is an endpoint of this same model, not the separate
# historical no-L2 variant.
GRP_SHAPE_PARAMS = {
    "eta": 6.627794351309641,
    "lam": 6.14421235332821e-06,
    "beta": 0.2629059619755788,
    "a_broad_text": 0.6462737477673589,
    "a_tech_code": 0.1657586322714625,
    "a_reasoning": 0.2076641777781618,
    "tau_broad_text": 3.193090495213877,
    "tau_tech_code": 6.2042610686315145,
    "tau_reasoning": 5.136810831800622,
}
GRP_L2_GRID = (
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    1.0,
    3.0,
    10.0,
)
LOWER_TAIL_FRACTION = 0.15
CACHE_VERSION = "debugger-fit-protocol-v1"
GRP_CACHE_VERSION = "regularized-grp-v1"
OLMIX_CANDIDATES = {
    "uncheatable": (
        SCRIPT_DIR / "reference_outputs/olmix_huber_delta_sweep_300m_20260625/delta_0p01/"
        "uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4/proposed_mixture_weights.csv"
    ),
    "table9": (
        SCRIPT_DIR / "reference_outputs/olmo_base_easy_paper_faithful_olmix_300m_20260625/"
        "single_tied_delta_0p01/proposed_mixture_weights.csv"
    ),
}


@dataclass(frozen=True)
class FitTask:
    key: str
    target: str
    model: str
    train_indices: tuple[int, ...]
    predict_weights: np.ndarray
    output_indices: tuple[int, ...]
    regularization: float | None = None


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def read_packet_csv(packet_path: Path, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(packet_path) as archive:
        return pd.read_csv(archive.open(f"{PACKET_ROOT}/{filename}"))


def normalized_weights(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    phase0 = frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float, copy=True)
    phase1 = frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float, copy=True)
    phase0 /= phase0.sum(axis=1, keepdims=True)
    phase1 /= phase1.sum(axis=1, keepdims=True)
    return np.stack([phase0, phase1], axis=1)


def weights_from_mixture_file(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("domain").loc[domains]
    phase0 = frame["phase_0_weight"].to_numpy(dtype=float, copy=True)
    phase1 = frame["phase_1_weight"].to_numpy(dtype=float, copy=True)
    phase0 /= phase0.sum()
    phase1 /= phase1.sum()
    return np.stack([phase0, phase1], axis=0)


def safe_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def row_identifier(split: str, packet_row_id: Any, run_name: str) -> str:
    source = safe_string(packet_row_id) or run_name
    compact = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", source).strip("-")
    return f"{split}:{compact}"


def merged_fit_rows(fit_frame: pd.DataFrame, all_frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if fit_frame["run_name"].duplicated().any():
        raise ValueError("Collapsed fit matrix has duplicate run names")
    metadata = all_frame.loc[all_frame["packet_panel"].eq("augmented_fit_panel")].copy()
    if metadata["run_name"].duplicated().any():
        raise ValueError("Raw augmented fit panel has duplicate run names")
    metadata = metadata.set_index("run_name")
    fit = fit_frame.set_index("run_name").loc[order].copy()
    missing_columns = [column for column in metadata.columns if column not in fit.columns]
    fit = pd.concat([fit, metadata.loc[fit.index, missing_columns]], axis=1)
    return fit.reset_index()


def build_row_frame(packet_path: Path, domains: list[str], fit_order: list[str]) -> pd.DataFrame:
    all_frame = read_packet_csv(packet_path, ALL_FILE)
    fit = merged_fit_rows(read_packet_csv(packet_path, FIT_FILE), all_frame, fit_order)
    fit = pd.concat([fit, pd.Series("fit", index=fit.index, name="display_split")], axis=1)
    heldout_frame = read_packet_csv(packet_path, HELDOUT_FILE)
    heldout = pd.concat(
        [
            heldout_frame,
            pd.Series("heldout", index=heldout_frame.index, name="display_split"),
        ],
        axis=1,
    )
    noise = all_frame.loc[all_frame["packet_panel"].eq("proportional_noise_reference")].copy()
    noise = pd.concat(
        [noise.reset_index(drop=True), pd.Series("noise_reference", index=range(len(noise)), name="display_split")],
        axis=1,
    )
    if (len(fit), len(heldout), len(noise)) != (280, 414, 10):
        raise ValueError(f"Unexpected packet accounting: fit={len(fit)}, heldout={len(heldout)}, noise={len(noise)}")
    rows = pd.concat([fit, heldout, noise], ignore_index=True, sort=False)
    identifiers = pd.Series(
        [
            row_identifier(split, packet_row_id, run_name)
            for split, packet_row_id, run_name in zip(
                rows["display_split"], rows.get("packet_row_id"), rows["run_name"], strict=True
            )
        ],
        name="dashboard_row_id",
    )
    rows = pd.concat([rows, identifiers], axis=1)
    if rows["dashboard_row_id"].duplicated().any():
        duplicates = rows.loc[rows["dashboard_row_id"].duplicated(), "dashboard_row_id"].tolist()
        raise ValueError(f"Duplicate dashboard row IDs: {duplicates[:5]}")
    return rows


def append_olmix_candidates(rows: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, path in OLMIX_CANDIDATES.items():
        weights = weights_from_mixture_file(path, domains)
        record: dict[str, Any] = {
            "display_split": "candidate",
            "split": "candidate",
            "policy_family": "single_phase",
            "training_phase_family": "single_phase",
            "phase_weight_structure": "tied_weights",
            "packet_panel": "comparison_candidate",
            "packet_method": "olmix_single_phase_fit",
            "run_name": f"olmix_single_phase_{target}",
            "dashboard_row_id": f"candidate:olmix_single_phase_{target}",
            "is_shared_checkpoint_alias": False,
            "candidate_target": target,
        }
        for phase in range(2):
            for domain_index, domain in enumerate(domains):
                record[f"phase_{phase}_{domain}"] = float(weights[phase, domain_index])
        records.append(record)
    return pd.concat([rows, pd.DataFrame.from_records(records)], ignore_index=True, sort=False)


def dataset_for_target(target: str) -> pooled.Dataset:
    if target not in TARGET_IDS:
        raise ValueError(f"Unknown target {target!r}")
    return pooled.load_300m_dataset(target)


def grp_packet(dataset: pooled.Dataset) -> GenericFamilyPacket:
    base = PacketData(
        frame=dataset.frame.copy(),
        name_col="run_name",
        y=np.asarray(dataset.y, dtype=float),
        w=np.asarray(dataset.weights, dtype=float),
        m=dataset.m,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )
    reference = load_generic_family_packet()
    if reference.base.domain_names != base.domain_names:
        raise ValueError("GRP domain ordering differs from the 300M fit panel")
    return replace(reference, base=base)


def grp_params(l2: float) -> dict[str, float]:
    if l2 < 0.0:
        raise ValueError(f"GRP requires nonnegative L2, got {l2}")
    return {**GRP_SHAPE_PARAMS, "reg": float(l2)}


def fit_grp_predictions(
    dataset: pooled.Dataset,
    train_indices: np.ndarray,
    predict_weights: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, int]:
    packet = grp_packet(dataset)
    model = build_penalty_calibration_surrogate(
        packet,
        params=grp_params(l2),
        variant_name=GRP_VARIANT,
    ).fit(dataset.weights[train_indices], dataset.y[train_indices])
    if model.coef_ is None:
        raise RuntimeError("Regularized GRP fit did not produce coefficients")
    return model.predict(predict_weights), len(model.coef_)


def select_grp_l2(dataset: pooled.Dataset, seeds: tuple[int, ...]) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for l2 in GRP_L2_GRID:
        seed_predictions: list[np.ndarray] = []
        for seed in seeds:
            oof = np.full(dataset.n, np.nan, dtype=float)
            for train_indices, test_indices in pooled.dataset_folds(dataset, seed, n_splits=5):
                prediction, _count = fit_grp_predictions(dataset, train_indices, dataset.weights[test_indices], l2)
                oof[test_indices] = prediction
            if not np.isfinite(oof).all():
                raise ValueError(f"Incomplete GRP OOF prediction for {dataset.name}, seed={seed}, L2={l2}")
            seed_predictions.append(oof)
        mean_prediction = np.mean(seed_predictions, axis=0)
        residual = mean_prediction - dataset.y
        rows.append(
            {
                "l2": float(l2),
                "oofRmse": float(np.sqrt(np.mean(residual**2))),
                "oofSpearman": float(spearmanr(dataset.y, mean_prediction).statistic),
            }
        )
    selected = min(rows, key=lambda row: (row["oofRmse"], -row["oofSpearman"], row["l2"]))
    return float(selected["l2"]), rows


def fit_and_predict(task: FitTask) -> tuple[np.ndarray, dict[str, Any]]:
    dataset = dataset_for_target(task.target)
    train_indices = np.asarray(task.train_indices, dtype=int)
    weights = np.asarray(task.predict_weights, dtype=float)
    if task.model == "grp":
        if task.regularization is None:
            raise ValueError("Regularized GRP task is missing its selected L2")
        prediction, linear_coefficient_count = fit_grp_predictions(
            dataset,
            train_indices,
            weights,
            task.regularization,
        )
        nonlinear_parameter_count = len(penalty_calibration_param_keys(GRP_VARIANT))
        summary = {
            "l2": task.regularization,
            "linearCoefficientCount": linear_coefficient_count,
            "interceptCount": 1,
            "nonlinearParameterCount": nonlinear_parameter_count,
            "parameterCount": linear_coefficient_count + 1 + nonlinear_parameter_count,
        }
        return np.asarray(prediction, dtype=float), summary
    if task.model == "separate_heads":
        packet, _domains, _natural, _token_counts, _target_budget, _folds = bowl.load_objective(task.target)
        l2 = SEPARATE_HEADS_L2[task.target]
        model = separate_heads.fit_separate_heads(packet, train_indices, l2)
        prediction = separate_heads.predict_separate_heads(model, packet, weights)
        summary = {
            "l2": l2,
            "intercept": float(model.intercept),
            "parameterCount": int(4 * packet.m + 1 + 2 * packet.m),
        }
        return np.asarray(prediction, dtype=float), summary

    use_geometry = task.model == "effective_exposure_geometry"
    variant_name = "canonical" if task.model == "canonical" else "effective_exposure"
    config = coverage_dsp.FitConfig(
        name=task.model,
        use_coverage=use_geometry,
        variant_name=variant_name,
    )
    model = coverage_dsp.fit_model(
        dataset,
        train_indices,
        config,
        linear_reg=DSP_LINEAR_REG,
        maxiter=DSP_MAXITER,
        coarse_top_k=DSP_COARSE_TOP_K,
    )
    alpha0, alpha1 = coverage_dsp.phase_fractions(dataset)
    prediction = coverage_dsp.predict(model, weights, alpha0, alpha1)
    summary = {
        "linearReg": DSP_LINEAR_REG,
        "gamma": safe_float(model.base.params.get("gamma")),
        "coverageCoefficients": model.coverage_coef.tolist(),
        "parameterCount": model.base.total_param_count + int(use_geometry) * 3,
    }
    return np.asarray(prediction, dtype=float), summary


def cache_fingerprint(
    packet_path: Path,
    row_ids: list[str],
    seeds: tuple[int, ...],
) -> str:
    stat = packet_path.stat()
    payload = {
        "cacheVersion": CACHE_VERSION,
        "packetSize": stat.st_size,
        "packetMtimeNs": stat.st_mtime_ns,
        "rowIds": row_ids,
        "seeds": seeds,
        "dspLinearReg": DSP_LINEAR_REG,
        "dspMaxiter": DSP_MAXITER,
        "separateHeadsL2": SEPARATE_HEADS_L2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(cache_dir: Path, fingerprint: str, task_key: str) -> Path:
    return cache_dir / fingerprint[:16] / f"{task_key}.npz"


def load_cached_prediction(path: Path, fingerprint: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            return None
        indices = np.asarray(data["indices"], dtype=int)
        prediction = np.asarray(data["prediction"], dtype=float)
        summary = json.loads(str(data["summary"].item()))
    return indices, prediction, summary


def write_cached_prediction(
    path: Path,
    fingerprint: str,
    indices: tuple[int, ...],
    prediction: np.ndarray,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        fingerprint=np.asarray(fingerprint),
        indices=np.asarray(indices, dtype=int),
        prediction=np.asarray(prediction, dtype=float),
        summary=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    temporary.replace(path)


def build_tasks(
    rows_weights: np.ndarray,
    seeds: tuple[int, ...],
    grp_l2_by_target: dict[str, float],
) -> list[FitTask]:
    tasks: list[FitTask] = []
    all_output_indices = tuple(range(len(rows_weights)))
    for target in TARGET_IDS:
        dataset = dataset_for_target(target)
        all_train = tuple(range(dataset.n))
        for model in MODEL_IDS:
            grp_suffix = f"__{GRP_CACHE_VERSION}__l2{grp_l2_by_target[target]:.12g}" if model == "grp" else ""
            tasks.append(
                FitTask(
                    key=f"full__{target}__{model}{grp_suffix}",
                    target=target,
                    model=model,
                    train_indices=all_train,
                    predict_weights=rows_weights,
                    output_indices=all_output_indices,
                    regularization=(grp_l2_by_target[target] if model == "grp" else None),
                )
            )
            for seed in seeds:
                for fold_id, (train_indices, test_indices) in enumerate(pooled.dataset_folds(dataset, seed, n_splits=5)):
                    tasks.append(
                        FitTask(
                            key=f"oof__{target}__{model}__seed{seed}__fold{fold_id}{grp_suffix}",
                            target=target,
                            model=model,
                            train_indices=tuple(int(index) for index in train_indices),
                            predict_weights=dataset.weights[test_indices],
                            output_indices=tuple(int(index) for index in test_indices),
                            regularization=(grp_l2_by_target[target] if model == "grp" else None),
                        )
                    )
    return tasks


def execute_tasks(
    tasks: list[FitTask],
    cache_dir: Path,
    fingerprint: str,
    workers: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    results: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    pending: list[FitTask] = []
    for task in tasks:
        cached = load_cached_prediction(cache_path(cache_dir, fingerprint, task.key), fingerprint)
        if cached is None:
            pending.append(task)
        else:
            results[task.key] = cached
    print(f"Prediction cache: {len(results)} hit(s), {len(pending)} fit task(s) pending", flush=True)
    if not pending:
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_and_predict, task): task for task in pending}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            prediction, summary = future.result()
            indices = np.asarray(task.output_indices, dtype=int)
            if len(prediction) != len(indices):
                raise ValueError(f"{task.key}: prediction length {len(prediction)} != {len(indices)}")
            path = cache_path(cache_dir, fingerprint, task.key)
            write_cached_prediction(path, fingerprint, task.output_indices, prediction, summary)
            results[task.key] = indices, prediction, summary
            completed += 1
            print(f"[{completed}/{len(pending)}] fitted {task.key}", flush=True)
    return results


def aggregate_predictions(
    rows: pd.DataFrame,
    seeds: tuple[int, ...],
    results: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: dict[str, Any] = {}
    model_summaries: dict[str, Any] = {}
    fit_count = 280
    for target in TARGET_IDS:
        predictions[target] = {}
        model_summaries[target] = {}
        for model in MODEL_IDS:
            grp_suffix = ""
            if model == "grp":
                grp_full_keys = [key for key in results if key.startswith(f"full__{target}__grp__{GRP_CACHE_VERSION}__")]
                if len(grp_full_keys) != 1:
                    raise ValueError(f"Expected one full-fit GRP result for {target}, got {grp_full_keys}")
                full_key = grp_full_keys[0]
                grp_suffix = full_key.removeprefix(f"full__{target}__grp")
            else:
                full_key = f"full__{target}__{model}"
            _indices, full_prediction, full_summary = results[full_key]
            seed_predictions: list[np.ndarray] = []
            for seed in seeds:
                oof = np.full(fit_count, np.nan, dtype=float)
                for fold_id in range(5):
                    fold_key = f"oof__{target}__{model}__seed{seed}__fold{fold_id}{grp_suffix}"
                    indices, fold_prediction, _summary = results[fold_key]
                    oof[indices] = fold_prediction
                if not np.isfinite(oof).all():
                    raise ValueError(f"Incomplete OOF prediction for {target}/{model}/seed={seed}")
                seed_predictions.append(oof)
            oof_mean = np.mean(seed_predictions, axis=0)
            honest_prediction = np.asarray(full_prediction, dtype=float).copy()
            honest_prediction[:fit_count] = oof_mean
            predictions[target][model] = {
                "prediction": [safe_float(value) for value in honest_prediction],
                "fullFitPrediction": [safe_float(value) for value in full_prediction],
            }
            model_summaries[target][model] = full_summary
    return predictions, model_summaries


def metric_summary(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(observed) & np.isfinite(prediction)
    if valid.sum() < 3:
        return {"n": int(valid.sum()), "rmse": None, "mae": None, "spearman": None}
    residual = prediction[valid] - observed[valid]
    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed[valid], prediction[valid]).statistic),
    }


def prediction_diagnostics(
    rows: pd.DataFrame,
    predictions: dict[str, Any],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    alias = rows["is_shared_checkpoint_alias"].map(safe_bool).to_numpy(dtype=bool)
    split = rows["display_split"].astype(str).to_numpy()
    phase_family = rows["training_phase_family"].astype(str).to_numpy()
    for target, target_column in TARGET_COLUMNS.items():
        observed = pd.to_numeric(rows.get(target_column), errors="coerce").to_numpy(dtype=float)
        diagnostics[target] = {}
        for model in MODEL_IDS:
            prediction = np.asarray(predictions[target][model]["prediction"], dtype=float)
            diagnostics[target][model] = {
                "fitOof": metric_summary(observed[split == "fit"], prediction[split == "fit"]),
                "heldout": metric_summary(
                    observed[(split == "heldout") & ~alias], prediction[(split == "heldout") & ~alias]
                ),
                "heldoutSinglePhase": metric_summary(
                    observed[(split == "heldout") & ~alias & (phase_family == "single_phase")],
                    prediction[(split == "heldout") & ~alias & (phase_family == "single_phase")],
                ),
                "heldoutTwoPhase": metric_summary(
                    observed[(split == "heldout") & ~alias & (phase_family == "two_phase")],
                    prediction[(split == "heldout") & ~alias & (phase_family == "two_phase")],
                ),
            }
    return diagnostics


def nearest_fit_diagnostics(all_weights: np.ndarray, fit_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nearest_indices = np.zeros(len(all_weights), dtype=int)
    nearest_distances = np.zeros(len(all_weights), dtype=float)
    for row_index, weights in enumerate(all_weights):
        distance = np.abs(fit_weights - weights[None, :, :]).sum(axis=(1, 2)) / 4.0
        if row_index < len(fit_weights):
            distance[row_index] = np.inf
        nearest = int(np.argmin(distance))
        nearest_indices[row_index] = nearest
        nearest_distances[row_index] = float(distance[nearest])
    return nearest_indices, nearest_distances


def target_noise_references(all_frame: pd.DataFrame) -> dict[str, Any]:
    mask = all_frame["packet_panel"].isin(["augmented_fit_panel", "proportional_noise_reference"])
    mask &= all_frame["run_name"].eq("baseline_proportional") | all_frame["packet_panel"].eq(
        "proportional_noise_reference"
    )
    references: dict[str, Any] = {}
    for target, column in TARGET_COLUMNS.items():
        values = pd.to_numeric(all_frame.loc[mask, column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) != 11:
            raise ValueError(f"Expected 11 proportional observations for {target}, got {len(values)}")
        standard_deviation = float(np.std(values, ddof=1))
        references[target] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "standardDeviation": standard_deviation,
            "differenceStandardDeviation": float(np.sqrt(2.0) * standard_deviation),
        }
    return references


def display_domain_name(domain: str) -> str:
    value = domain.replace("dolma3_", "").replace("dolmino_", "")
    value = value.replace("cc/", "CC · ").replace("_", " ")
    return value.replace(" and ", " & ").title()


def domain_group(domain: str) -> str:
    if "/" in domain:
        return "Common Crawl taxonomy"
    if domain.startswith("dolmino_"):
        return "Dolmino"
    return "Dolma 3"


def make_row_records(
    rows: pd.DataFrame,
    weights: np.ndarray,
    domains: list[str],
    natural: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> list[dict[str, Any]]:
    fit_weights = weights[:280]
    nearest_indices, nearest_distances = nearest_fit_diagnostics(weights, fit_weights)
    natural_clipped = np.clip(natural, 1e-12, 1.0)
    records: list[dict[str, Any]] = []
    for row_index, (_, source) in enumerate(rows.iterrows()):
        row_weights = weights[row_index]
        aggregate = alpha0 * row_weights[0] + alpha1 * row_weights[1]
        phase0_epochs = row_weights[0] * c0
        phase1_epochs = row_weights[1] * c1
        total_epochs = phase0_epochs + phase1_epochs
        aggregate_clipped = np.clip(aggregate, 1e-12, 1.0)
        records.append(
            {
                "id": str(source["dashboard_row_id"]),
                "name": str(source["run_name"]),
                "split": str(source["display_split"]),
                "policyFamily": safe_string(source.get("policy_family")),
                "phaseFamily": safe_string(source.get("training_phase_family")),
                "phaseStructure": safe_string(source.get("phase_weight_structure")),
                "panel": safe_string(source.get("packet_panel")),
                "method": safe_string(source.get("packet_method")),
                "sourceExperiment": safe_string(source.get("source_experiment")),
                "wandbUrl": safe_string(source.get("training_wandb_url")),
                "interventionType": safe_string(source.get("intervention_type")),
                "targetDomain": safe_string(source.get("target_domain")),
                "directionType": safe_string(source.get("direction_type")),
                "directionId": safe_string(source.get("direction_id")),
                "isSharedAlias": safe_bool(source.get("is_shared_checkpoint_alias")),
                "pairedRow": (
                    safe_string(source.get("paired_single_phase_run_name"))
                    or safe_string(source.get("paired_two_phase_run_name"))
                ),
                "candidateTarget": safe_string(source.get("candidate_target")),
                "observed": {target: safe_float(source.get(column)) for target, column in TARGET_COLUMNS.items()},
                "phase0": row_weights[0].tolist(),
                "phase1": row_weights[1].tolist(),
                "aggregate": aggregate.tolist(),
                "phase0Epochs": phase0_epochs.tolist(),
                "phase1Epochs": phase1_epochs.tolist(),
                "totalEpochs": total_epochs.tolist(),
                "diagnostics": {
                    "phaseTv": float(0.5 * np.abs(row_weights[0] - row_weights[1]).sum()),
                    "aggregateTvToProportional": float(0.5 * np.abs(aggregate - natural).sum()),
                    "aggregateKlToProportional": float(
                        np.sum(aggregate_clipped * np.log(aggregate_clipped / natural_clipped))
                    ),
                    "maxEpoch": float(np.max(total_epochs)),
                    "nearestFitId": str(rows.iloc[nearest_indices[row_index]]["dashboard_row_id"]),
                    "supportDistance": float(nearest_distances[row_index]),
                },
            }
        )
    return records


def baseline_registry(rows: pd.DataFrame) -> dict[str, Any]:
    by_name = dict(zip(rows["run_name"], rows["dashboard_row_id"], strict=True))
    registry: dict[str, Any] = {}
    independent_heldout = rows["display_split"].eq("heldout") & ~rows["is_shared_checkpoint_alias"].map(safe_bool)
    for target, column in TARGET_COLUMNS.items():
        values = pd.to_numeric(rows[column], errors="coerce")
        single_mask = independent_heldout & rows["training_phase_family"].eq("single_phase") & values.notna()
        two_mask = independent_heldout & rows["training_phase_family"].eq("two_phase") & values.notna()
        single_index = values.loc[single_mask].idxmin()
        two_index = values.loc[two_mask].idxmin()
        registry[target] = [
            {"id": by_name["baseline_proportional"], "label": "Proportional"},
            {"id": by_name["baseline_unimax"], "label": "UniMax-8"},
            {
                "id": str(rows.loc[single_index, "dashboard_row_id"]),
                "label": "Empirical one-phase frontier",
            },
            {
                "id": str(rows.loc[two_index, "dashboard_row_id"]),
                "label": "Empirical two-phase frontier",
            },
            {
                "id": f"candidate:olmix_single_phase_{target}",
                "label": "OLMix one-phase fit (unvalidated at 300M)",
            },
        ]
    return registry


def model_metadata(model_summaries: dict[str, Any]) -> dict[str, Any]:
    labels = {
        "canonical": "Canonical DSP",
        "effective_exposure": "Effective-exposure DSP",
        "effective_exposure_geometry": "Eff-exp DSP + geometry",
        "separate_heads": "Separate heads",
        "grp": "GRP (regularized)",
    }
    descriptions = {
        "canonical": "Shared phase premium on benefit; overexposure uses raw total exposure.",
        "effective_exposure": "Shared phase multiplier enters both saturation and overexposure exposure.",
        "effective_exposure_geometry": (
            "Effective-exposure DSP plus phase TV and aggregate/late-phase concentration terms."
        ),
        "separate_heads": "Independent two-sided exposure bowls for phase 0 and phase 1.",
        "grp": (
            "Historical power-family GRP with CC pairing, family benefit terms, family overexposure penalties, "
            "and a target-tuned ridge penalty."
        ),
    }
    return {
        model: {
            "id": model,
            "label": labels[model],
            "description": descriptions[model],
            "protocol": {
                "oof": "Five-fold panel-stratified; predictions averaged over configured seeds.",
                "fullFit": "Refit on all 280 collapsed designs for heldout and candidate prediction.",
                "targetParameters": {target: model_summaries[target][model] for target in TARGET_IDS},
            },
        }
        for model in MODEL_IDS
    }


def write_bundle(
    packet_path: Path,
    output_json: Path,
    cache_dir: Path,
    seeds: tuple[int, ...],
    workers: int,
) -> dict[str, Any]:
    datasets = {target: dataset_for_target(target) for target in TARGET_IDS}
    fit_order = datasets["uncheatable"].frame["run_name"].astype(str).tolist()
    if datasets["table9"].frame["run_name"].astype(str).tolist() != fit_order:
        raise ValueError("Uncheatable and Table-9 fit panels have different row order")
    domains = list(datasets["uncheatable"].domain_names)
    rows = append_olmix_candidates(build_row_frame(packet_path, domains, fit_order), domains)
    weights = normalized_weights(rows, domains)
    dataset = datasets["uncheatable"]
    _packet, _domains, natural, token_counts, target_budget, _folds = bowl.load_objective("uncheatable")
    alpha0, alpha1 = coverage_dsp.phase_fractions(dataset)
    grp_sweeps: dict[str, list[dict[str, float]]] = {}
    grp_l2_by_target: dict[str, float] = {}
    for target, target_dataset in datasets.items():
        selected_l2, sweep_rows = select_grp_l2(target_dataset, seeds)
        grp_l2_by_target[target] = selected_l2
        grp_sweeps[target] = sweep_rows
        print(f"Regularized GRP {target}: selected L2={selected_l2:g}", flush=True)
    fingerprint = cache_fingerprint(packet_path, rows["dashboard_row_id"].astype(str).tolist(), seeds)
    tasks = build_tasks(weights, seeds, grp_l2_by_target)
    results = execute_tasks(tasks, cache_dir, fingerprint, workers)
    predictions, model_summaries = aggregate_predictions(rows, seeds, results)
    for target in TARGET_IDS:
        selected_row = next(row for row in grp_sweeps[target] if row["l2"] == grp_l2_by_target[target])
        model_summaries[target]["grp"].update(
            {
                "l2SelectionMetric": "three-seed panel-stratified OOF RMSE",
                "l2SelectionOofRmse": selected_row["oofRmse"],
                "l2SelectionOofSpearman": selected_row["oofSpearman"],
                "l2Sweep": grp_sweeps[target],
            }
        )
    all_frame = read_packet_csv(packet_path, ALL_FILE)
    bundle = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataset": {
            "label": "300M / 6B-token Dolma 3 + Dolmino swarm",
            "fitDesignCount": 280,
            "rawFitObservationCount": 290,
            "heldoutCount": 414,
            "noiseReferenceCount": 10,
            "supplementalCandidateCount": 2,
            "phaseFractions": [alpha0, alpha1],
            "targetBudget": int(target_budget),
            "oofSeeds": list(seeds),
            "fitProtocol": "Collapsed proportional mean; grouped nested OOF for fit points; full-fit for heldouts.",
        },
        "domains": [
            {
                "id": domain,
                "label": display_domain_name(domain),
                "group": domain_group(domain),
                "proportionalWeight": float(natural[index]),
                "tokenCount": float(token_counts[index]),
                "phase0EpochFactor": float(dataset.c0[index]),
                "phase1EpochFactor": float(dataset.c1[index]),
            }
            for index, domain in enumerate(domains)
        ],
        "targets": {
            target: {
                "id": target,
                "label": TARGET_LABELS[target],
                "metricColumn": TARGET_COLUMNS[target],
                "lowerIsBetter": True,
                "noiseReference": target_noise_references(all_frame)[target],
            }
            for target in TARGET_IDS
        },
        "models": model_metadata(model_summaries),
        "rows": make_row_records(
            rows,
            weights,
            domains,
            np.asarray(natural, dtype=float),
            dataset.c0,
            dataset.c1,
            alpha0,
            alpha1,
        ),
        "predictions": predictions,
        "diagnostics": prediction_diagnostics(rows, predictions),
        "baselines": baseline_registry(rows),
        "provenance": {
            "packet": packet_path.name,
            "packetSha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "cacheFingerprint": fingerprint,
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {output_json} ({output_json.stat().st_size / 1_000_000:.2f} MB)", flush=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = parse_int_tuple(args.seeds)
    if not seeds:
        raise ValueError("At least one OOF seed is required")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    write_bundle(args.packet, args.output_json, args.cache_dir, seeds, args.workers)


if __name__ == "__main__":
    main()
