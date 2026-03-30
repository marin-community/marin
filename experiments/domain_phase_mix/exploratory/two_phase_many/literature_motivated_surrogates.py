# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.11",
# ]
# ///
"""Literature-motivated surrogates for phase-wise data-mixing.

This script implements four structured families motivated by recent mixture-law
and curriculum papers:

1. PBPM  : Power-Law Bucketed Phase Moments
2. SJPL  : Saturating Joint Phase Law
3. TBPTL : Thresholded Bucket Phase-Transition Law
4. TSJL  : Thresholded Saturating Joint Law

The implementation is designed for:
- the attached 2-phase, 39-domain MMLU packet
- the attached 3-phase Starcoder packet

It can also merge in precomputed baseline tables when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HUBER_DELTA = 0.02


@dataclass(frozen=True)
class DatasetSpec:
    weights: np.ndarray  # (R, N, M)
    y: np.ndarray  # (R,)
    phase_names: list[str]
    domain_names: list[str]

    @property
    def R(self) -> int:
        return int(self.weights.shape[0])

    @property
    def N(self) -> int:
        return int(self.weights.shape[1])

    @property
    def M(self) -> int:
        return int(self.weights.shape[2])


@dataclass(frozen=True)
class GroupSpec:
    names: list[str]
    matrix: np.ndarray  # (M, G)


def load_generic_phase_dataset(
    csv_path: str | Path,
    *,
    target_column: str,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, DatasetSpec]:
    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r}")
    if dropna_target:
        frame = frame.dropna(subset=[target_column]).reset_index(drop=True)

    phase_names = sorted(
        {
            f"{parts[0]}_{parts[1]}"
            for column in frame.columns
            if column.startswith("phase_")
            for parts in [column.split("_", 2)]
        }
    )
    if not phase_names:
        raise ValueError("No phase_* columns found")
    first_phase = phase_names[0]
    domain_names = [
        column.removeprefix(f"{first_phase}_") for column in frame.columns if column.startswith(f"{first_phase}_")
    ]
    if not domain_names:
        raise ValueError("No domain names found")

    weights = np.zeros((len(frame), len(phase_names), len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(phase_names):
        for domain_idx, domain_name in enumerate(domain_names):
            column = f"{phase_name}_{domain_name}"
            if column not in frame.columns:
                raise ValueError(f"Missing expected column {column!r}")
            weights[:, phase_idx, domain_idx] = frame[column].to_numpy(dtype=float)

    spec = DatasetSpec(
        weights=weights,
        y=frame[target_column].to_numpy(dtype=float),
        phase_names=phase_names,
        domain_names=domain_names,
    )
    return frame, spec


def huber_loss(residuals: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    abs_res = np.abs(residuals)
    return np.where(
        abs_res <= delta,
        0.5 * residuals * residuals,
        delta * (abs_res - 0.5 * delta),
    )


def compute_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    residuals = y_hat - y
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    chosen_idx = int(np.argmax(y_hat) if higher_is_better else np.argmin(y_hat))
    best_idx = int(np.argmax(y) if higher_is_better else np.argmin(y))
    regret = float(y[best_idx] - y[chosen_idx]) if higher_is_better else float(y[chosen_idx] - y[best_idx])
    return {
        "r2": float(1.0 - sse / sst),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "spearman": float(spearmanr(y, y_hat).statistic),
        "huber_loss_mean_delta_0p02": float(np.mean(huber_loss(residuals))),
        "regret_at_1": regret,
        "chosen_idx": chosen_idx,
        "best_idx": best_idx,
    }


def kfold_indices(n_rows: int, *, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows, dtype=int)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    return [(np.setdiff1d(indices, test_idx, assume_unique=True), test_idx) for test_idx in folds]


def ridge_fit_predict(
    features: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    features_new: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(features.shape[0], dtype=float), features])
    penalty = np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + float(lam) * penalty, x.T @ y)
    if features_new is None:
        features_new = features
    x_new = np.column_stack([np.ones(len(features_new), dtype=float), np.asarray(features_new, dtype=float)])
    return beta, x_new @ beta


def cv_metric(
    features: np.ndarray,
    y: np.ndarray,
    *,
    higher_is_better: bool,
    lam_grid: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    seed: int = 0,
) -> tuple[dict[str, float | int], float]:
    folds = kfold_indices(len(y), seed=seed)
    best_metrics: dict[str, float | int] | None = None
    best_lam = float(lam_grid[0])
    for lam in lam_grid:
        y_hat = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in folds:
            beta, _ = ridge_fit_predict(features[train_idx], y[train_idx], lam=float(lam))
            x_test = np.column_stack([np.ones(len(test_idx), dtype=float), features[test_idx]])
            y_hat[test_idx] = x_test @ beta
        fold_metrics = compute_metrics(y, y_hat, higher_is_better=higher_is_better)
        if best_metrics is None or float(fold_metrics["r2"]) > float(best_metrics["r2"]):
            best_metrics = fold_metrics
            best_lam = float(lam)
    assert best_metrics is not None
    return best_metrics, best_lam


def phase_basis(n_phases: int) -> np.ndarray:
    if n_phases < 1:
        raise ValueError("n_phases must be >= 1")
    t = np.linspace(-1.0, 1.0, n_phases)
    vandermonde = np.vstack([t**k for k in range(n_phases)]).T
    basis = np.zeros_like(vandermonde, dtype=float)
    for col_idx in range(n_phases):
        vec = vandermonde[:, col_idx].copy()
        for prev_idx in range(col_idx):
            vec -= basis[:, prev_idx] * float(np.dot(basis[:, prev_idx], vec))
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            raise RuntimeError("Degenerate phase basis")
        basis[:, col_idx] = vec / norm
    if basis[:, 0].sum() < 0.0:
        basis[:, 0] *= -1.0
    if n_phases > 1 and basis[-1, 1] < basis[0, 1]:
        basis[:, 1] *= -1.0
    return basis


def build_group_spec(domain_names: list[str], scheme: str) -> GroupSpec:
    if scheme == "identity":
        return GroupSpec(names=list(domain_names), matrix=np.eye(len(domain_names), dtype=float))

    if scheme != "source15":
        raise ValueError(f"Unknown grouping scheme {scheme!r}")

    def map_domain(domain_name: str) -> str:
        if domain_name == "dolma3_arxiv":
            return "arxiv"
        if domain_name.startswith("dolma3_cc/"):
            return "cc_high" if domain_name.endswith("_high") else "cc_low"
        if domain_name == "dolma3_finemath_3plus":
            return "finemath"
        if domain_name == "dolma3_stack_edu":
            return "stack_edu"
        if domain_name == "dolma3_wikipedia":
            return "wikipedia"
        if domain_name == "dolmino_common_crawl_hq":
            return "common_crawl_hq"
        if domain_name == "dolmino_olmocr_pdfs_hq":
            return "olmocr_pdfs_hq"
        if domain_name == "dolmino_stack_edu_fim":
            return "stack_edu_fim"
        if domain_name == "dolmino_stem_heavy_crawl":
            return "stem_heavy_crawl"
        if domain_name.startswith("dolmino_synth_"):
            return domain_name.removeprefix("dolmino_")
        raise KeyError(domain_name)

    mapped = [map_domain(name) for name in domain_names]
    unique_names = sorted(dict.fromkeys(mapped))
    matrix = np.zeros((len(domain_names), len(unique_names)), dtype=float)
    for domain_idx, group_name in enumerate(mapped):
        matrix[domain_idx, unique_names.index(group_name)] = 1.0
    return GroupSpec(names=unique_names, matrix=matrix)


def infer_default_grouping(domain_names: list[str]) -> str:
    if "dolma3_arxiv" in domain_names and "dolmino_synth_thinking" in domain_names:
        return "source15"
    return "identity"


def grouped_phase_arrays(spec: DatasetSpec, grouping: str) -> tuple[GroupSpec, np.ndarray, np.ndarray, np.ndarray]:
    grouping_use = infer_default_grouping(spec.domain_names) if grouping == "auto" else grouping
    group_spec = build_group_spec(spec.domain_names, grouping_use)
    grouped = np.einsum("rnm,mg->rng", spec.weights, group_spec.matrix)
    basis = phase_basis(spec.N)
    moments = np.einsum("rng,nl->rgl", grouped, basis)
    totals = grouped.sum(axis=1)
    higher = moments[:, :, 1:]
    return group_spec, totals, higher, moments


def signed_power(x: np.ndarray, rho: float) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** float(rho))


def hill(x: np.ndarray, kappa: float, p: float) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    kp = float(max(kappa, 1e-12)) ** float(p)
    xp = x ** float(p)
    return xp / (xp + kp + 1e-12)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def build_pbpm_features(
    spec: DatasetSpec,
    *,
    grouping: str = "auto",
    rho_t: float = 0.75,
    rho_m: float = 0.5,
    include_t2: bool = True,
    include_cross: bool = True,
    include_m2: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    group_spec, totals, higher, _ = grouped_phase_arrays(spec, grouping)
    cols: list[np.ndarray] = []
    total_pow = np.maximum(totals, 0.0) ** float(rho_t)
    for group_idx, _group_name in enumerate(group_spec.names):
        cols.append(total_pow[:, group_idx])
        if include_t2:
            cols.append(totals[:, group_idx] ** 2)
        for moment_idx in range(higher.shape[2]):
            moment = signed_power(higher[:, group_idx, moment_idx], rho_m)
            cols.append(moment)
            if include_m2:
                cols.append(higher[:, group_idx, moment_idx] ** 2)
            if include_cross:
                cols.append(total_pow[:, group_idx] * moment)
    features = np.column_stack(cols)
    info = {
        "family": "PBPM",
        "grouping": group_spec.names,
        "n_params": int(features.shape[1] + 1),
        "hyperparams": {
            "rho_t": rho_t,
            "rho_m": rho_m,
            "include_t2": include_t2,
            "include_cross": include_cross,
            "include_m2": include_m2,
        },
    }
    return features, info


def build_sjpl_features(
    spec: DatasetSpec,
    *,
    grouping: str = "auto",
    kappa: float = 0.1,
    p: float = 1.0,
    include_t2: bool = True,
    include_m2: bool = True,
    include_cross: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    group_spec, totals, higher, _ = grouped_phase_arrays(spec, grouping)
    cols: list[np.ndarray] = []
    total_sat = hill(totals, kappa, p)
    for group_idx, _group_name in enumerate(group_spec.names):
        cols.append(total_sat[:, group_idx])
        if include_t2:
            cols.append(totals[:, group_idx] ** 2)
        for moment_idx in range(higher.shape[2]):
            moment = higher[:, group_idx, moment_idx]
            cols.append(moment)
            if include_m2:
                cols.append(moment**2)
            if include_cross:
                cols.append(total_sat[:, group_idx] * moment)
    features = np.column_stack(cols)
    info = {
        "family": "SJPL",
        "grouping": group_spec.names,
        "n_params": int(features.shape[1] + 1),
        "hyperparams": {
            "kappa": kappa,
            "p": p,
            "include_t2": include_t2,
            "include_m2": include_m2,
            "include_cross": include_cross,
        },
    }
    return features, info


def build_tbptl_features(
    spec: DatasetSpec,
    *,
    grouping: str = "auto",
    tau_t: float = 0.2,
    sigma_t: float = 0.1,
    tau_m: float = 0.02,
    sigma_m: float = 0.1,
    include_raw: bool = True,
    include_cross: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    group_spec, totals, higher, _ = grouped_phase_arrays(spec, grouping)
    cols: list[np.ndarray] = []
    total_th = softplus((totals - float(tau_t)) / float(sigma_t)) * float(sigma_t)
    for group_idx, _group_name in enumerate(group_spec.names):
        if include_raw:
            cols.append(totals[:, group_idx])
        cols.append(total_th[:, group_idx])
        cols.append(totals[:, group_idx] ** 2)
        for moment_idx in range(higher.shape[2]):
            moment = higher[:, group_idx, moment_idx]
            moment_th = np.sign(moment) * softplus((np.abs(moment) - float(tau_m)) / float(sigma_m)) * float(sigma_m)
            cols.append(moment)
            cols.append(moment_th)
            cols.append(moment**2)
            if include_cross:
                cols.append(total_th[:, group_idx] * moment_th)
                cols.append(total_th[:, group_idx] * moment)
    features = np.column_stack(cols)
    info = {
        "family": "TBPTL",
        "grouping": group_spec.names,
        "n_params": int(features.shape[1] + 1),
        "hyperparams": {
            "tau_t": tau_t,
            "sigma_t": sigma_t,
            "tau_m": tau_m,
            "sigma_m": sigma_m,
            "include_raw": include_raw,
            "include_cross": include_cross,
        },
    }
    return features, info


def build_tsjl_features(
    spec: DatasetSpec,
    *,
    grouping: str = "auto",
    kappa: float = 0.1,
    p: float = 1.5,
    tau_t: float = 0.2,
    sigma_t: float = 0.1,
    tau_m: float = 0.05,
    sigma_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    group_spec, totals, higher, _ = grouped_phase_arrays(spec, grouping)
    cols: list[np.ndarray] = []
    total_sat = hill(totals, kappa, p)
    total_th = softplus((totals - float(tau_t)) / float(sigma_t)) * float(sigma_t)
    for group_idx, _group_name in enumerate(group_spec.names):
        cols.extend([total_sat[:, group_idx], total_th[:, group_idx], totals[:, group_idx] ** 2])
        for moment_idx in range(higher.shape[2]):
            moment = higher[:, group_idx, moment_idx]
            moment_th = np.sign(moment) * softplus((np.abs(moment) - float(tau_m)) / float(sigma_m)) * float(sigma_m)
            cols.extend(
                [
                    moment,
                    moment_th,
                    moment**2,
                    total_sat[:, group_idx] * moment,
                    total_th[:, group_idx] * moment_th,
                ]
            )
    features = np.column_stack(cols)
    info = {
        "family": "TSJL",
        "grouping": group_spec.names,
        "n_params": int(features.shape[1] + 1),
        "hyperparams": {
            "kappa": kappa,
            "p": p,
            "tau_t": tau_t,
            "sigma_t": sigma_t,
            "tau_m": tau_m,
            "sigma_m": sigma_m,
        },
    }
    return features, info


def fit_grid(
    spec: DatasetSpec,
    *,
    higher_is_better: bool,
    family_name: str,
    build_fn: Callable[..., tuple[np.ndarray, dict[str, object]]],
    grid: list[dict[str, object]],
    grouping: str,
    lam_grid: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> tuple[dict[str, object], dict[str, object]]:
    best_fit_record: dict[str, object] | None = None
    best_cv_record: dict[str, object] | None = None

    for hyperparams in grid:
        features, info = build_fn(spec, grouping=grouping, **hyperparams)
        fit_best_metrics: dict[str, float | int] | None = None
        fit_best_lam = float(lam_grid[0])
        fit_best_y_hat: np.ndarray | None = None
        for lam in lam_grid:
            _, y_hat = ridge_fit_predict(features, spec.y, lam=float(lam))
            fit_metrics = compute_metrics(spec.y, y_hat, higher_is_better=higher_is_better)
            if fit_best_metrics is None or float(fit_metrics["r2"]) > float(fit_best_metrics["r2"]):
                fit_best_metrics = fit_metrics
                fit_best_lam = float(lam)
                fit_best_y_hat = y_hat
        assert fit_best_metrics is not None and fit_best_y_hat is not None
        cv_metrics, cv_lam = cv_metric(features, spec.y, higher_is_better=higher_is_better, lam_grid=lam_grid)
        record = {
            "dataset_y": spec.y,
            "family_name": family_name,
            "features": features,
            "info": info,
            "hyperparams": hyperparams,
            "fit_metrics": fit_best_metrics,
            "fit_lam": fit_best_lam,
            "fit_y_hat": fit_best_y_hat,
            "cv_metrics": cv_metrics,
            "cv_lam": cv_lam,
        }
        if best_fit_record is None or float(fit_best_metrics["r2"]) > float(best_fit_record["fit_metrics"]["r2"]):
            best_fit_record = record
        if best_cv_record is None or float(cv_metrics["r2"]) > float(best_cv_record["cv_metrics"]["r2"]):
            best_cv_record = record

    assert best_fit_record is not None and best_cv_record is not None
    return best_fit_record, best_cv_record


def make_family_grids() -> dict[str, list[dict[str, object]]]:
    return {
        "PBPM": [
            {"rho_t": rho_t, "rho_m": rho_m, "include_t2": True, "include_cross": True, "include_m2": True}
            for rho_t in (0.3, 0.5, 0.75, 1.0)
            for rho_m in (0.5, 1.0, 1.5)
        ],
        "SJPL": [
            {"kappa": kappa, "p": p, "include_t2": True, "include_m2": True, "include_cross": True}
            for kappa in (0.1, 0.2, 0.33, 0.5, 0.75, 1.0)
            for p in (0.5, 1.0, 1.5, 2.0)
        ],
        "TBPTL": [
            {
                "tau_t": tau_t,
                "sigma_t": sigma_t,
                "tau_m": tau_m,
                "sigma_m": sigma_m,
                "include_raw": True,
                "include_cross": True,
            }
            for tau_t in (0.1, 0.2, 0.33, 0.5, 0.75, 1.0)
            for sigma_t in (0.05, 0.1)
            for tau_m in (0.02, 0.05, 0.1, 0.2)
            for sigma_m in (0.05, 0.1)
        ],
        "TSJL": [
            {
                "kappa": kappa,
                "p": p,
                "tau_t": tau_t,
                "sigma_t": sigma_t,
                "tau_m": tau_m,
                "sigma_m": sigma_m,
            }
            for kappa in (0.1, 0.2, 0.5)
            for p in (1.0, 1.5)
            for tau_t in (0.2, 0.5)
            for sigma_t in (0.05, 0.1)
            for tau_m in (0.05, 0.1, 0.2)
            for sigma_m in (0.05, 0.1)
        ],
    }


def record_to_row(dataset: str, model: str, record: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "n_params": int(record["info"]["n_params"]),
        "r2": float(record["fit_metrics"]["r2"]),
        "rmse": float(record["fit_metrics"]["rmse"]),
        "spearman": float(record["fit_metrics"]["spearman"]),
        "huber_loss_mean_delta_0p02": float(record["fit_metrics"]["huber_loss_mean_delta_0p02"]),
        "regret_at_1": float(record["fit_metrics"]["regret_at_1"]),
        "notes": json.dumps(
            {
                "fit_lam": float(record["fit_lam"]),
                "cv_lam": float(record["cv_lam"]),
                "hyperparams": record["hyperparams"],
            },
            sort_keys=True,
        ),
    }


def record_to_cv_row(dataset: str, model: str, record: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "n_params": int(record["info"]["n_params"]),
        "best_cv_lam": float(record["cv_lam"]),
        "r2": float(record["cv_metrics"]["r2"]),
        "rmse": float(record["cv_metrics"]["rmse"]),
        "spearman": float(record["cv_metrics"]["spearman"]),
        "huber_loss_mean_delta_0p02": float(record["cv_metrics"]["huber_loss_mean_delta_0p02"]),
        "regret_at_1": float(record["cv_metrics"]["regret_at_1"]),
        "notes": json.dumps({"hyperparams": record["hyperparams"]}, sort_keys=True),
    }


def benchmark(
    *,
    mmlu_csv: str | Path,
    starcoder_csv: str | Path,
    base_compare_csv: str | Path | None,
    base_cv_csv: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grids = make_family_grids()
    builders: dict[str, Callable[..., tuple[np.ndarray, dict[str, object]]]] = {
        "Power-Law Bucketed Phase Moments (PBPM)": build_pbpm_features,
        "Saturating Joint Phase Law (SJPL)": build_sjpl_features,
        "Thresholded Bucket Phase-Transition Law (TBPTL)": build_tbptl_features,
        "Thresholded Saturating Joint Law (TSJL)": build_tsjl_features,
    }

    compare_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []

    if base_compare_csv is not None and Path(base_compare_csv).exists():
        compare_frames.append(pd.read_csv(base_compare_csv))
    if base_cv_csv is not None and Path(base_cv_csv).exists():
        base_cv = pd.read_csv(base_cv_csv)
        if "n_params" not in base_cv.columns:
            base_cv["n_params"] = np.nan
        cv_frames.append(base_cv)

    datasets = [
        ("mmlu_two_phase_39d", mmlu_csv, "choice_logprob_norm_mean", True, "source15"),
        ("starcoder_three_phase_2d", starcoder_csv, "eval/paloma/dolma_100_programing_languages/bpb", False, "identity"),
    ]

    for dataset_name, csv_path, target_column, higher_is_better, grouping in datasets:
        _, spec = load_generic_phase_dataset(csv_path, target_column=target_column)
        for model_name, build_fn in builders.items():
            family_key = model_name.split("(")[-1].removesuffix(")")
            best_fit, best_cv = fit_grid(
                spec,
                higher_is_better=higher_is_better,
                family_name=model_name,
                build_fn=build_fn,
                grid=grids[family_key],
                grouping=grouping,
            )
            compare_frames.append(pd.DataFrame([record_to_row(dataset_name, model_name, best_fit)]))
            cv_frames.append(pd.DataFrame([record_to_cv_row(dataset_name, model_name, best_cv)]))

    compare = pd.concat(compare_frames, ignore_index=True, sort=False)
    cv = pd.concat(cv_frames, ignore_index=True, sort=False)

    if "dataset" in compare.columns and "model" in compare.columns:
        param_lookup = {(row.dataset, row.model): row.n_params for row in compare.itertuples()}
        cv["n_params"] = cv.apply(lambda row: param_lookup.get((row["dataset"], row["model"]), row["n_params"]), axis=1)

    return compare, cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmlu-csv",
        type=Path,
        default=Path("qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_candidate_summary.csv"),
    )
    parser.add_argument("--starcoder-csv", type=Path, default=Path("three_phase_starcoder.csv"))
    parser.add_argument("--base-compare-csv", type=Path, default=Path("cross_phase_bucket_comparison.csv"))
    parser.add_argument("--base-cv-csv", type=Path, default=Path("cross_phase_bucket_cv_summary.csv"))
    parser.add_argument("--out-compare-csv", type=Path, default=Path("literature_motivated_comparison.csv"))
    parser.add_argument("--out-cv-csv", type=Path, default=Path("literature_motivated_cv.csv"))
    args = parser.parse_args()

    compare, cv = benchmark(
        mmlu_csv=args.mmlu_csv,
        starcoder_csv=args.starcoder_csv,
        base_compare_csv=args.base_compare_csv,
        base_cv_csv=args.base_cv_csv,
    )
    compare.to_csv(args.out_compare_csv, index=False)
    cv.to_csv(args.out_cv_csv, index=False)
    print(compare.to_string(index=False))
    print()
    print(cv.to_string(index=False))


if __name__ == "__main__":
    main()
