# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5416 aggregate projection for data-mixture task metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Issue5416Projection:
    """Fitted projection from task metrics to the issue #5416 aggregate coordinate."""

    task_columns: tuple[str, ...]
    task_signs: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    loadings: tuple[tuple[float, ...], ...]
    uniquenesses: tuple[float, ...]
    factor_count: int
    projection_vector: tuple[float, ...]
    horn_real_eigenvalues: tuple[float, ...]
    horn_random_p95: tuple[float, ...]
    item_noise_share: tuple[float | None, ...]


MMLU_KEEP = {"lm_eval/mmlu_sl_verb_5shot/bpb"}
AGGREGATE_DROP = {
    "eval/bpb",
    "eval/macro_bpb",
    "eval/paloma/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/bpb",
    "eval/uncheatable_eval/macro_bpb",
}
TASK_DROP = {
    "teacher_forced/gsm8k_5shot_answer_hash/bpb",
    "mcq_smooth/sciq_5shot/bpb",
}


def select_issue5416_task_columns(columns: list[str]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Select and orient metric columns using the latest issue #5416 gist rules."""
    column_set = set(columns)
    task_columns: list[str] = []
    task_signs: list[float] = []
    for column in columns:
        if not column.endswith("/bpb"):
            continue
        if column in AGGREGATE_DROP or column in TASK_DROP:
            continue
        if not column.startswith(("eval/uncheatable_eval/", "lm_eval/", "mcq_smooth/", "teacher_forced/")):
            continue
        if column.startswith("lm_eval/mmlu_") and column not in MMLU_KEEP:
            continue
        base = column.removesuffix("/bpb")
        if base.startswith(("lm_eval/", "mcq_smooth/")):
            choice_logprob_column = f"{base}/choice_logprob"
            if choice_logprob_column in column_set:
                task_columns.append(choice_logprob_column)
                task_signs.append(1.0)
                continue
        task_columns.append(column)
        task_signs.append(-1.0)
    return tuple(task_columns), tuple(task_signs)


def _standardize_columns(frame: pd.DataFrame, task_columns: tuple[str, ...], task_signs: np.ndarray) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    missing = [column for column in task_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing aggregate task columns: {missing}")
    x_signed = frame.loc[:, list(task_columns)].to_numpy(dtype=np.float64) * task_signs[None, :]
    means = x_signed.mean(axis=0)
    stds = x_signed.std(axis=0)
    if np.any(stds <= 1e-12):
        bad = [task_columns[index] for index in np.flatnonzero(stds <= 1e-12)]
        raise ValueError(f"Zero-variance aggregate task columns: {bad}")
    return (x_signed - means) / stds, means, stds


def _horn_factor_count(z: np.ndarray, *, seed: int, n_mc: int) -> tuple[int, np.ndarray, np.ndarray]:
    n, p = z.shape
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(z.T)))[::-1]
    rng = np.random.default_rng(seed)
    random_eigs = np.empty((n_mc, p), dtype=np.float64)
    for index in range(n_mc):
        z_random = rng.standard_normal((n, p))
        z_random = (z_random - z_random.mean(axis=0)) / z_random.std(axis=0)
        random_eigs[index] = np.sort(np.linalg.eigvalsh(np.corrcoef(z_random.T)))[::-1]
    random_p95 = np.percentile(random_eigs, 95, axis=0)
    factor_count = max(1, int(np.sum(real > random_p95)))
    return factor_count, real, random_p95


def _nonnegative_factor_projection(
    z: np.ndarray,
    noise_share: np.ndarray,
    *,
    factor_count: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = z.shape
    psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    psi_fixed = ~np.isnan(psi_anchor)
    rng = np.random.default_rng(seed)
    loadings = np.abs(rng.normal(scale=0.1, size=(p, factor_count)))
    uniquenesses = np.where(psi_fixed, psi_anchor, 1.0)
    for _ in range(max_iter):
        weighted_loadings = loadings / uniquenesses[:, None]
        posterior_cov = np.linalg.inv(np.eye(factor_count) + loadings.T @ weighted_loadings)
        theta = z @ weighted_loadings @ posterior_cov
        sufficient = n * posterior_cov + theta.T @ theta
        z_theta = z.T @ theta
        next_loadings = z_theta @ np.linalg.inv(sufficient)
        next_loadings = np.clip(next_loadings, 0.0, None)
        free_psi = (
            (z**2).mean(axis=0)
            - 2 * (z_theta * next_loadings).sum(axis=1) / n
            + ((next_loadings @ sufficient) * next_loadings).sum(axis=1) / n
        )
        free_psi = np.clip(free_psi, 1e-6, None)
        next_uniquenesses = np.where(psi_fixed, psi_anchor, free_psi)
        if np.max(np.abs(next_loadings - loadings)) < tolerance:
            loadings = next_loadings
            uniquenesses = next_uniquenesses
            break
        loadings = next_loadings
        uniquenesses = next_uniquenesses
    order = np.argsort(-(loadings**2).sum(axis=0))
    loadings = loadings[:, order]
    weighted_loadings = loadings / uniquenesses[:, None]
    posterior_cov = np.linalg.inv(np.eye(factor_count) + loadings.T @ weighted_loadings)
    projection_vector = (weighted_loadings @ posterior_cov).mean(axis=1)
    return loadings, uniquenesses, projection_vector


def fit_issue5416_projection(
    *,
    signal_frame: pd.DataFrame,
    noise_frame: pd.DataFrame,
    seed: int = 42,
    horn_n_mc: int = 500,
    max_iter: int = 5000,
) -> Issue5416Projection:
    """Fit the fixed aggregate projection from current 300M signal and noise rows."""
    task_columns, task_signs_tuple = select_issue5416_task_columns(list(signal_frame.columns))
    if not task_columns:
        raise ValueError("No issue #5416 aggregate task columns were selected")
    task_signs = np.asarray(task_signs_tuple, dtype=np.float64)
    z_signal, means, stds = _standardize_columns(signal_frame, task_columns, task_signs)

    present_in_noise = np.asarray([column in noise_frame.columns for column in task_columns])
    noise_share = np.full(len(task_columns), np.nan)
    if present_in_noise.any():
        present_columns = [column for column, present in zip(task_columns, present_in_noise, strict=True) if present]
        present_signs = task_signs[present_in_noise]
        noise_values = noise_frame.loc[:, present_columns].to_numpy(dtype=np.float64) * present_signs[None, :]
        noise_std = noise_values.std(axis=0, ddof=1)
        signal_std = (signal_frame.loc[:, present_columns].to_numpy(dtype=np.float64) * present_signs[None, :]).std(
            axis=0, ddof=1
        )
        noise_share[present_in_noise] = (noise_std / signal_std) ** 2

    factor_count, real_eigenvalues, random_p95 = _horn_factor_count(z_signal, seed=seed, n_mc=horn_n_mc)
    loadings, uniquenesses, projection_vector = _nonnegative_factor_projection(
        z_signal,
        noise_share,
        factor_count=factor_count,
        seed=0,
        max_iter=max_iter,
        tolerance=1e-7,
    )
    return Issue5416Projection(
        task_columns=task_columns,
        task_signs=task_signs_tuple,
        means=tuple(float(value) for value in means),
        stds=tuple(float(value) for value in stds),
        loadings=tuple(tuple(float(value) for value in row) for row in loadings),
        uniquenesses=tuple(float(value) for value in uniquenesses),
        factor_count=factor_count,
        projection_vector=tuple(float(value) for value in projection_vector),
        horn_real_eigenvalues=tuple(float(value) for value in real_eigenvalues),
        horn_random_p95=tuple(float(value) for value in random_p95),
        item_noise_share=tuple(None if np.isnan(value) else float(value) for value in noise_share),
    )


def score_issue5416_aggregate(
    frame: pd.DataFrame,
    projection: Issue5416Projection,
    *,
    fail_missing: bool,
) -> pd.Series:
    """Score rows with a fitted issue #5416 projection."""
    missing = [column for column in projection.task_columns if column not in frame.columns]
    if missing and fail_missing:
        raise ValueError(f"Missing aggregate task columns for scoring: {missing}")
    complete_mask = frame.loc[:, list(projection.task_columns)].notna().all(axis=1)
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    if not complete_mask.any():
        return scores
    values = frame.loc[complete_mask, list(projection.task_columns)].to_numpy(dtype=np.float64)
    signs = np.asarray(projection.task_signs, dtype=np.float64)
    means = np.asarray(projection.means, dtype=np.float64)
    stds = np.asarray(projection.stds, dtype=np.float64)
    vector = np.asarray(projection.projection_vector, dtype=np.float64)
    z = (values * signs[None, :] - means[None, :]) / stds[None, :]
    scores.loc[complete_mask] = z @ vector
    return scores


def write_issue5416_projection(projection: Issue5416Projection, path: Path) -> None:
    """Write projection metadata as deterministic JSON."""
    payload: dict[str, Any] = asdict(projection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
