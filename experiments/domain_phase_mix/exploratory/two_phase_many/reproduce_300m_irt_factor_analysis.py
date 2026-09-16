# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Reproduce the 300M BPB factor-analysis / IRT-style collaborator study.

This mirrors the collaborator gist's non-negative Gaussian factor model:
negate BPB so higher is better, z-score each item, choose k by Horn parallel
analysis, and anchor item residual variances to same-mixture noise estimates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_MATRIX_DIR = ROOT / "metric_registry" / "raw_metric_matrix_300m"
DEFAULT_OUTPUT_DIR = ROOT / "reference_outputs" / "300m_irt_factor_analysis_20260501"

MMLU_KEEP = {
    "lm_eval/mmlu_5shot/bpb",
    "lm_eval/mmlu_sl_verb_5shot/bpb",
}
AGG_DROP = {
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
IRT_PREFIXES = (
    "eval/paloma/",
    "eval/uncheatable_eval/",
    "lm_eval/",
    "mcq_smooth/",
    "teacher_forced/",
)


@dataclass(frozen=True)
class FactorResult:
    name: str
    task_cols: list[str]
    k_horn: int
    k_mean: int
    eigenvalues: np.ndarray
    random_p95: np.ndarray
    random_mean: np.ndarray
    loadings: np.ndarray
    psi: np.ndarray
    theta: np.ndarray
    communality: np.ndarray
    noise_share: np.ndarray
    h2_ceiling: np.ndarray


def keep_irt_item(column: str) -> bool:
    if not column.endswith("/bpb"):
        return False
    if column in AGG_DROP or column in TASK_DROP:
        return False
    if not column.startswith(IRT_PREFIXES):
        return False
    if column.startswith("lm_eval/mmlu_") and column not in MMLU_KEEP:
        return False
    return True


def load_signal_rows() -> pd.DataFrame:
    raw = pd.read_csv(RAW_MATRIX_DIR / "raw_metric_matrix_300m.csv", low_memory=False)
    if "status" not in raw.columns:
        raise ValueError("raw matrix is missing status")
    completed = raw[raw["status"] == "completed"].copy()
    if len(completed) != 242:
        raise ValueError(f"expected 242 completed signal rows, got {len(completed)}")
    return completed


def zscore_signed_bpb(signal: pd.DataFrame, task_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bpb = signal[task_cols].to_numpy(dtype=np.float64)
    if np.isnan(x_bpb).any():
        missing = signal[task_cols].isna().sum()
        missing_cols = missing[missing > 0].index.tolist()
        raise ValueError(f"signal rows have missing IRT item values: {missing_cols}")
    signed = -x_bpb
    means = signed.mean(axis=0)
    stds = signed.std(axis=0)
    if np.any(stds <= 1e-12):
        bad = [task_cols[i] for i in np.where(stds <= 1e-12)[0]]
        raise ValueError(f"constant IRT item columns: {bad}")
    z = (signed - means) / stds
    return z, means, stds


def horn_parallel_analysis(
    z: np.ndarray, seed: int = 42, draws: int = 500
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    n, p = z.shape
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(z.T)))[::-1]
    rng = np.random.default_rng(seed)
    random = np.empty((draws, p), dtype=np.float64)
    for i in range(draws):
        zr = rng.standard_normal((n, p))
        zr = (zr - zr.mean(axis=0)) / zr.std(axis=0)
        random[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(zr.T)))[::-1]
    p95 = np.percentile(random, 95, axis=0)
    mean = random.mean(axis=0)
    return int(np.sum(real > p95)), int(np.sum(real > mean)), real, p95, mean


def noise_share_for_items(
    signal: pd.DataFrame,
    noise: pd.DataFrame,
    task_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = [c for c in task_cols if c in noise.columns]
    has = np.array([c in noise.columns for c in task_cols], dtype=bool)
    noise_share = np.full(len(task_cols), np.nan, dtype=np.float64)
    sweep_sd = np.full(len(task_cols), np.nan, dtype=np.float64)
    noise_sd = np.full(len(task_cols), np.nan, dtype=np.float64)
    if present:
        noise_x = noise[present].to_numpy(dtype=np.float64)
        sweep_x = signal[present].to_numpy(dtype=np.float64)
        if np.isnan(noise_x).any():
            missing = noise[present].isna().sum()
            missing_cols = missing[missing > 0].index.tolist()
            raise ValueError(f"noise rows have missing item values: {missing_cols}")
        ns = noise_x.std(axis=0, ddof=1)
        ss = sweep_x.std(axis=0, ddof=1)
        noise_sd[has] = ns
        sweep_sd[has] = ss
        noise_share[has] = (ns / ss) ** 2
    h2_ceiling = np.where(np.isnan(noise_share), np.nan, np.clip(1.0 - noise_share, 0.0, 1.0))
    return noise_share, h2_ceiling, noise_sd


def fit_nonnegative_factor_model(
    z: np.ndarray,
    k: int,
    noise_share: np.ndarray,
    seed: int = 0,
    max_iter: int = 5000,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, p = z.shape
    psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    psi_fixed = ~np.isnan(psi_anchor)
    rng = np.random.default_rng(seed)
    loadings = np.abs(rng.normal(scale=0.1, size=(p, k)))
    psi = np.where(psi_fixed, psi_anchor, 1.0)
    for _ in range(max_iter):
        loadings_psi_inv = loadings / psi[:, None]
        v_post = np.linalg.inv(np.eye(k) + loadings.T @ loadings_psi_inv)
        theta_hat = z @ loadings_psi_inv @ v_post
        s_thth = n * v_post + theta_hat.T @ theta_hat
        zt_th = z.T @ theta_hat
        next_loadings = zt_th @ np.linalg.inv(s_thth)
        next_loadings = np.clip(next_loadings, 0.0, None)
        psi_free = (
            (z**2).mean(axis=0)
            - 2 * (zt_th * next_loadings).sum(axis=1) / n
            + ((next_loadings @ s_thth) * next_loadings).sum(axis=1) / n
        )
        psi_free = np.clip(psi_free, 1e-6, None)
        next_psi = np.where(psi_fixed, psi_anchor, psi_free)
        max_delta = float(np.max(np.abs(next_loadings - loadings)))
        loadings, psi = next_loadings, next_psi
        if max_delta < tolerance:
            break
    order = np.argsort(-(loadings**2).sum(axis=0))
    loadings = loadings[:, order]
    loadings_psi_inv = loadings / psi[:, None]
    v_post = np.linalg.inv(np.eye(k) + loadings.T @ loadings_psi_inv)
    theta = z @ loadings_psi_inv @ v_post
    communality = (loadings**2).sum(axis=1) / ((loadings**2).sum(axis=1) + psi)
    return loadings, psi, theta, communality


def run_model(name: str, signal: pd.DataFrame, noise: pd.DataFrame, task_cols: list[str]) -> FactorResult:
    z, _, _ = zscore_signed_bpb(signal, task_cols)
    k_horn, k_mean, eigenvalues, random_p95, random_mean = horn_parallel_analysis(z)
    if k_horn <= 0:
        raise ValueError(f"{name}: Horn selected no factors")
    noise_share, h2_ceiling, _ = noise_share_for_items(signal, noise, task_cols)
    loadings, psi, theta, communality = fit_nonnegative_factor_model(z, k_horn, noise_share)
    return FactorResult(
        name=name,
        task_cols=task_cols,
        k_horn=k_horn,
        k_mean=k_mean,
        eigenvalues=eigenvalues,
        random_p95=random_p95,
        random_mean=random_mean,
        loadings=loadings,
        psi=psi,
        theta=theta,
        communality=communality,
        noise_share=noise_share,
        h2_ceiling=h2_ceiling,
    )


def write_result(result: FactorResult, signal: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    prefix = result.name
    loadings_rows = []
    for i, metric in enumerate(result.task_cols):
        row = {
            "metric": metric,
            "communality": result.communality[i],
            "h2_ceiling": result.h2_ceiling[i],
            "noise_share": result.noise_share[i],
            "psi": result.psi[i],
            "dominant_factor": int(np.argmax(result.loadings[i]) + 1),
            "max_loading": float(result.loadings[i].max()),
        }
        for k in range(result.loadings.shape[1]):
            row[f"loading_factor_{k + 1}"] = result.loadings[i, k]
        loadings_rows.append(row)
    loadings = pd.DataFrame(loadings_rows)
    loadings.to_csv(output_dir / f"{prefix}_loadings.csv", index=False)

    leaderboard = signal[
        [
            "run_name",
            "is_qsplit240_core",
            "eval/paloma/macro_loss",
            "eval/uncheatable_eval/macro_loss",
        ]
    ].copy()
    for k in range(result.theta.shape[1]):
        leaderboard[f"theta_{k + 1}"] = result.theta[:, k]
    leaderboard["aggregate_mean_theta"] = result.theta.mean(axis=1)
    leaderboard = leaderboard.sort_values("aggregate_mean_theta", ascending=False)
    leaderboard.to_csv(output_dir / f"{prefix}_leaderboard.csv", index=False)

    horn = pd.DataFrame(
        {
            "rank": np.arange(1, len(result.eigenvalues) + 1),
            "real_eigenvalue": result.eigenvalues,
            "random_p95": result.random_p95,
            "random_mean": result.random_mean,
        }
    )
    horn.to_csv(output_dir / f"{prefix}_horn_parallel.csv", index=False)
    return leaderboard


def compare_leaderboards(leaderboards: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    merged = None
    for name, lb in leaderboards.items():
        slim = lb[["run_name", "aggregate_mean_theta"]].copy()
        slim[f"{name}_rank"] = np.arange(1, len(slim) + 1)
        slim = slim.rename(columns={"aggregate_mean_theta": f"{name}_aggregate_mean_theta"})
        merged = slim if merged is None else merged.merge(slim, on="run_name", how="outer")
    if merged is None:
        raise ValueError("no leaderboards to compare")
    score_cols = [c for c in merged.columns if c.endswith("_aggregate_mean_theta")]
    ordered = merged.sort_values(score_cols[0], ascending=False)
    ordered.to_csv(output_dir / "leaderboard_comparison.csv", index=False)

    summary_rows = []
    names = list(leaderboards)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            ranks = merged[[f"{left}_rank", f"{right}_rank"]].dropna()
            # The columns are already ranks; Spearman is just Pearson on ranks.
            rank_corr = float(ranks[f"{left}_rank"].corr(ranks[f"{right}_rank"]))
            top10_left = set(merged.nsmallest(10, f"{left}_rank")["run_name"])
            top10_right = set(merged.nsmallest(10, f"{right}_rank")["run_name"])
            top20_left = set(merged.nsmallest(20, f"{left}_rank")["run_name"])
            top20_right = set(merged.nsmallest(20, f"{right}_rank")["run_name"])
            summary_rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_rank_corr": rank_corr,
                    "top10_overlap": len(top10_left & top10_right),
                    "top20_overlap": len(top20_left & top20_right),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "leaderboard_comparison_summary.csv", index=False)
    return summary


def write_factor_noise_snr(
    results: dict[str, FactorResult],
    signal: pd.DataFrame,
    noise_by_model: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        z_signal, means, stds = zscore_signed_bpb(signal, result.task_cols)
        noise = noise_by_model[name]
        z_noise = np.zeros((len(noise), len(result.task_cols)), dtype=np.float64)
        present = [c for c in result.task_cols if c in noise.columns]
        if present:
            present_mask = np.array([c in present for c in result.task_cols], dtype=bool)
            noise_signed = -noise[present].to_numpy(dtype=np.float64)
            z_noise[:, present_mask] = (noise_signed - means[present_mask]) / stds[present_mask]
        projection = (result.loadings / result.psi[:, None]) @ np.linalg.inv(
            np.eye(result.loadings.shape[1]) + result.loadings.T @ (result.loadings / result.psi[:, None])
        )
        theta_signal = z_signal @ projection
        theta_noise = z_noise @ projection
        for k in range(result.loadings.shape[1]):
            signal_var = float(theta_signal[:, k].var(ddof=1))
            noise_var = float(theta_noise[:, k].var(ddof=1))
            rows.append(
                {
                    "model": name,
                    "factor": f"theta_{k + 1}",
                    "signal_var": signal_var,
                    "noise_var": noise_var,
                    "snr": (signal_var - noise_var) / noise_var,
                    "noise_share": noise_var / signal_var,
                }
            )
        signal_aggregate = theta_signal.mean(axis=1)
        noise_aggregate = theta_noise.mean(axis=1)
        signal_var = float(signal_aggregate.var(ddof=1))
        noise_var = float(noise_aggregate.var(ddof=1))
        rows.append(
            {
                "model": name,
                "factor": "aggregate_mean_theta",
                "signal_var": signal_var,
                "noise_var": noise_var,
                "snr": (signal_var - noise_var) / noise_var,
                "noise_share": noise_var / signal_var,
            }
        )
    factor_snr = pd.DataFrame(rows)
    factor_snr.to_csv(output_dir / "factor_noise_snr.csv", index=False)
    return factor_snr


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = load_signal_rows()
    legacy_noise = pd.read_csv(RAW_MATRIX_DIR / "noise_baseline_run00097_300m.csv", low_memory=False)
    fixed_noise = pd.read_csv(RAW_MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv", low_memory=False)
    variable_noise = pd.read_csv(RAW_MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv", low_memory=False)

    base_cols = [c for c in signal.columns if keep_irt_item(c)]
    valid_cols = []
    for c in base_cols:
        if signal[c].notna().all() and signal[c].std(ddof=0) > 1e-12:
            valid_cols.append(c)
    fixed_available = [c for c in valid_cols if c in fixed_noise.columns and fixed_noise[c].notna().all()]
    variable_available = [c for c in valid_cols if c in variable_noise.columns and variable_noise[c].notna().all()]
    common_available = [c for c in valid_cols if c in fixed_available and c in variable_available]
    legacy_available = [c for c in valid_cols if c in legacy_noise.columns and legacy_noise[c].notna().all()]

    coverage = pd.DataFrame(
        {
            "metric": valid_cols,
            "in_legacy_noise": [c in legacy_available for c in valid_cols],
            "in_fixed_noise": [c in fixed_available for c in valid_cols],
            "in_variable_noise": [c in variable_available for c in valid_cols],
            "in_common_noise": [c in common_available for c in valid_cols],
        }
    )
    coverage.to_csv(output_dir / "irt_item_noise_coverage.csv", index=False)

    fixed_common_name = f"fixed_common{len(common_available)}"
    variable_common_name = f"variable_common{len(common_available)}"
    results = {
        "legacy_alias_all43": run_model("legacy_alias_all43", signal, legacy_noise, valid_cols),
        "fixed_all43": run_model("fixed_all43", signal, fixed_noise, fixed_available),
        fixed_common_name: run_model(fixed_common_name, signal, fixed_noise, common_available),
        variable_common_name: run_model(variable_common_name, signal, variable_noise, common_available),
    }
    leaderboards = {name: write_result(result, signal, output_dir) for name, result in results.items()}
    comparison = compare_leaderboards(leaderboards, output_dir)
    factor_snr = write_factor_noise_snr(
        results,
        signal,
        {
            "legacy_alias_all43": legacy_noise,
            "fixed_all43": fixed_noise,
            fixed_common_name: fixed_noise,
            variable_common_name: variable_noise,
        },
        output_dir,
    )

    summary = {
        "rows": {
            "signal": len(signal),
            "fixed_noise": len(fixed_noise),
            "variable_noise": len(variable_noise),
        },
        "item_counts": {
            "valid_irt_items": len(valid_cols),
            "legacy_noise_available": len(legacy_available),
            "fixed_noise_available": len(fixed_available),
            "variable_noise_available": len(variable_available),
            "common_noise_available": len(common_available),
            "missing_legacy": [c for c in valid_cols if c not in legacy_available],
            "missing_fixed": [c for c in valid_cols if c not in fixed_available],
            "missing_variable": [c for c in valid_cols if c not in variable_available],
        },
        "models": {
            name: {
                "k_horn": int(result.k_horn),
                "k_mean": int(result.k_mean),
                "items": len(result.task_cols),
                "mean_noise_share": float(np.nanmean(result.noise_share)),
                "median_noise_share": float(np.nanmedian(result.noise_share)),
                "mean_h2_ceiling": float(np.nanmean(result.h2_ceiling)),
                "median_h2_ceiling": float(np.nanmedian(result.h2_ceiling)),
                "mean_communality": float(np.nanmean(result.communality)),
                "median_communality": float(np.nanmedian(result.communality)),
                "top5_runs": leaderboards[name]["run_name"].head(5).tolist(),
            }
            for name, result in results.items()
        },
        "leaderboard_comparison": comparison.to_dict(orient="records"),
        "factor_noise_snr": factor_snr.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
