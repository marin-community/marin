#!/usr/bin/env python3
"""Fit a compact MCT-LRQ-style joint mixture/scale law from packet CSVs.

This is a self-contained reference implementation. It is intentionally smaller
than the internal research scripts, but preserves the important structure:

    L(w,N,D) = E_LRQ(w)
             + A ((N/N0)^(-alpha) - 1)
             + B_fam(w) ((D/D0)^(-beta) - 1)
             + C ((N/N0)^(-gamma) (D/D0)^(-delta) - 1)

At N=N0 and D=D0, this reduces exactly to the learned mixture regression
E_LRQ(w). For a fixed mixture w, it reduces to a low-dimensional scaling law.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_runs.csv"
DEFAULT_TARGET = "eval/uncheatable_eval/bpb"


@dataclass(frozen=True)
class ScaleConstants:
    n0: float = 102_648_576.0
    d0: float = 5_999_951_872.0
    alpha: float = 0.154791
    beta: float = 0.146425
    gamma: float = 0.014295
    delta: float = 1.063376


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(c.removeprefix("phase_0_") for c in columns if c.startswith("phase_0_"))
    if not domains:
        raise ValueError("No phase_0_* columns found")
    missing = [domain for domain in domains if f"phase_1_{domain}" not in columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for {len(missing)} domains")
    return domains


def normalized_phase_arrays(df: pd.DataFrame, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w0 = df[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = df[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    for name, weights in (("phase_0", w0), ("phase_1", w1)):
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError(f"{name} contains non-positive row mass")
        weights /= row_sums[:, None]
    return w0, w1


def domain_family(domain: str) -> str:
    low = domain.lower()
    if "synth_math" in low or "finemath" in low:
        return "math"
    if "synth_code" in low or "stack" in low:
        return "code"
    if "synth_thinking" in low or "synth_qa" in low or "instruction" in low:
        return "synthetic_reasoning"
    if "arxiv" in low or "stem" in low or "science_math" in low:
        return "stem"
    if "wikipedia" in low:
        return "wiki"
    if "olmocr" in low:
        return "pdf"
    if "common_crawl" in low or "dolma3_cc/" in low:
        return "web"
    return "other"


def family_matrix(domains: list[str]) -> tuple[list[str], np.ndarray]:
    families = sorted({domain_family(domain) for domain in domains})
    matrix = np.zeros((len(domains), len(families)), dtype=float)
    index = {family: i for i, family in enumerate(families)}
    for row, domain in enumerate(domains):
        matrix[row, index[domain_family(domain)]] = 1.0
    return families, matrix


def lrq_features(w0: np.ndarray, w1: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    exposure = 0.8 * w0 + 0.2 * w1
    shift = w1 - w0
    families, fam_map = family_matrix(domains)
    fam_exposure = exposure @ fam_map
    fam_shift = shift @ fam_map
    entropy0 = -(w0 * np.log(np.clip(w0, 1e-12, None))).sum(axis=1, keepdims=True)
    entropy1 = -(w1 * np.log(np.clip(w1, 1e-12, None))).sum(axis=1, keepdims=True)
    phase_tv = 0.5 * np.abs(w1 - w0).sum(axis=1, keepdims=True)

    blocks = [
        np.ones((w0.shape[0], 1)),
        exposure,
        np.sqrt(np.clip(exposure, 0.0, None)),
        np.log1p(20.0 * exposure),
        shift,
        fam_exposure,
        fam_shift,
        entropy0,
        entropy1,
        phase_tv,
    ]
    names = (
        ["intercept"]
        + [f"exposure:{domain}" for domain in domains]
        + [f"sqrt_exposure:{domain}" for domain in domains]
        + [f"log_exposure:{domain}" for domain in domains]
        + [f"phase_shift:{domain}" for domain in domains]
        + [f"family_exposure:{family}" for family in families]
        + [f"family_shift:{family}" for family in families]
        + ["entropy_phase0", "entropy_phase1", "phase_tv"]
    )
    return names, np.hstack(blocks)


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    return {
        "rows": float(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan"),
        "spearman": rank_corr(y, pred) if len(y) > 1 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=PACKET_ROOT / "outputs" / "mct_lrq_demo")
    parser.add_argument("--ridge-anchor", type=float, default=1e-4)
    parser.add_argument("--ridge-scale", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    if args.target not in df:
        raise ValueError(f"Target column not found: {args.target}")
    required = ["non_embedding_params", "target_budget"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    domains = phase_domains(list(df.columns))
    w0_all, w1_all = normalized_phase_arrays(df, domains)
    y_all = pd.to_numeric(df[args.target], errors="coerce").to_numpy(dtype=float)
    n_all = pd.to_numeric(df["non_embedding_params"], errors="coerce").to_numpy(dtype=float)
    d_all = pd.to_numeric(df["target_budget"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_all) & np.isfinite(n_all) & np.isfinite(d_all) & (n_all > 0) & (d_all > 0)
    df = df.loc[valid].reset_index(drop=True)
    w0 = w0_all[valid]
    w1 = w1_all[valid]
    y = y_all[valid]
    n = n_all[valid]
    d = d_all[valid]

    constants = ScaleConstants()
    names, x_anchor = lrq_features(w0, w1, domains)
    anchor_mask = (np.abs(n / constants.n0 - 1.0) < 0.03) & (np.abs(d / constants.d0 - 1.0) < 0.03)
    if anchor_mask.sum() < max(20, x_anchor.shape[1] // 4):
        anchor_mask = np.ones(len(y), dtype=bool)

    x_anchor_std, x_mean, x_std = standardize_fit(x_anchor[anchor_mask])
    theta_anchor_std = ridge_fit(x_anchor_std, y[anchor_mask], args.ridge_anchor)
    theta_anchor = theta_anchor_std / x_std
    theta_anchor[0] = theta_anchor_std[0] - np.dot(x_mean / x_std, theta_anchor_std)
    anchor_pred = x_anchor @ theta_anchor

    families, fam_map = family_matrix(domains)
    family_share = (0.8 * w0 + 0.2 * w1) @ fam_map
    n_term = (n / constants.n0) ** (-constants.alpha) - 1.0
    d_term = (d / constants.d0) ** (-constants.beta) - 1.0
    cross_term = (n / constants.n0) ** (-constants.gamma) * (d / constants.d0) ** (-constants.delta) - 1.0
    z_scale = np.column_stack([n_term, d_term[:, None] * family_share, cross_term])
    scale_names = ["A_global_N"] + [f"B_family_D:{family}" for family in families] + ["C_cross_ND"]

    z_std, z_mean, z_scale_std = standardize_fit(np.column_stack([np.ones(len(y)), z_scale]))
    residual = y - anchor_pred
    theta_scale_std = ridge_fit(z_std, residual, args.ridge_scale)
    theta_scale = theta_scale_std[1:] / z_scale_std[1:]
    scale_intercept = theta_scale_std[0] - np.dot(z_mean[1:] / z_scale_std[1:], theta_scale_std[1:])
    pred = anchor_pred + scale_intercept + z_scale @ theta_scale

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pred_df = df[
        [
            column
            for column in [
                "registry_run_key",
                "run_name",
                "scale",
                "scale_display_label",
                "target_budget_multiplier",
                "fit_role",
            ]
            if column in df
        ]
    ].copy()
    pred_df["actual"] = y
    pred_df["predicted"] = pred
    pred_df["residual"] = pred - y
    pred_df.to_csv(out / "predictions.csv", index=False)

    summary: dict[str, object] = {
        "target": args.target,
        "constants": asdict(constants),
        "rows": int(len(y)),
        "domains": int(len(domains)),
        "families": families,
        "anchor_rows": int(anchor_mask.sum()),
        "overall": metrics(y, pred),
        "scale_coefficients": dict(zip(scale_names, map(float, theta_scale), strict=True)),
    }
    if "scale_display_label" in df:
        summary["by_scale"] = {
            str(scale): metrics(y[group.index.to_numpy()], pred[group.index.to_numpy()])
            for scale, group in df.groupby("scale_display_label")
        }
    if "target_budget_multiplier" in df:
        summary["by_target_budget_multiplier"] = {
            str(mult): metrics(y[group.index.to_numpy()], pred[group.index.to_numpy()])
            for mult, group in df.groupby("target_budget_multiplier")
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pd.DataFrame({"feature": names, "coefficient": theta_anchor}).to_csv(out / "anchor_coefficients.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
