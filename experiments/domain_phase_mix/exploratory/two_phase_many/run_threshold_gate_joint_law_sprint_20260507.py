#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test scale-dependent threshold gates on top of an MCT-LRQ-style law.

This is a local analysis script. It does not launch training or mutate
registries. The goal is to test whether a Gu et al.-style frequency threshold
term can improve joint mixture/scale prediction without destroying optimum
quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[4]
DATASET = ROOT / "analysis_dataset" / "nd_scale_runs.csv"
CANONICAL_SPLIT_FILE = (
    ROOT
    / "reference_outputs"
    / "joint_model_refreshed_20260426"
    / "mct_lrq_no_barrier_canonical"
    / "csv"
    / "row_predictions.csv"
)
OUTPUT_DIR = ROOT / "reference_outputs" / "threshold_gate_joint_law_20260507"
TARGET = "eval/uncheatable_eval/bpb"

N0 = 102_648_576.0
D0 = 5_999_951_872.0
ALPHA = 0.154791
BETA = 0.146425
GAMMA = 0.014295
DELTA = 1.063376

EPS = 1e-8
RNG_SEED = 20260507
RIDGE_ANCHOR = 1e-4
RIDGE_SCALE = 1e-4


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    description: str
    group_fn_name: str


@dataclass(frozen=True)
class FitSpec:
    partition: str
    quantile: float
    steepness: float
    nu: float

    @property
    def name(self) -> str:
        if self.partition == "smooth_baseline":
            return "smooth_mct_lrq_like"
        return f"threshold_{self.partition}_q{self.quantile:g}_k{self.steepness:g}_nu{self.nu:g}"


@dataclass
class FittedModel:
    spec: FitSpec
    domains: list[str]
    anchor_feature_names: list[str]
    anchor_coef: np.ndarray
    anchor_mean: np.ndarray
    anchor_std: np.ndarray
    scale_feature_names: list[str]
    scale_coef: np.ndarray
    threshold_groups: list[str]
    threshold_tau: np.ndarray
    canonical_families: list[str]
    canonical_family_map: np.ndarray
    partition_map: np.ndarray | None


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(column.removeprefix("phase_0_") for column in columns if column.startswith("phase_0_"))
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


def canonical_family(domain: str) -> str:
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


def dense_vs_broad_family(domain: str) -> str:
    low = domain.lower()
    if "dolma3_cc/" in low or "dolmino_common_crawl" in low:
        return "broad_web"
    if "stack" in low or "code" in low:
        return "code"
    if "math" in low or "arxiv" in low or "stem" in low or "science" in low:
        return "stem_math"
    if "wikipedia" in low or "olmocr" in low:
        return "knowledge_docs"
    if "synth" in low or "instruction" in low:
        return "synthetic_reasoning"
    return "other"


def quality_three_family(domain: str) -> str:
    low = domain.lower()
    if low.endswith("_high") or "dolmino" in low or "stack_edu" in low or "arxiv" in low or "wikipedia" in low:
        return "high_quality_or_curated"
    if low.endswith("_low"):
        return "low_quality_cc"
    if "synth" in low or "finemath" in low:
        return "synthetic_or_math"
    return "other"


def topic_quality_family(domain: str) -> str:
    low = domain.lower()
    if "dolma3_cc/" in low:
        topic = low.split("dolma3_cc/", 1)[1]
        if topic.endswith("_high"):
            return "cc_high"
        if topic.endswith("_low"):
            return "cc_low"
        return "cc_other"
    return dense_vs_broad_family(domain)


def current_source_family(domain: str) -> str:
    low = domain.lower()
    if "dolma3_cc/" in low:
        return "dolma3_cc"
    if "dolmino_common_crawl" in low:
        return "dolmino_cc"
    if "stack" in low or "code" in low:
        return "code"
    if "math" in low or "arxiv" in low or "stem" in low:
        return "stem_math"
    if "synth" in low or "instruction" in low:
        return "synthetic_reasoning"
    if "wikipedia" in low:
        return "wiki"
    if "olmocr" in low:
        return "pdf"
    return "other"


def group_matrix(domains: list[str], group_fn_name: str) -> tuple[list[str], np.ndarray]:
    functions = {
        "canonical": canonical_family,
        "dense_vs_broad": dense_vs_broad_family,
        "quality_three": quality_three_family,
        "topic_quality": topic_quality_family,
        "current_source": current_source_family,
        "all": lambda _: "all_domains",
    }
    group_fn = functions[group_fn_name]
    groups = sorted({group_fn(domain) for domain in domains})
    group_index = {group: index for index, group in enumerate(groups)}
    matrix = np.zeros((len(domains), len(groups)), dtype=float)
    for row, domain in enumerate(domains):
        matrix[row, group_index[group_fn(domain)]] = 1.0
    return groups, matrix


def partition_specs() -> list[PartitionSpec]:
    return [
        PartitionSpec(
            name="dense_vs_broad",
            description="Broad web, code, STEM/math, knowledge docs, synthetic reasoning, other.",
            group_fn_name="dense_vs_broad",
        ),
        PartitionSpec(
            name="quality_three",
            description="Low-quality CC, high-quality/curated, synthetic/math, other.",
            group_fn_name="quality_three",
        ),
        PartitionSpec(
            name="current_source",
            description="Source-aware groups aligned with Dolma/Dolmino source families.",
            group_fn_name="current_source",
        ),
        PartitionSpec(
            name="canonical",
            description="The existing canonical MCT-LRQ family partition.",
            group_fn_name="canonical",
        ),
        PartitionSpec(
            name="all",
            description="Single global threshold gate shared across every domain.",
            group_fn_name="all",
        ),
    ]


def lrq_anchor_features(w0: np.ndarray, w1: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    exposure = 0.8 * w0 + 0.2 * w1
    shift = w1 - w0
    families, family_map = group_matrix(domains, "canonical")
    family_exposure = exposure @ family_map
    family_shift = shift @ family_map
    entropy0 = -(w0 * np.log(np.clip(w0, EPS, None))).sum(axis=1, keepdims=True)
    entropy1 = -(w1 * np.log(np.clip(w1, EPS, None))).sum(axis=1, keepdims=True)
    phase_tv = 0.5 * np.abs(w1 - w0).sum(axis=1, keepdims=True)
    blocks = [
        np.ones((w0.shape[0], 1)),
        exposure,
        np.sqrt(np.clip(exposure, 0.0, None)),
        np.log1p(20.0 * exposure),
        shift,
        family_exposure,
        family_shift,
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


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float, penalize_intercept: bool) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    actual_std = float(np.std(y))
    pred_std = float(np.std(pred))
    if actual_std > 0 and pred_std > 0:
        slope_pred_actual, intercept_pred_actual = np.polyfit(y, pred, 1)
        slope_actual_pred, intercept_actual_pred = np.polyfit(pred, y, 1)
    else:
        slope_pred_actual = intercept_pred_actual = slope_actual_pred = intercept_actual_pred = float("nan")
    low_tail = y <= np.quantile(y, 0.2)
    return {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_actual": float(np.mean(residual)),
        "actual_std": actual_std,
        "pred_std": pred_std,
        "std_ratio": float(pred_std / actual_std) if actual_std > 0 else float("nan"),
        "spearman": rank_corr(y, pred),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan"),
        "slope_pred_on_actual": float(slope_pred_actual),
        "intercept_pred_on_actual": float(intercept_pred_actual),
        "slope_actual_on_pred": float(slope_actual_pred),
        "intercept_actual_on_pred": float(intercept_actual_pred),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[low_tail] ** 2))) if np.any(low_tail) else float("nan"),
    }


def scale_features(
    w0: np.ndarray,
    w1: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    domains: list[str],
    canonical_family_map: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    exposure = 0.8 * w0 + 0.2 * w1
    family_share = exposure @ canonical_family_map
    families, _ = group_matrix(domains, "canonical")
    n_term = (n / N0) ** (-ALPHA) - 1.0
    d_term = (d / D0) ** (-BETA) - 1.0
    cross_term = (n / N0) ** (-GAMMA) * (d / D0) ** (-DELTA) - 1.0
    names = ["A_global_N"] + [f"B_family_D:{family}" for family in families] + ["C_cross_ND"]
    features = np.column_stack([n_term, d_term[:, None] * family_share, cross_term])
    return names, features


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def threshold_features(
    w0: np.ndarray,
    w1: np.ndarray,
    n: np.ndarray,
    partition_map: np.ndarray,
    tau: np.ndarray,
    steepness: float,
    nu: float,
) -> np.ndarray:
    exposure = (0.8 * w0 + 0.2 * w1) @ partition_map
    log_exposure = np.log(np.clip(exposure, EPS, None))
    threshold = np.clip(tau[None, :] * (n[:, None] / N0) ** (-nu), EPS, None)
    anchor_threshold = np.clip(tau[None, :], EPS, None)
    gate = sigmoid(steepness * (log_exposure - np.log(threshold)))
    anchor_gate = sigmoid(steepness * (log_exposure - np.log(anchor_threshold)))
    return gate - anchor_gate


def threshold_tau_from_anchor(
    w0: np.ndarray,
    w1: np.ndarray,
    partition_map: np.ndarray,
    anchor_mask: np.ndarray,
    quantile: float,
) -> np.ndarray:
    exposure = (0.8 * w0 + 0.2 * w1) @ partition_map
    anchor_exposure = exposure[anchor_mask]
    tau = np.zeros(anchor_exposure.shape[1], dtype=float)
    for index in range(anchor_exposure.shape[1]):
        values = anchor_exposure[:, index]
        values = values[values > EPS]
        tau[index] = float(np.quantile(values, quantile)) if len(values) else EPS
    return np.clip(tau, EPS, None)


def fit_model(
    df: pd.DataFrame,
    w0: np.ndarray,
    w1: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    domains: list[str],
    spec: FitSpec,
) -> tuple[FittedModel, np.ndarray]:
    canonical_families, canonical_family_map = group_matrix(domains, "canonical")
    anchor_names, anchor_x = lrq_anchor_features(w0, w1, domains)
    anchor_mask = (
        (df["scale_display_label"] == "100M/6B")
        & (np.isclose(df["target_budget_multiplier"].to_numpy(dtype=float), 1.0))
        & (df["fit_role"] == "fit_region")
    ).to_numpy()
    if anchor_mask.sum() < 50:
        raise ValueError(f"Anchor mask too small: {anchor_mask.sum()}")
    anchor_x_std, anchor_mean, anchor_std = standardize_fit(anchor_x[anchor_mask])
    anchor_coef_std = ridge_fit(anchor_x_std, y[anchor_mask], RIDGE_ANCHOR, penalize_intercept=False)
    anchor_pred = standardize_apply(anchor_x, anchor_mean, anchor_std) @ anchor_coef_std

    scale_names, scale_x = scale_features(w0, w1, n, d, domains, canonical_family_map)
    partition_map = None
    threshold_groups: list[str] = []
    tau = np.zeros(0, dtype=float)
    z = scale_x
    z_names = list(scale_names)
    if spec.partition != "smooth_baseline":
        partition = next(item for item in partition_specs() if item.name == spec.partition)
        threshold_groups, partition_map = group_matrix(domains, partition.group_fn_name)
        tau = threshold_tau_from_anchor(w0, w1, partition_map, anchor_mask, spec.quantile)
        gate_x = threshold_features(w0, w1, n, partition_map, tau, spec.steepness, spec.nu)
        z = np.hstack([scale_x, gate_x])
        z_names += [f"threshold_gate:{group}" for group in threshold_groups]

    residual = y - anchor_pred
    fit_mask = split_column(df, "seed7_train")
    if not np.any(fit_mask):
        fit_mask = (df["fit_role"] == "fit_region").to_numpy()
    coef = ridge_fit(z[fit_mask], residual[fit_mask], RIDGE_SCALE, penalize_intercept=True)
    pred = anchor_pred + z @ coef
    model = FittedModel(
        spec=spec,
        domains=domains,
        anchor_feature_names=anchor_names,
        anchor_coef=anchor_coef_std,
        anchor_mean=anchor_mean,
        anchor_std=anchor_std,
        scale_feature_names=z_names,
        scale_coef=coef,
        threshold_groups=threshold_groups,
        threshold_tau=tau,
        canonical_families=canonical_families,
        canonical_family_map=canonical_family_map,
        partition_map=partition_map,
    )
    return model, pred


def predict_model(model: FittedModel, w0: np.ndarray, w1: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    _, anchor_x = lrq_anchor_features(w0, w1, model.domains)
    anchor_pred = standardize_apply(anchor_x, model.anchor_mean, model.anchor_std) @ model.anchor_coef
    _, scale_x = scale_features(w0, w1, n, d, model.domains, model.canonical_family_map)
    z = scale_x
    if model.spec.partition != "smooth_baseline":
        if model.partition_map is None:
            raise ValueError("Threshold model missing partition map")
        gate_x = threshold_features(
            w0,
            w1,
            n,
            model.partition_map,
            model.threshold_tau,
            model.spec.steepness,
            model.spec.nu,
        )
        z = np.hstack([scale_x, gate_x])
    return anchor_pred + z @ model.scale_coef


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    scale = df["scale_display_label"]
    multiplier = df["target_budget_multiplier"].to_numpy(dtype=float)
    fit_role = df["fit_role"]
    masks = {
        "all_rows": np.ones(len(df), dtype=bool),
        "fit_region": (fit_role == "fit_region").to_numpy(),
        "anchor_100m_1x": ((scale == "100M/6B") & np.isclose(multiplier, 1.0)).to_numpy(),
        "fixed340_all": (scale == "340M/10.4B").to_numpy(),
        "fixed340_1x": ((scale == "340M/10.4B") & np.isclose(multiplier, 1.0)).to_numpy(),
        "external_60m": (scale == "60M/1.2B").to_numpy(),
        "external_900m": (scale == "900M/24B").to_numpy(),
    }
    for column in ("seed7_train", "seed7_holdout", "fixed340_holdout", "random_supplement", "all900_holdout"):
        mask = split_column(df, column)
        if np.any(mask):
            masks[column] = mask
    return masks


def split_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.zeros(len(df), dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False).to_numpy(dtype=bool)
    return values.fillna(False).astype(bool).to_numpy()


def summarize_model(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, model_name: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for split, mask in split_masks(df).items():
        if not np.any(mask):
            continue
        metrics = regression_metrics(y[mask], pred[mask])
        rows.append({"model": model_name, "split": split, **metrics})
    return rows


def fixed340_drop_summary(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> dict[str, float | str]:
    work = df.copy()
    work["pred"] = pred
    work = work[work["scale_display_label"] == "340M/10.4B"]
    pivot_actual = work.pivot_table(
        index="mixture_id",
        columns="target_budget_multiplier",
        values=TARGET,
        aggfunc="mean",
    )
    pivot_pred = work.pivot_table(
        index="mixture_id",
        columns="target_budget_multiplier",
        values="pred",
        aggfunc="mean",
    )
    out: dict[str, float | str] = {"model": model_name}
    for left, right, label in ((0.5, 1.0, "drop_0p5_to_1"), (0.5, 2.0, "drop_0p5_to_2"), (1.0, 2.0, "drop_1_to_2")):
        actual_rows = (
            pivot_actual[[left, right]].dropna() if left in pivot_actual and right in pivot_actual else pd.DataFrame()
        )
        pred_rows = pivot_pred[[left, right]].dropna() if left in pivot_pred and right in pivot_pred else pd.DataFrame()
        common = actual_rows.index.intersection(pred_rows.index)
        if len(common) == 0:
            out[f"{label}_actual_mean"] = float("nan")
            out[f"{label}_pred_mean"] = float("nan")
            out[f"{label}_ratio"] = float("nan")
            out[f"{label}_n"] = 0.0
            continue
        actual_drop = actual_rows.loc[common, left] - actual_rows.loc[common, right]
        pred_drop = pred_rows.loc[common, left] - pred_rows.loc[common, right]
        actual_mean = float(actual_drop.mean())
        pred_mean = float(pred_drop.mean())
        out[f"{label}_actual_mean"] = actual_mean
        out[f"{label}_pred_mean"] = pred_mean
        out[f"{label}_ratio"] = pred_mean / actual_mean if actual_mean != 0 else float("nan")
        out[f"{label}_n"] = float(len(common))
    return out


def sample_candidate_weights(
    domains: list[str],
    observed_w0: np.ndarray,
    observed_w1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(RNG_SEED)
    candidates0 = [observed_w0, observed_w1]
    candidates1 = [observed_w1, observed_w0]
    labels = ["observed", "observed_swapped"]
    for alpha in (0.15, 0.4, 1.0):
        candidates0.append(rng.dirichlet(np.full(len(domains), alpha), size=6_000))
        candidates1.append(rng.dirichlet(np.full(len(domains), alpha), size=6_000))
        labels.append(f"dirichlet_alpha_{alpha:g}")
    corners = np.eye(len(domains))
    corner0 = np.repeat(corners, len(domains), axis=0)
    corner1 = np.tile(corners, (len(domains), 1))
    candidates0.append(corner0)
    candidates1.append(corner1)
    labels.append("corner_pairs")
    return np.vstack(candidates0), np.vstack(candidates1), ",".join(labels)


def family_summary_for_weights(domains: list[str], w0: np.ndarray, w1: np.ndarray) -> dict[str, float]:
    groups, group_map = group_matrix(domains, "dense_vs_broad")
    p0 = w0 @ group_map
    p1 = w1 @ group_map
    row: dict[str, float] = {
        "p0_support_inv_l2": float(1.0 / np.sum(w0**2)),
        "p1_support_inv_l2": float(1.0 / np.sum(w1**2)),
        "p0_max_domain_weight": float(np.max(w0)),
        "p1_max_domain_weight": float(np.max(w1)),
        "max_domain_weight": float(max(np.max(w0), np.max(w1))),
    }
    for index, group in enumerate(groups):
        row[f"p0_{group}_share"] = float(p0[index])
        row[f"p1_{group}_share"] = float(p1[index])
    return row


def nearest_observed_phase_mean_tv(
    w0: np.ndarray,
    w1: np.ndarray,
    observed_w0: np.ndarray,
    observed_w1: np.ndarray,
) -> float:
    tv0 = 0.5 * np.abs(observed_w0 - w0[None, :]).sum(axis=1)
    tv1 = 0.5 * np.abs(observed_w1 - w1[None, :]).sum(axis=1)
    return float(np.min(0.5 * (tv0 + tv1)))


def optimum_diagnostics(
    models: list[FittedModel],
    df: pd.DataFrame,
    observed_w0: np.ndarray,
    observed_w1: np.ndarray,
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand_w0, cand_w1, candidate_sources = sample_candidate_weights(domains, observed_w0, observed_w1)
    scales = [
        ("100M/6B", N0, D0),
        ("340M/10.4B", 339_813_888.0, 10_399_940_608.0),
        ("900M/24B", 906_037_248.0, 23_999_803_392.0),
    ]
    rows: list[dict[str, float | str | bool]] = []
    top_domain_rows: list[dict[str, float | str]] = []
    for model in models:
        for scale_name, n_value, d_value in scales:
            n = np.full(len(cand_w0), n_value, dtype=float)
            d = np.full(len(cand_w0), d_value, dtype=float)
            pred = predict_model(model, cand_w0, cand_w1, n, d)
            best_index = int(np.argmin(pred))
            best_w0 = cand_w0[best_index]
            best_w1 = cand_w1[best_index]
            row = {
                "model": model.spec.name,
                "partition": model.spec.partition,
                "target_scale": scale_name,
                "opt_kind": "raw_random_plus_corners",
                "candidate_count": float(len(cand_w0)),
                "candidate_sources": candidate_sources,
                "predicted_bpb": float(pred[best_index]),
                "nearest_observed_phase_mean_tv": nearest_observed_phase_mean_tv(
                    best_w0,
                    best_w1,
                    observed_w0,
                    observed_w1,
                ),
                "hard_corner_flag": bool(max(np.max(best_w0), np.max(best_w1)) > 0.5),
                "any_family_collapse_flag": bool(
                    max(
                        np.max(best_w0 @ model.canonical_family_map),
                        np.max(best_w1 @ model.canonical_family_map),
                    )
                    > 0.95
                ),
                **family_summary_for_weights(domains, best_w0, best_w1),
            }
            rows.append(row)
            for phase, weights in (("phase0", best_w0), ("phase1", best_w1)):
                order = np.argsort(weights)[::-1][:10]
                for rank, domain_index in enumerate(order, start=1):
                    top_domain_rows.append(
                        {
                            "model": model.spec.name,
                            "target_scale": scale_name,
                            "phase": phase,
                            "rank": float(rank),
                            "domain": domains[domain_index],
                            "weight": float(weights[domain_index]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(top_domain_rows)


def build_specs() -> list[FitSpec]:
    specs = [FitSpec("smooth_baseline", 0.0, 0.0, 0.0)]
    for partition in [item.name for item in partition_specs()]:
        for quantile in (0.25, 0.5, 0.75):
            for steepness in (3.0, 6.0, 10.0):
                for nu in (0.75, ALPHA + 1.0, 1.5):
                    specs.append(FitSpec(partition, quantile, steepness, nu))
    return specs


def selected_models(summary_df: pd.DataFrame) -> list[str]:
    pivot = summary_df.pivot_table(index="model", columns="split", values="rmse", aggfunc="first")
    base = pivot["seed7_holdout"] if "seed7_holdout" in pivot else pivot["fit_region"]
    fixed = pivot["fixed340_holdout"] if "fixed340_holdout" in pivot else pivot["fixed340_all"]
    random = pivot["random_supplement"] if "random_supplement" in pivot else pivot["external_60m"]
    all900 = pivot["all900_holdout"] if "all900_holdout" in pivot else pivot["external_900m"]
    pivot["selection_score"] = base + fixed + random + all900
    best = pivot.sort_values("selection_score").head(8).index.tolist()
    if "smooth_mct_lrq_like" not in best:
        best = ["smooth_mct_lrq_like", *best[:7]]
    return best


def write_plot(fig: go.Figure, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path)
    try:
        fig.write_image(html_path.with_suffix(".png"), scale=2)
    except ValueError:
        # Static image export depends on kaleido; HTML is the canonical output.
        pass


def make_plots(out_dir: Path, predictions: pd.DataFrame, summary: pd.DataFrame, optima: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    selected = selected_models(summary)
    pred_plot = predictions[predictions["model"].isin(selected)].copy()
    fig = px.scatter(
        pred_plot,
        x="actual_bpb",
        y="pred_bpb",
        color="scale_display_label",
        facet_col="model",
        facet_col_wrap=3,
        hover_data=["run_name", "target_budget_multiplier", "fit_role", "residual_pred_minus_actual"],
        title="Threshold-gate joint-law predicted vs actual BPB",
    )
    lo = min(pred_plot["actual_bpb"].min(), pred_plot["pred_bpb"].min())
    hi = max(pred_plot["actual_bpb"].max(), pred_plot["pred_bpb"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash", "color": "black"})
    write_plot(fig, plot_dir / "threshold_gate_pred_actual.html")

    external = summary[summary["split"].isin(["external_60m", "external_900m", "fixed340_all"])].copy()
    external = external[external["model"].isin(selected)]
    fig = px.bar(
        external,
        x="model",
        y="rmse",
        color="split",
        barmode="group",
        title="Holdout and fixed-340M RMSE for selected threshold-gate variants",
    )
    fig.update_layout(xaxis_tickangle=-35)
    write_plot(fig, plot_dir / "threshold_gate_rmse_selected.html")

    opt_plot = optima[optima["model"].isin(selected)].copy()
    shares = [
        "p0_broad_web_share",
        "p0_code_share",
        "p0_stem_math_share",
        "p0_knowledge_docs_share",
        "p0_synthetic_reasoning_share",
        "p1_broad_web_share",
        "p1_code_share",
        "p1_stem_math_share",
        "p1_knowledge_docs_share",
        "p1_synthetic_reasoning_share",
    ]
    present = [column for column in shares if column in opt_plot]
    long = opt_plot.melt(
        id_vars=["model", "target_scale", "nearest_observed_phase_mean_tv", "hard_corner_flag"],
        value_vars=present,
        var_name="share",
        value_name="weight",
    )
    fig = px.bar(
        long,
        x="model",
        y="weight",
        color="share",
        facet_col="target_scale",
        title="Raw random-search optimum family shares for selected variants",
        hover_data=["nearest_observed_phase_mean_tv", "hard_corner_flag"],
    )
    fig.update_layout(xaxis_tickangle=-35)
    write_plot(fig, plot_dir / "threshold_gate_optimum_family_shares.html")


def make_canonical_residual_plot(out_dir: Path, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    selected = [
        "canonical_mct_lrq69_drop",
        *[model for model in summary.sort_values("score")["model"].tolist() if model != "canonical_mct_lrq69_drop"][:5],
    ]
    work = predictions[predictions["model"].isin(selected)].copy()
    fig = px.scatter(
        work,
        x="actual_bpb",
        y="pred_bpb",
        color="scale_display_label",
        facet_col="model",
        facet_col_wrap=3,
        hover_data=["run_name", "target_budget_multiplier", "fit_role", "residual_pred_minus_actual"],
        title="Canonical MCT-LRQ plus threshold-residual corrections",
    )
    lo = min(work["actual_bpb"].min(), work["pred_bpb"].min())
    hi = max(work["actual_bpb"].max(), work["pred_bpb"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash", "color": "black"})
    write_plot(fig, plot_dir / "canonical_residual_threshold_pred_actual.html")


def run_canonical_residual_diagnostic(
    df: pd.DataFrame,
    w0: np.ndarray,
    w1: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "pred_bpb" not in df:
        canonical = pd.read_csv(CANONICAL_SPLIT_FILE)
        canonical = canonical[
            (canonical["model"] == "mct_lrq69_drop_no_barrier") & (canonical["fit_protocol"] == "seed7")
        ]
        pred_map = canonical.set_index("registry_run_key")["pred_bpb"]
        base_pred = df["registry_run_key"].map(pred_map).to_numpy(dtype=float)
    else:
        base_pred = df["pred_bpb"].to_numpy(dtype=float)
    finite = np.isfinite(base_pred)
    work = df.loc[finite].reset_index(drop=True)
    local_w0 = w0[finite]
    local_w1 = w1[finite]
    local_y = y[finite]
    local_n = n[finite]
    base_pred = base_pred[finite]
    residual = local_y - base_pred
    anchor_mask = (
        (work["scale_display_label"] == "100M/6B")
        & (np.isclose(work["target_budget_multiplier"].to_numpy(dtype=float), 1.0))
        & (work["fit_role"] == "fit_region")
    ).to_numpy()
    fit_mask = split_column(work, "seed7_train")
    rows: list[dict[str, float | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for partition in partition_specs():
        groups, partition_map = group_matrix(domains, partition.group_fn_name)
        del groups
        for quantile in (0.25, 0.5, 0.75):
            for steepness in (3.0, 6.0, 10.0):
                for nu in (0.75, ALPHA + 1.0, 1.5):
                    tau = threshold_tau_from_anchor(local_w0, local_w1, partition_map, anchor_mask, quantile)
                    features = threshold_features(local_w0, local_w1, local_n, partition_map, tau, steepness, nu)
                    coef = ridge_fit(features[fit_mask], residual[fit_mask], RIDGE_SCALE, penalize_intercept=True)
                    pred = base_pred + features @ coef
                    name = f"canon_resid_{partition.name}_q{quantile:g}_k{steepness:g}_nu{nu:g}"
                    row: dict[str, float | str] = {
                        "model": name,
                        "partition": partition.name,
                        "quantile": quantile,
                        "steepness": steepness,
                        "nu": nu,
                    }
                    for split, mask in split_masks(work).items():
                        if split not in {
                            "seed7_train",
                            "seed7_holdout",
                            "fixed340_holdout",
                            "random_supplement",
                            "all900_holdout",
                            "fixed340_all",
                        }:
                            continue
                        if not np.any(mask):
                            continue
                        metrics = regression_metrics(local_y[mask], pred[mask])
                        row[f"{split}_rmse"] = metrics["rmse"]
                        row[f"{split}_spearman"] = metrics["spearman"]
                    rows.append(row)
                    pred_df = work[
                        [
                            column
                            for column in [
                                "registry_run_key",
                                "run_name",
                                "mixture_id",
                                "scale",
                                "scale_display_label",
                                "target_budget_multiplier",
                                "fit_role",
                            ]
                            if column in work
                        ]
                    ].copy()
                    pred_df["model"] = name
                    pred_df["partition"] = partition.name
                    pred_df["actual_bpb"] = local_y
                    pred_df["pred_bpb"] = pred
                    pred_df["residual_pred_minus_actual"] = pred - local_y
                    prediction_rows.append(pred_df)
    baseline: dict[str, float | str] = {
        "model": "canonical_mct_lrq69_drop",
        "partition": "baseline",
        "quantile": float("nan"),
        "steepness": float("nan"),
        "nu": float("nan"),
    }
    for split, mask in split_masks(work).items():
        if split not in {
            "seed7_train",
            "seed7_holdout",
            "fixed340_holdout",
            "random_supplement",
            "all900_holdout",
            "fixed340_all",
        }:
            continue
        if not np.any(mask):
            continue
        metrics = regression_metrics(local_y[mask], base_pred[mask])
        baseline[f"{split}_rmse"] = metrics["rmse"]
        baseline[f"{split}_spearman"] = metrics["spearman"]
    rows.append(baseline)
    pred_df = work[
        [
            column
            for column in [
                "registry_run_key",
                "run_name",
                "mixture_id",
                "scale",
                "scale_display_label",
                "target_budget_multiplier",
                "fit_role",
            ]
            if column in work
        ]
    ].copy()
    pred_df["model"] = "canonical_mct_lrq69_drop"
    pred_df["partition"] = "baseline"
    pred_df["actual_bpb"] = local_y
    pred_df["pred_bpb"] = base_pred
    pred_df["residual_pred_minus_actual"] = base_pred - local_y
    prediction_rows.append(pred_df)
    result = pd.DataFrame(rows)
    result["score"] = (
        result["seed7_holdout_rmse"]
        + result["fixed340_holdout_rmse"]
        + result["random_supplement_rmse"]
        + result["all900_holdout_rmse"]
    )
    return result.sort_values("score").reset_index(drop=True), pd.concat(prediction_rows, ignore_index=True)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    work = df[columns].copy()
    for column in work.columns:
        if pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
    return work.to_markdown(index=False)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    drops: pd.DataFrame,
    optima: pd.DataFrame,
    canonical_residual: pd.DataFrame,
    partition_rows: list[dict[str, str]],
) -> None:
    selected = selected_models(summary)
    pivot = summary.pivot_table(index="model", columns="split", values=["rmse", "spearman"], aggfunc="first")
    flat = pd.DataFrame(
        {
            "model": pivot.index,
            "seed7_train_rmse": pivot[("rmse", "seed7_train")].to_numpy(),
            "seed7_holdout_rmse": pivot[("rmse", "seed7_holdout")].to_numpy(),
            "fixed340_rmse": pivot[("rmse", "fixed340_all")].to_numpy(),
            "fixed340_holdout_rmse": pivot[("rmse", "fixed340_holdout")].to_numpy(),
            "random_supplement_rmse": pivot[("rmse", "random_supplement")].to_numpy(),
            "external60_rmse": pivot[("rmse", "external_60m")].to_numpy(),
            "external900_rmse": pivot[("rmse", "external_900m")].to_numpy(),
            "all900_holdout_rmse": pivot[("rmse", "all900_holdout")].to_numpy(),
            "seed7_holdout_spearman": pivot[("spearman", "seed7_holdout")].to_numpy(),
        }
    )
    flat["selection_score"] = (
        flat["seed7_holdout_rmse"]
        + flat["fixed340_holdout_rmse"]
        + flat["random_supplement_rmse"]
        + flat["all900_holdout_rmse"]
    )
    flat = flat.sort_values("selection_score")
    drop_cols = ["model", "drop_0p5_to_1_ratio", "drop_0p5_to_2_ratio", "drop_1_to_2_ratio"]
    top = flat.head(12).merge(drops[drop_cols], on="model", how="left")
    optimum_top = optima[(optima["model"].isin(selected)) & (optima["target_scale"] == "340M/10.4B")].copy()
    optimum_top = optimum_top.sort_values(["hard_corner_flag", "nearest_observed_phase_mean_tv", "predicted_bpb"])
    partition_df = pd.DataFrame(partition_rows)
    canonical_top = canonical_residual.sort_values("score").head(12).copy()

    report = [
        "# Threshold-Gated Joint Law Sprint",
        "",
        "## Goal",
        "",
        (
            "Test whether a Gu et al.-style scale-dependent acquisition threshold improves the compact "
            "joint mixture/scale law."
        ),
        "The threshold features are centered at the 100M/6B anchor, so the anchor mixture regression is preserved.",
        "",
        "The fitted family is:",
        "",
        "```latex",
        r"\pi_g(w,N) = \sigma\left(k_g [\log(e_g(w)+\epsilon) - \log(\tau_g (N/N_0)^{-\nu})]\right)",
        r"```",
        "",
        "and the added residual features are:",
        "",
        "```latex",
        r"\Delta_g(w,N) = \pi_g(w,N) - \pi_g(w,N_0).",
        "```",
        "",
        "The model therefore changes fixed-mixture scaling from a single global power law into a smooth crossover law.",
        "",
        "There are two diagnostics in this sprint:",
        "",
        (
            "1. `canonical residual`: fit only the threshold gate residual on top of the existing "
            "canonical MCT-LRQ row predictions. This is the reliable diagnostic for whether threshold "
            "gates explain current model errors on observed rows."
        ),
        (
            "2. `full local model`: fit a compact standalone MCT-LRQ-like model plus gates. This is "
            "useful for raw-optimum stress testing, but it is not numerically identical to the internal "
            "canonical MCT-LRQ implementation."
        ),
        "",
        "## Partitions Tested",
        "",
        markdown_table(partition_df, ["partition", "groups", "description"], float_digits=4),
        "",
        "## Canonical MCT-LRQ Residual Diagnostic",
        "",
        (
            "These rows start from the existing canonical `mct_lrq69_drop_no_barrier` predictions and "
            "fit only threshold-gate corrections to the residuals."
        ),
        "",
        markdown_table(
            canonical_top,
            [
                "model",
                "seed7_holdout_rmse",
                "fixed340_holdout_rmse",
                "random_supplement_rmse",
                "all900_holdout_rmse",
                "score",
                "seed7_holdout_spearman",
                "fixed340_holdout_spearman",
            ],
            float_digits=5,
        ),
        "",
        (
            "Interpretation: the best residual gates reduce the aggregate diagnostic score, mostly by "
            "correcting the tiny 900M set and sometimes the fixed-340M holdout. This is evidence that a "
            "threshold-like feature can explain some residual structure, but the result is not strong "
            "enough to replace the current law because the 900M set has only four rows and some variants "
            "trade off fixed-340M quality."
        ),
        "",
        "## Full Local Model Fit Results",
        "",
        (
            "These rows refit a compact standalone MCT-LRQ-like model plus gates. They should be "
            "interpreted as ablations, not as exact canonical-MCT replacements."
        ),
        "",
        markdown_table(
            top,
            [
                "model",
                "seed7_train_rmse",
                "seed7_holdout_rmse",
                "fixed340_rmse",
                "fixed340_holdout_rmse",
                "random_supplement_rmse",
                "external60_rmse",
                "external900_rmse",
                "all900_holdout_rmse",
                "seed7_holdout_spearman",
                "drop_0p5_to_1_ratio",
                "drop_0p5_to_2_ratio",
                "drop_1_to_2_ratio",
                "selection_score",
            ],
            float_digits=5,
        ),
        "",
        "## Raw Optimum Diagnostics at 340M/10.4B",
        "",
        markdown_table(
            optimum_top,
            [
                "model",
                "predicted_bpb",
                "nearest_observed_phase_mean_tv",
                "hard_corner_flag",
                "any_family_collapse_flag",
                "max_domain_weight",
                "p0_support_inv_l2",
                "p1_support_inv_l2",
            ],
            float_digits=5,
        ),
        "",
        "## Interpretation",
        "",
        "- Threshold gates do appear to capture some residual scale/mixture interaction on observed rows.",
        (
            "- The cleanest residual partition in this run is `current_source`, not the coarse `all` "
            "gate. This suggests source-aware thresholds are more plausible than a single global switch."
        ),
        (
            "- The full local model still raw-optimizes to hard corners, so threshold gates alone do not "
            "solve optimum quality."
        ),
        (
            "- This should be treated as a modeling direction, not a promotion candidate. A production "
            "version should graft constrained threshold gates into the exact canonical MCT-LRQ "
            "implementation and evaluate constrained optima."
        ),
        "",
        "## Artifacts",
        "",
        "- `csv/model_summary.csv`: split metrics for every grid point.",
        (
            "- `csv/canonical_residual_threshold_summary.csv`: threshold-gate residual corrections on top "
            "of canonical MCT-LRQ row predictions."
        ),
        "- `csv/fixed340_drop_summary.csv`: same-mixture target-budget drop ratios.",
        "- `csv/optimum_diagnostics.csv`: random-search plus corner optimum diagnostics.",
        "- `csv/top_optimum_domains.csv`: top domains in raw optima.",
        "- `plots/threshold_gate_pred_actual.html`: prediction scatter.",
        "- `plots/threshold_gate_rmse_selected.html`: selected RMSE comparison.",
        "- `plots/threshold_gate_optimum_family_shares.html`: raw optimum family shares.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    out_dir = OUTPUT_DIR
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)
    if CANONICAL_SPLIT_FILE.exists():
        splits = pd.read_csv(CANONICAL_SPLIT_FILE)
        splits = splits[(splits["model"] == "mct_lrq69_drop_no_barrier") & (splits["fit_protocol"] == "seed7")]
        split_columns = [
            "registry_run_key",
            "seed7_train",
            "seed7_holdout",
            "fixed340_holdout",
            "random_supplement",
            "all900_holdout",
        ]
        df = df.merge(splits[split_columns], on="registry_run_key", how="left")
        for column in split_columns[1:]:
            df[column] = df[column].where(df[column].notna(), False).astype(bool)
    domains = phase_domains(list(df.columns))
    valid = pd.to_numeric(df[TARGET], errors="coerce").notna()
    df = df.loc[valid].reset_index(drop=True)
    w0, w1 = normalized_phase_arrays(df, domains)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df["non_embedding_params"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(df["target_budget"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(n) & np.isfinite(d) & (n > 0) & (d > 0)
    df = df.loc[finite].reset_index(drop=True)
    w0 = w0[finite]
    w1 = w1[finite]
    y = y[finite]
    n = n[finite]
    d = d[finite]

    partition_rows = []
    for spec in partition_specs():
        groups, _ = group_matrix(domains, spec.group_fn_name)
        partition_rows.append(
            {
                "partition": spec.name,
                "groups": ", ".join(groups),
                "description": spec.description,
            }
        )

    summary_rows: list[dict[str, float | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    drop_rows: list[dict[str, float | str]] = []
    models: list[FittedModel] = []
    for spec in build_specs():
        model, pred = fit_model(df, w0, w1, n, d, y, domains, spec)
        models.append(model)
        summary_rows.extend(summarize_model(df, y, pred, spec.name))
        drop_rows.append(fixed340_drop_summary(df, pred, spec.name))
        pred_df = df[
            [
                column
                for column in [
                    "registry_run_key",
                    "run_name",
                    "mixture_id",
                    "scale",
                    "scale_display_label",
                    "target_budget_multiplier",
                    "fit_role",
                ]
                if column in df
            ]
        ].copy()
        pred_df["model"] = spec.name
        pred_df["partition"] = spec.partition
        pred_df["actual_bpb"] = y
        pred_df["pred_bpb"] = pred
        pred_df["residual_pred_minus_actual"] = pred - y
        prediction_rows.append(pred_df)

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    drops = pd.DataFrame(drop_rows)
    selected_names = set(selected_models(summary))
    selected_model_objects = [model for model in models if model.spec.name in selected_names]
    optima, top_domains = optimum_diagnostics(selected_model_objects, df, w0, w1, domains)
    canonical_residual, canonical_residual_predictions = run_canonical_residual_diagnostic(df, w0, w1, y, n, domains)

    summary.to_csv(csv_dir / "model_summary.csv", index=False)
    predictions.to_csv(csv_dir / "row_predictions.csv", index=False)
    drops.to_csv(csv_dir / "fixed340_drop_summary.csv", index=False)
    optima.to_csv(csv_dir / "optimum_diagnostics.csv", index=False)
    top_domains.to_csv(csv_dir / "top_optimum_domains.csv", index=False)
    canonical_residual.to_csv(csv_dir / "canonical_residual_threshold_summary.csv", index=False)
    canonical_residual_predictions.to_csv(csv_dir / "canonical_residual_threshold_predictions.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(csv_dir / "partition_definitions.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target": TARGET,
                "rows": len(df),
                "domains": len(domains),
                "model_count": len(models),
                "selected_models": selected_models(summary),
                "constants": {
                    "n0": N0,
                    "d0": D0,
                    "alpha": ALPHA,
                    "beta": BETA,
                    "gamma": GAMMA,
                    "delta": DELTA,
                    "ridge_anchor": RIDGE_ANCHOR,
                    "ridge_scale": RIDGE_SCALE,
                },
            },
            f,
            indent=2,
        )
    make_plots(out_dir, predictions, summary, optima)
    make_canonical_residual_plot(out_dir, canonical_residual_predictions, canonical_residual)
    write_report(out_dir, summary, drops, optima, canonical_residual, partition_rows)
    ds_store = out_dir / ".DS_Store"
    if ds_store.exists():
        ds_store.unlink()
    print(f"Wrote threshold-gate sprint outputs to {out_dir}")


if __name__ == "__main__":
    main()
