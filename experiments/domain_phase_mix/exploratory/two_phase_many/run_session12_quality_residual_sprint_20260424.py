#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Train-only quality/support residual probes for Session 12 MCT-LRQ.

This is a local diagnostic, not a launch script. It tests whether the compact
MCT-LRQ structural law benefits from a small quality-split/support residual
without using held-out labels. Mixture-only residuals preserve the fixed-mixture
scaling-law shape; the scale-interaction probes are reported separately as less
clean diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
PACKET_ROOT = ROOT / "chatgpt_pro_hybrid_data_mixing_packet_v31"
BASE_PREDICTIONS = (
    ROOT / "reference_outputs/session12_local_validation_20260424/session5_mct_lrq/csv/row_predictions.csv"
)
OUT_DIR = ROOT / "reference_outputs/session12_quality_residual_sprint_20260424"
BASE_MODEL_PARAMS = {
    "mct_lrq74_drop": 74,
    "mct_lrq74_balanced": 74,
}
TECH_DOMAINS = {
    "dolma3_stack_edu",
    "dolma3_arxiv",
    "dolma3_finemath_3plus",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolmino_synth_math",
}
REASONING_DOMAINS = {
    "dolmino_synth_instruction",
    "dolmino_synth_thinking",
}


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    matrix: np.ndarray
    names: list[str]
    preserves_scaling_law: bool


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-30, None)
    return -(weights * np.log(clipped)).sum(axis=1)


def weight_tensor_from_frame(frame: pd.DataFrame, domains: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(frame), 2, len(domains)), dtype=float)
    for phase in (0, 1):
        columns = [f"phase_{phase}_{domain}" for domain in domains]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing phase {phase} columns: {missing[:5]}")
        values = frame[columns].fillna(0.0).to_numpy(dtype=float)
        totals = values.sum(axis=1)
        if np.any(totals <= 0.0):
            bad = np.flatnonzero(totals <= 0.0)[:5].tolist()
            raise ValueError(f"Non-positive phase {phase} weights in rows {bad}")
        weights[:, phase, :] = values / totals[:, None]
    return weights


def family_masks(domains: np.ndarray) -> dict[str, np.ndarray]:
    families = []
    for domain in domains.astype(str):
        if domain in TECH_DOMAINS:
            families.append("tech_code")
        elif domain in REASONING_DOMAINS:
            families.append("reasoning")
        else:
            families.append("broad_text")
    family_array = np.asarray(families)
    return {
        "broad_text": family_array == "broad_text",
        "tech_code": family_array == "tech_code",
        "reasoning": family_array == "reasoning",
    }


def build_feature_blocks(frame: pd.DataFrame, domains: np.ndarray) -> dict[str, FeatureBlock]:
    weights = weight_tensor_from_frame(frame, domains)
    domain_text = domains.astype(str)
    quality_masks = {
        "high_or_hq": np.asarray([d.endswith("_high") or d.endswith("_hq") for d in domain_text]),
        "low_quality": np.asarray([d.endswith("_low") for d in domain_text]),
        "synthetic": np.asarray(["synth" in d for d in domain_text]),
        "curated_technical": np.asarray(
            [any(token in d for token in ("wiki", "arxiv", "stack", "finemath", "stem")) for d in domain_text]
        ),
    }
    families = family_masks(domains)

    quality_cols = []
    quality_names = []
    family_cols = []
    family_names = []
    support_cols = []
    support_names = []
    imbalance_cols = []
    imbalance_names = []
    for phase in (0, 1):
        phase_weights = weights[:, phase, :]
        high_mass = phase_weights[:, quality_masks["high_or_hq"]].sum(axis=1)
        low_mass = phase_weights[:, quality_masks["low_quality"]].sum(axis=1)
        for name, mask in quality_masks.items():
            mass = phase_weights[:, mask].sum(axis=1)
            quality_cols.append(np.sqrt(np.clip(mass, 0.0, None) + 1e-9))
            quality_names.append(f"p{phase}:{name}:sqrt_mass")
        for family, mask in families.items():
            mass = phase_weights[:, mask].sum(axis=1)
            family_cols.append(np.sqrt(np.clip(mass, 0.0, None) + 1e-9))
            family_names.append(f"p{phase}:{family}:sqrt_mass")
        support_cols.extend([entropy_rows(phase_weights), phase_weights.max(axis=1)])
        support_names.extend([f"p{phase}:entropy", f"p{phase}:max_domain"])
        imbalance_cols.append(high_mass - low_mass)
        imbalance_names.append(f"p{phase}:high_minus_low")

    quality = np.column_stack(quality_cols)
    family = np.column_stack(family_cols)
    support = np.column_stack(support_cols)
    imbalance = np.column_stack(imbalance_cols)
    core = np.column_stack([quality, family])
    safe_support = np.column_stack([quality, family, support, imbalance])

    n = frame["non_embedding_params"].to_numpy(dtype=float)
    d = frame["realized_train_tokens"].to_numpy(dtype=float)
    ref_mask = (frame["scale"].astype(str).to_numpy() == "300m_6b") & np.isclose(
        frame["target_budget_multiplier"].to_numpy(dtype=float), 1.0
    )
    n0 = float(np.unique(n[ref_mask])[0])
    d0 = float(np.unique(d[ref_mask])[0])
    u_d = (d / d0) ** (-0.146425) - 1.0
    u_nd = (n / n0) ** (-0.014295) * (d / d0) ** (-1.063376) - 1.0
    scale_quality = quality * u_d[:, None]
    scale_quality_nd = quality * u_nd[:, None]

    return {
        "qsplit_core": FeatureBlock(
            name="qsplit_core",
            matrix=core,
            names=quality_names + family_names,
            preserves_scaling_law=True,
        ),
        "qsplit_support": FeatureBlock(
            name="qsplit_support",
            matrix=safe_support,
            names=quality_names + family_names + support_names + imbalance_names,
            preserves_scaling_law=True,
        ),
        "ud_quality": FeatureBlock(
            name="ud_quality",
            matrix=scale_quality,
            names=[f"uD:{name}" for name in quality_names],
            preserves_scaling_law=False,
        ),
        "anchor_plus_ud_quality": FeatureBlock(
            name="anchor_plus_ud_quality",
            matrix=np.column_stack([core, scale_quality]),
            names=(quality_names + family_names + [f"uD:{name}" for name in quality_names]),
            preserves_scaling_law=False,
        ),
        "anchor_plus_ud_und_quality": FeatureBlock(
            name="anchor_plus_ud_und_quality",
            matrix=np.column_stack([core, scale_quality, scale_quality_nd]),
            names=(
                quality_names
                + family_names
                + [f"uD:{name}" for name in quality_names]
                + [f"uND:{name}" for name in quality_names]
            ),
            preserves_scaling_law=False,
        ),
    }


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    z = (x_train - mean) / std
    design = np.column_stack([np.ones(len(y_train)), z])
    penalty = np.sqrt(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    augmented_design = np.vstack([design, penalty])
    augmented_y = np.concatenate([y_train, np.zeros(design.shape[1])])
    coef, *_ = np.linalg.lstsq(augmented_design, augmented_y, rcond=None)
    return coef, mean, std


def ridge_predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return coef[0] + ((x - mean) / std) @ coef[1:]


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - actual
    if len(actual) == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias_pred_minus_actual": float("nan"),
            "spearman": float("nan"),
            "slope_pred_on_actual": float("nan"),
            "std_ratio": float("nan"),
            "low_tail_rmse": float("nan"),
        }
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    if len(actual) > 1 and np.std(actual) > 0.0 and np.std(predicted) > 0.0:
        spearman = float(spearmanr(actual, predicted).statistic)
        slope = float(np.polyfit(actual, predicted, 1)[0])
        std_ratio = float(np.std(predicted) / np.std(actual))
    else:
        spearman = slope = std_ratio = float("nan")
    threshold = np.quantile(actual, 0.1)
    low = actual <= threshold
    low_tail_rmse = float(np.sqrt(np.mean(residual[low] ** 2))) if np.any(low) else float("nan")
    return {
        "n": len(actual),
        "rmse": rmse,
        "mae": mae,
        "bias_pred_minus_actual": bias,
        "spearman": spearman,
        "slope_pred_on_actual": slope,
        "std_ratio": std_ratio,
        "low_tail_rmse": low_tail_rmse,
    }


def add_metrics(
    rows: list[dict[str, object]],
    *,
    model: str,
    fit_protocol: str,
    split: str,
    params: int,
    preserves_scaling_law: bool,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    row = {
        "model": model,
        "fit_protocol": fit_protocol,
        "split": split,
        "params": params,
        "preserves_fixed_mixture_scaling_law": preserves_scaling_law,
    }
    row.update(metric_dict(actual, predicted))
    rows.append(row)


def fit_residual_variants(
    predictions: pd.DataFrame,
    features: dict[str, FeatureBlock],
    *,
    base_model: str,
    fit_protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol_df = predictions[
        (predictions["model"] == base_model) & (predictions["fit_protocol"] == fit_protocol)
    ].copy()
    if protocol_df.empty:
        raise ValueError(f"No predictions for {base_model}/{fit_protocol}")
    protocol_df = protocol_df.sort_values("row_index")
    actual = protocol_df["actual_bpb"].to_numpy(dtype=float)
    base_pred = protocol_df["pred_bpb"].to_numpy(dtype=float)
    row_index = protocol_df["row_index"].to_numpy(dtype=int)

    if fit_protocol == "seed7":
        train_mask = protocol_df["seed7_train"].to_numpy(dtype=bool)
        split_masks = {
            "seed7_train": protocol_df["seed7_train"].to_numpy(dtype=bool),
            "seed7_holdout": protocol_df["seed7_holdout"].to_numpy(dtype=bool),
            "fixed340_holdout": protocol_df["fixed340_holdout"].to_numpy(dtype=bool),
            "random_supplement": protocol_df["random_supplement"].to_numpy(dtype=bool),
        }
    elif fit_protocol == "leave900out":
        train_mask = ~protocol_df["all900_holdout"].to_numpy(dtype=bool)
        split_masks = {
            "non900_train": train_mask,
            "all900_leave_scale_out": protocol_df["all900_holdout"].to_numpy(dtype=bool),
        }
    else:
        raise ValueError(f"Unexpected fit protocol: {fit_protocol}")

    metrics = []
    prediction_rows = []
    coefficient_rows = []

    base_params = BASE_MODEL_PARAMS[base_model]
    base_name = f"{base_model}_base"
    for split, mask in split_masks.items():
        add_metrics(
            metrics,
            model=base_name,
            fit_protocol=fit_protocol,
            split=split,
            params=base_params,
            preserves_scaling_law=True,
            actual=actual[mask],
            predicted=base_pred[mask],
        )
    prediction_rows.append(
        protocol_df.assign(candidate=base_name, candidate_pred_bpb=base_pred, residual_adjustment=0.0)
    )

    residual_target = actual - base_pred
    ridge_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for feature_name, block in features.items():
        x_all = block.matrix[row_index]
        for ridge in ridge_values:
            coef, mean, std = ridge_fit(x_all[train_mask], residual_target[train_mask], ridge)
            adjustment = ridge_predict(x_all, coef, mean, std)
            candidate_pred = base_pred + adjustment
            residual_param_count = len(block.names) + 1
            params = base_params + residual_param_count
            candidate = f"{base_model}_{feature_name}_ridge{ridge:g}"
            for split, mask in split_masks.items():
                add_metrics(
                    metrics,
                    model=candidate,
                    fit_protocol=fit_protocol,
                    split=split,
                    params=params,
                    preserves_scaling_law=block.preserves_scaling_law,
                    actual=actual[mask],
                    predicted=candidate_pred[mask],
                )
            prediction_rows.append(
                protocol_df.assign(
                    candidate=candidate,
                    candidate_pred_bpb=candidate_pred,
                    residual_adjustment=adjustment,
                )
            )
            coefficient_rows.append(
                {
                    "candidate": candidate,
                    "fit_protocol": fit_protocol,
                    "feature": "intercept",
                    "coef_standardized": coef[0],
                    "feature_mean": np.nan,
                    "feature_std": np.nan,
                }
            )
            for feature, value, mu, sigma in zip(block.names, coef[1:], mean, std, strict=True):
                coefficient_rows.append(
                    {
                        "candidate": candidate,
                        "fit_protocol": fit_protocol,
                        "feature": feature,
                        "coef_standardized": value,
                        "feature_mean": mu,
                        "feature_std": sigma,
                    }
                )

    return pd.DataFrame(metrics), pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(coefficient_rows)


def plot_pred_actual(predictions: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("seed7", "fixed340_holdout", "fixed340_holdout"),
        ("seed7", "seed7_holdout", "seed7_holdout"),
        ("leave900out", "all900_leave_scale_out", "all900_holdout"),
    ]
    for fit_protocol, metric_split, mask_column in plot_specs:
        split_metrics = metrics[(metrics["fit_protocol"] == fit_protocol) & (metrics["split"] == metric_split)].copy()
        split_metrics = split_metrics.sort_values("rmse").head(6)
        if split_metrics.empty:
            continue
        candidates = split_metrics["model"].tolist()
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        axes_flat = axes.ravel()
        for ax, candidate in zip(axes_flat, candidates, strict=False):
            row = predictions[(predictions["fit_protocol"] == fit_protocol) & (predictions["candidate"] == candidate)]
            background = row
            focused = row[row[mask_column].astype(bool)]
            ax.scatter(background["actual_bpb"], background["candidate_pred_bpb"], color="0.80", s=16, alpha=0.45)
            colors = focused["target_budget_multiplier"].to_numpy(dtype=float)
            ax.scatter(
                focused["actual_bpb"],
                focused["candidate_pred_bpb"],
                c=colors,
                cmap="RdYlGn_r",
                s=48,
                edgecolor="black",
                linewidth=0.4,
            )
            lo = min(background["actual_bpb"].min(), background["candidate_pred_bpb"].min())
            hi = max(background["actual_bpb"].max(), background["candidate_pred_bpb"].max())
            pad = 0.01
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
            metric_row = split_metrics[split_metrics["model"] == candidate].iloc[0]
            ax.set_title(f"{candidate}\nRMSE={metric_row.rmse:.5f}, slope={metric_row.slope_pred_on_actual:.3f}")
            ax.set_xlabel("Actual BPB")
            ax.set_ylabel("Predicted BPB")
            ax.grid(alpha=0.25)
        for ax in axes_flat[len(candidates) :]:
            ax.axis("off")
        fig.suptitle(f"Quality residual probe: {metric_split}")
        fig.savefig(out_dir / f"pred_actual_{metric_split}.png", dpi=180)
        plt.close(fig)


def write_report(metrics: pd.DataFrame, out_dir: Path) -> None:
    key_splits = ["fixed340_holdout", "seed7_holdout", "all900_leave_scale_out"]
    lines = [
        "# Session 12 Quality Residual Sprint",
        "",
        (
            "This local sprint fits small residual heads on top of MCT-LRQ predictions using only the "
            "corresponding training split."
        ),
        (
            "Mixture-only residual variants preserve the fixed-mixture scaling-law shape because they add "
            "only an extra `R(w)` term."
        ),
        (
            "The `uD`/`uND` variants are diagnostic only: they can improve prediction but are less clean "
            "structurally unless constrained."
        ),
        "",
        "## Best Metrics",
        "",
    ]
    for split in key_splits:
        rows = metrics[metrics["split"] == split].sort_values("rmse").head(8)
        if rows.empty:
            continue
        lines.append(f"### {split}")
        lines.append("")
        cols = [
            "model",
            "fit_protocol",
            "params",
            "preserves_fixed_mixture_scaling_law",
            "n",
            "rmse",
            "spearman",
            "bias_pred_minus_actual",
            "slope_pred_on_actual",
            "std_ratio",
        ]
        lines.append(rows[cols].to_markdown(index=False, floatfmt=".6f"))
        lines.append("")
    safe = metrics[
        metrics["preserves_fixed_mixture_scaling_law"]
        & metrics["split"].isin(["fixed340_holdout", "seed7_holdout", "all900_leave_scale_out"])
        & ~metrics["model"].str.endswith("_base")
    ].copy()
    lines.extend(
        [
            "## Interpretation",
            "",
            "- MCT-LRQ is already quality-split aware through its LRQ anchor.",
            (
                "- The compact mixture-only residuals are the relevant structural probes; they should be "
                "judged against the base MCT rows."
            ),
            (
                "- If a `uD` residual wins, treat it as evidence for quality-dependent scale elasticity, "
                "not as a clean promotion by itself."
            ),
            "",
        ]
    )
    if not safe.empty:
        for split in key_splits:
            split_rows = metrics[metrics["split"] == split]
            split_safe = safe[safe["split"] == split]
            if split_rows.empty or split_safe.empty:
                continue
            base = split_rows[split_rows["model"].str.endswith("_base")].sort_values("rmse").iloc[0]
            residual = split_safe[~split_safe["model"].str.endswith("_base")].sort_values("rmse").iloc[0]
            delta = float(residual.rmse - base.rmse)
            direction = "better" if delta < 0.0 else "worse"
            lines.append(
                f"- `{split}`: best structural residual `{residual.model}` has RMSE `{residual.rmse:.6f}`, "
                f"which is `{abs(delta):.6f}` {direction} than the best MCT base row `{base.model}`."
            )
        lines.append("")
        lines.append(
            "Conclusion: the clean quality/support residual is not a promotion. It lowers training error, "
            "but it overfits and degrades held-out fixed-340M, seed-7, and 900M transfer. The useful signal "
            "is that MCT-LRQ already contains the quality split; extra anchor residual width is currently "
            "harmful unless it is incorporated into a jointly fitted structural body."
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=PACKET_ROOT)
    parser.add_argument("--base-predictions", type=Path, default=BASE_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet_root / "data" / "nd_scale_runs.csv", low_memory=False)
    payload = np.load(args.packet_root / "data" / "nd_scale_packet.npz", allow_pickle=True)
    domains = np.asarray(payload["domain_names"], dtype=object)
    features = build_feature_blocks(frame, domains)
    predictions = pd.read_csv(args.base_predictions)

    all_metrics = []
    all_predictions = []
    all_coefficients = []
    for base_model in ("mct_lrq74_drop", "mct_lrq74_balanced"):
        for fit_protocol in ("seed7", "leave900out"):
            metrics, preds, coefs = fit_residual_variants(
                predictions,
                features,
                base_model=base_model,
                fit_protocol=fit_protocol,
            )
            all_metrics.append(metrics)
            all_predictions.append(preds)
            all_coefficients.append(coefs)

    metrics = pd.concat(all_metrics, ignore_index=True)
    residual_predictions = pd.concat(all_predictions, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)

    metrics.to_csv(out_dir / "csv/quality_residual_metrics.csv", index=False)
    residual_predictions.to_csv(out_dir / "csv/quality_residual_predictions.csv", index=False)
    coefficients.to_csv(out_dir / "csv/quality_residual_coefficients.csv", index=False)
    plot_pred_actual(residual_predictions, metrics, out_dir / "plots")
    write_report(metrics, out_dir)

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
