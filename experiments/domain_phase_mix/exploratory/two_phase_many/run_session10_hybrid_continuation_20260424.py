#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Test Session 1-style continuation corrections on Session 2 structural laws."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

SESSION2_SCRIPT = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/session10_candidate_revalidation_20260424"
    / "session2_structural/run_joint_mixture_scale_law.py"
)
DEFAULT_PACKET_ROOT = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v30"
)
DEFAULT_OUT_DIR = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/session10_candidate_revalidation_20260424"
    / "hybrid_continuation"
)


@dataclass(frozen=True)
class CorrectionSpec:
    """A nonnegative mixture-conditioned target-budget continuation correction."""

    name: str
    feature_mode: str
    beta_ref: float
    ridge: float
    pair_weight: float


@dataclass(frozen=True)
class CorrectedModel:
    """Base structural law plus a nonnegative `mu^-beta - 1` residual head."""

    name: str
    base_name: str
    spec: CorrectionSpec
    coef: np.ndarray
    base_model: object
    feature_factory: object
    n0: float

    @property
    def param_count(self) -> int:
        return int(self.base_model.total_constant_count + len(self.coef))

    def correction_features(self, weights: np.ndarray, model_sizes: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
        beta = self.spec.beta_ref * np.ones_like(np.asarray(multipliers, dtype=float))
        hmu = np.exp(-beta * np.log(np.asarray(multipliers, dtype=float))) - 1.0
        cfeat = self.feature_factory.scale_head_features(weights, self.spec.feature_mode)
        return hmu[:, None] * cfeat

    def predict_all(self, data: object) -> np.ndarray:
        pred = self.base_model.predict_all()
        features = self.correction_features(data.W, data.N, data.mu)
        return pred + features @ self.coef


def load_module(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("session2_structural_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["session2_structural_module"] = module
    spec.loader.exec_module(module)
    return module


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid**2)))
    bias = float(np.mean(resid))
    actual_std = float(np.std(y_true, ddof=0))
    pred_std = float(np.std(y_pred, ddof=0))
    if len(y_true) > 1 and actual_std > 0 and pred_std > 0:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        spearman = float(spearmanr(y_true, y_pred).statistic)
        std_ratio = float(pred_std / actual_std)
    else:
        slope = intercept = spearman = std_ratio = float("nan")
    return {
        "rows": len(y_true),
        "rmse": rmse,
        "bias": bias,
        "spearman": spearman,
        "slope": float(slope),
        "intercept": float(intercept),
        "actual_std": actual_std,
        "pred_std": pred_std,
        "std_ratio": std_ratio,
    }


def fit_session2_compact(module: object, data: object, feature_factory: object, train_mask: np.ndarray) -> object:
    return module.TwoStagePowerLaw(
        "compact_grp_famsqrt_b025",
        data,
        feature_factory,
        "grp_famsqrt",
        (0.20, 0.25, 0.30, 0.65),
        pair_weight=6.0,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        head_A="constant",
        head_B="family",
        head_C="constant",
    ).fit(train_mask)


def fit_correction(
    data: object,
    feature_factory: object,
    base_model: object,
    train_mask: np.ndarray,
    spec: CorrectionSpec,
) -> CorrectedModel:
    base_pred = np.asarray(base_model.predict_all(), dtype=float)
    features = CorrectedModel(
        name=spec.name,
        base_name=base_model.model_id,
        spec=spec,
        coef=np.zeros(1),
        base_model=base_model,
        feature_factory=feature_factory,
        n0=float(data.N0),
    ).correction_features(data.W, data.N, data.mu)

    train_idx = np.flatnonzero(train_mask & data.all900_train)
    rows = [features[train_idx]]
    targets = [(data.y - base_pred)[train_idx]]

    pair_rows: list[np.ndarray] = []
    pair_targets: list[float] = []
    frame = pd.DataFrame(
        {
            "mixture_id": data.mixture_ids[train_idx].astype(str),
            "N": data.N[train_idx].astype(float),
            "idx": train_idx,
        }
    )
    for _, group in frame.groupby(["mixture_id", "N"], sort=False):
        idx = group["idx"].to_numpy(dtype=int)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(data.mu[idx])]
        for left in range(len(idx)):
            for right in range(left + 1, len(idx)):
                i = int(idx[left])
                j = int(idx[right])
                pair_rows.append(features[i] - features[j])
                pair_targets.append((data.y[i] - base_pred[i]) - (data.y[j] - base_pred[j]))
    if pair_rows:
        rows.append(np.sqrt(spec.pair_weight) * np.vstack(pair_rows))
        targets.append(np.sqrt(spec.pair_weight) * np.asarray(pair_targets, dtype=float))

    rows.append(np.sqrt(spec.ridge) * np.eye(features.shape[1]))
    targets.append(np.zeros(features.shape[1], dtype=float))
    result = lsq_linear(
        np.vstack(rows),
        np.concatenate(targets),
        bounds=(np.zeros(features.shape[1]), np.inf * np.ones(features.shape[1])),
        tol=1e-10,
        max_iter=1000,
    )
    return CorrectedModel(spec.name, base_model.model_id, spec, result.x, base_model, feature_factory, float(data.N0))


def evaluate_models(data: object, models: dict[str, object], predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    subsets = {
        "seed7_holdout": data.seed7_holdout,
        "fixed_340m_10p4b": data.fixed340,
        "random_supplement": data.random_supplement,
        "all_900m_24b": data.all900_holdout,
    }
    rows = []
    for name, pred in predictions.items():
        params = getattr(models[name], "param_count", getattr(models[name], "total_constant_count", np.nan))
        for subset, mask in subsets.items():
            row = metric_dict(data.y[mask], pred[mask])
            row.update(model=name, subset=subset, params=int(params))
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_drop_summary(data: object, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fixed_idx = np.flatnonzero(data.fixed340)
    frame = pd.DataFrame(
        {
            "mixture_id": data.mixture_ids[fixed_idx].astype(str),
            "idx": fixed_idx,
            "mu": data.mu[fixed_idx],
        }
    )
    for name, pred in predictions.items():
        for mixture_id, group in frame.groupby("mixture_id", sort=False):
            by_mu = {float(row.mu): int(row.idx) for row in group.itertuples()}
            for start, end, label in [(0.5, 1.0, "0.5->1"), (0.5, 2.0, "0.5->2"), (1.0, 2.0, "1->2")]:
                if start not in by_mu or end not in by_mu:
                    continue
                i = by_mu[start]
                j = by_mu[end]
                actual_drop = float(data.y[i] - data.y[j])
                predicted_drop = float(pred[i] - pred[j])
                rows.append(
                    {
                        "model": name,
                        "mixture_id": mixture_id,
                        "drop": label,
                        "actual_drop": actual_drop,
                        "predicted_drop": predicted_drop,
                        "drop_error": predicted_drop - actual_drop,
                        "drop_ratio": predicted_drop / actual_drop if actual_drop else np.nan,
                    }
                )
    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["model", "drop"], sort=False)
        .agg(
            n=("mixture_id", "size"),
            actual_drop=("actual_drop", "mean"),
            predicted_drop=("predicted_drop", "mean"),
            drop_ratio=("drop_ratio", "mean"),
            drop_rmse=("drop_error", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))),
        )
        .reset_index()
    )
    return details, summary


def prediction_frame(data: object, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "registry_run_key": data.registry_keys.astype(str),
            "mixture_id": data.mixture_ids.astype(str),
            "scale": data.scale_names_by_row.astype(str),
            "mu": data.mu.astype(float),
            "actual": data.y.astype(float),
            "seed7_holdout": data.seed7_holdout,
            "fixed_340m_10p4b": data.fixed340,
            "all_900m_24b": data.all900_holdout,
        }
    )
    for name, pred in predictions.items():
        frame[name] = pred
        frame[f"{name}_residual"] = pred - data.y
    return frame


def plot_fixed_predictions(predictions: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    models = [
        "s2_compact_base",
        "s2_compact_mu_const_b0p8",
        "s2_compact_mu_family_b0p8",
        "s2_compact_mu_domain_sqrt_b0p8",
    ]
    fixed = predictions[predictions["fixed_340m_10p4b"]].copy()
    fig, axes = plt.subplots(1, len(models), figsize=(18, 4.5), constrained_layout=True)
    lo = fixed[["actual", *models]].to_numpy(dtype=float).min() - 0.006
    hi = fixed[["actual", *models]].to_numpy(dtype=float).max() + 0.006
    cmap = plt.get_cmap("RdYlGn_r")
    for ax, model in zip(axes, models, strict=False):
        row = metrics[(metrics["model"] == model) & (metrics["subset"] == "fixed_340m_10p4b")].iloc[0]
        scatter = ax.scatter(
            fixed["actual"],
            fixed[model],
            c=fixed["mu"],
            cmap=cmap,
            edgecolor="black",
            linewidth=0.4,
            s=45,
        )
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        ax.set_title(f"{model}\nRMSE={row.rmse:.4f}, slope={row.slope:.3f}")
        ax.set_xlabel("Actual BPB")
        ax.set_ylabel("Predicted BPB")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="target-budget multiplier")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    module = load_module(SESSION2_SCRIPT)
    data = module.load_packet(args.packet_root)
    feature_factory = module.FeatureFactory(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_seed7 = fit_session2_compact(module, data, feature_factory, data.seed7_train)
    base_all900 = fit_session2_compact(module, data, feature_factory, data.all900_train)

    specs = [
        CorrectionSpec("s2_compact_mu_const_b0p8", "constant", 0.8, 1e-4, 4.0),
        CorrectionSpec("s2_compact_mu_family_b0p4", "family", 0.4, 1e-4, 4.0),
        CorrectionSpec("s2_compact_mu_family_b0p8", "family", 0.8, 1e-4, 4.0),
        CorrectionSpec("s2_compact_mu_family_b1p2", "family", 1.2, 1e-4, 4.0),
        CorrectionSpec("s2_compact_mu_domain_sqrt_b0p8", "domain_sqrt", 0.8, 1e-3, 4.0),
    ]

    models: dict[str, object] = {
        "s2_compact_base": base_seed7,
    }
    predictions: dict[str, np.ndarray] = {
        "s2_compact_base": np.asarray(base_seed7.predict_all(), dtype=float),
    }

    for spec in specs:
        corrected = fit_correction(data, feature_factory, base_seed7, data.seed7_train, spec)
        models[corrected.name] = corrected
        predictions[corrected.name] = corrected.predict_all(data)

    metrics = evaluate_models(data, models, predictions)
    drop_details, drop_summary = fixed_drop_summary(data, predictions)
    pred_frame = prediction_frame(data, predictions)

    # Official all-900 protocol uses a separate all-non-900M refit. Corrections
    # are anchored to mu=1, so they are zero on these 900M rows; still report the
    # refit base metric explicitly to avoid seed-7 leakage.
    all900_base_pred = np.asarray(base_all900.predict_all(), dtype=float)
    all900_rows = []
    all900_metric = metric_dict(data.y[data.all900_holdout], all900_base_pred[data.all900_holdout])
    all900_metric.update(
        model="s2_compact_base_all_non900_refit", subset="all_900m_24b", params=base_all900.total_constant_count
    )
    all900_rows.append(all900_metric)
    pd.DataFrame(all900_rows).to_csv(args.out_dir / "all900_protocol_metrics.csv", index=False)

    metrics.to_csv(args.out_dir / "hybrid_metrics_seed7_fit.csv", index=False)
    drop_details.to_csv(args.out_dir / "hybrid_fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(args.out_dir / "hybrid_fixed340_drop_summary.csv", index=False)
    pred_frame.to_csv(args.out_dir / "hybrid_predictions_seed7_fit.csv", index=False)
    plot_fixed_predictions(pred_frame, metrics, args.out_dir / "hybrid_fixed340_pred_vs_actual.png")

    print(metrics[metrics["subset"].isin(["fixed_340m_10p4b", "all_900m_24b"])].to_string(index=False))
    print()
    print(drop_summary.to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
