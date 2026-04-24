#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "jax", "jaxlib", "matplotlib", "scikit-learn"]
# ///
"""Evaluate compact q/support on refreshed registry data with 1.2B held out."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"text.usetex": False})


MODEL_NAME = "qsupport_floor_ci_rpm50_gap015_1p2b_holdout"
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
RUN_NAME_HOLDOUTS = {"baseline_proportional", "baseline_unimax"}
SCALE_ORDER = ("60m_1p2b", "130m_2p6b", "300m_6b", "520m_10p4b", "1_2b_24b")


@dataclass(frozen=True)
class PatchSummary:
    packet_primary_rows_before: int
    packet_primary_rows_after: int
    updated_primary_labels: int
    newly_enabled_primary_labels: int
    registry_rows_matched: int


@dataclass(frozen=True)
class WeightSourceSummary:
    mismatched_rows_after_load: int
    max_phase_tv_after_load: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_packet = Path(__file__).resolve().parent / "chatgpt_pro_hybrid_data_mixing_packet_v28"
    parser.add_argument("--packet-root", type=Path, default=default_packet)
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "run_registry" / "strong_tier_perplexity_ready.csv",
    )
    parser.add_argument(
        "--session7-code-dir",
        type=Path,
        default=Path("/Users/calvinxu/Downloads/chatgpt_pro_session_7/4"),
        help="Directory containing search_qsupport_floor_candidates_20260423.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "chatgpt_pro_hybrid_data_mixing_packet_v28"
        / "reference_outputs"
        / "qsupport_1p2b_holdout_20260423",
    )
    parser.add_argument("--fit-maxiter", type=int, default=50)
    return parser.parse_args()


def ensure_import_paths(packet_root: Path, session7_code_dir: Path) -> None:
    for path in (packet_root / "code", session7_code_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required code directory not found: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def registry_key(row: pd.Series) -> tuple[str, str, str, str, float]:
    return (
        str(row["scale"]),
        str(row["study_path"]),
        str(row["source_experiment"]),
        str(row["run_name"]),
        float(row["target_budget_multiplier"]),
    )


def packet_key(row: pd.Series) -> tuple[str, str, str, str, float]:
    return (
        str(row["scale"]),
        str(row["path"]),
        str(row["source_experiment"]),
        str(row["run_name"]),
        float(row["target_budget_multiplier"]),
    )


def patch_packet_from_registry(benchmark: Any, registry_csv: Path) -> PatchSummary:
    packet = benchmark.packet
    frame = packet.frame
    before_count = int(packet.primary_mask.sum())
    registry = pd.read_csv(registry_csv)
    ready = registry.loc[
        registry["is_perplexity_ready"].astype(bool) & registry["objective_metric"].eq(PRIMARY_METRIC)
    ].copy()
    ready_lookup = {registry_key(row): row for _, row in ready.iterrows()}

    updated = 0
    newly_enabled = 0
    matched = 0
    for packet_index, packet_row in frame.iterrows():
        row = ready_lookup.get(packet_key(packet_row))
        if row is None:
            continue
        matched += 1
        was_primary = bool(packet.primary_mask[packet_index])
        new_value = float(row["objective_metric_value"])
        old_value = float(packet.primary_y[packet_index]) if was_primary else np.nan
        packet.primary_y[packet_index] = new_value
        packet.primary_mask[packet_index] = True
        if PRIMARY_METRIC in frame.columns:
            frame.at[packet_index, PRIMARY_METRIC] = new_value
        frame.at[packet_index, "objective_metric_value"] = new_value
        if "is_perplexity_ready" in frame.columns:
            frame.at[packet_index, "is_perplexity_ready"] = True
        if "has_primary_label" in frame.columns:
            frame.at[packet_index, "has_primary_label"] = True
        if not was_primary:
            newly_enabled += 1
        elif not np.isclose(old_value, new_value, rtol=0.0, atol=1e-12):
            updated += 1

    return PatchSummary(
        packet_primary_rows_before=before_count,
        packet_primary_rows_after=int(packet.primary_mask.sum()),
        updated_primary_labels=updated,
        newly_enabled_primary_labels=newly_enabled,
        registry_rows_matched=matched,
    )


def summarize_packet_weight_source(benchmark: Any) -> WeightSourceSummary:
    packet = benchmark.packet
    frame = packet.frame
    reconstructed = np.zeros_like(packet.weights, dtype=float)
    for phase_index in (0, 1):
        columns = [f"phase_{phase_index}_{domain_name}" for domain_name in packet.domain_names.astype(str)]
        phase_weights = frame[columns].fillna(0.0).to_numpy(dtype=float)
        phase_weights /= phase_weights.sum(axis=1)[:, None]
        reconstructed[:, phase_index, :] = phase_weights
    tv = 0.5 * np.abs(np.asarray(packet.weights, dtype=float) - reconstructed).sum(axis=2)
    return WeightSourceSummary(
        mismatched_rows_after_load=int((tv > 1e-8).any(axis=1).sum()),
        max_phase_tv_after_load=float(tv.max()),
    )


def qsupport_config(module: Any, fit_maxiter: int) -> Any:
    return module.CandidateConfig(
        name=MODEL_NAME,
        title=f"Compact q/support floor, 1.2B holdout, maxiter={fit_maxiter}",
        variant="centered_interactions",
        alpha=0.3,
        prior=1e-4,
        residual_penalty_mult=100.0,
        gap_max=0.15,
        fit_maxiter=int(fit_maxiter),
        mix_opt_maxiter=0,
    )


def metric_bundle(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    actual_arr = actual_arr[mask]
    pred_arr = pred_arr[mask]
    if len(actual_arr) == 0:
        return {"rows": 0}
    residual = pred_arr - actual_arr
    actual_std = float(np.std(actual_arr))
    pred_std = float(np.std(pred_arr))
    if len(actual_arr) >= 2 and actual_std > 0.0:
        slope, intercept = np.polyfit(actual_arr, pred_arr, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "rows": len(actual_arr),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_residual": float(np.mean(residual)),
        "slope": float(slope),
        "intercept": float(intercept),
        "actual_std": actual_std,
        "pred_std": pred_std,
        "std_ratio": float(pred_std / actual_std) if actual_std > 0.0 else None,
    }


def prediction_frame(
    benchmark: Any,
    model: dict[str, Any],
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> pd.DataFrame:
    packet = benchmark.packet
    primary_indices = np.flatnonzero(packet.primary_mask)
    pred = benchmark.predict_indices(model, primary_indices)
    frame = packet.frame.iloc[primary_indices].copy()
    frame["target_index"] = primary_indices
    frame["actual_bpb"] = packet.primary_y[primary_indices].astype(float)
    frame["predicted_bpb"] = pred.astype(float)
    frame["residual"] = frame["predicted_bpb"] - frame["actual_bpb"]
    train_set = set(np.asarray(train_indices, dtype=int).tolist())
    holdout_set = set(np.asarray(holdout_indices, dtype=int).tolist())
    frame["eval_role"] = [
        "validation_1p2b" if int(idx) in holdout_set else "train" if int(idx) in train_set else "unused"
        for idx in frame["target_index"].to_numpy(int)
    ]
    return frame


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "train_all": predictions["eval_role"].eq("train"),
        "validation_1p2b": predictions["eval_role"].eq("validation_1p2b"),
        "all_primary": predictions["eval_role"].isin(["train", "validation_1p2b"]),
    }
    for scale in sorted(predictions["scale"].astype(str).unique()):
        subsets[f"train_scale_{scale}"] = predictions["eval_role"].eq("train") & predictions["scale"].eq(scale)
    for name, mask in subsets.items():
        subset = predictions.loc[mask].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "subset": name,
                **metric_bundle(
                    subset["actual_bpb"].to_numpy(float),
                    subset["predicted_bpb"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_predicted_vs_actual(predictions: pd.DataFrame, out_path: Path) -> None:
    train = predictions.loc[predictions["eval_role"].eq("train")]
    holdout = predictions.loc[predictions["eval_role"].eq("validation_1p2b")]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.scatter(
        train["actual_bpb"],
        train["predicted_bpb"],
        s=22,
        color="0.70",
        alpha=0.35,
        edgecolors="none",
        label=f"training primary rows ({len(train)})",
    )
    cmap = plt.get_cmap("RdYlGn_r")
    colors = {name: cmap(i / max(1, len(RUN_NAME_HOLDOUTS) - 1)) for i, name in enumerate(sorted(RUN_NAME_HOLDOUTS))}
    for run_name, group in holdout.groupby("run_name", sort=True):
        ax.scatter(
            group["actual_bpb"],
            group["predicted_bpb"],
            s=115,
            color=colors.get(str(run_name), cmap(0.5)),
            edgecolors="black",
            linewidths=0.8,
            label=f"1.2B holdout: {run_name}",
            zorder=5,
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(row["run_name"]).replace("baseline_", ""),
                (float(row["actual_bpb"]), float(row["predicted_bpb"])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )
    low = float(min(predictions["actual_bpb"].min(), predictions["predicted_bpb"].min()) - 0.01)
    high = float(max(predictions["actual_bpb"].max(), predictions["predicted_bpb"].max()) + 0.01)
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0)
    metrics = summarize_predictions(predictions)
    holdout_metrics = metrics.loc[metrics["subset"].eq("validation_1p2b")]
    if not holdout_metrics.empty:
        row = holdout_metrics.iloc[0]
        ax.text(
            0.02,
            0.98,
            f"1.2B holdout RMSE={row['rmse']:.4f}\nmean residual={row['mean_residual']:+.4f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual BPB")
    ax.set_ylabel("Predicted BPB")
    ax.set_title("Compact q/support 1.2B holdout: predicted vs actual BPB")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_baseline_trajectories(predictions: pd.DataFrame, out_path: Path) -> None:
    subset = predictions.loc[predictions["run_name"].isin(RUN_NAME_HOLDOUTS)].copy()
    subset["scale_order"] = subset["scale"].astype(str).map({scale: i for i, scale in enumerate(SCALE_ORDER)})
    subset = subset.sort_values(["run_name", "scale_order", "target_budget_multiplier"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    cmap = plt.get_cmap("RdYlGn_r")
    for ax, run_name in zip(axes, sorted(RUN_NAME_HOLDOUTS), strict=False):
        group = subset.loc[subset["run_name"].eq(run_name)]
        labels = [
            f"{scale}\n{mult:g}x"
            for scale, mult in zip(
                group["scale"].astype(str),
                group["target_budget_multiplier"].astype(float),
                strict=False,
            )
        ]
        x = np.arange(len(group), dtype=float)
        ax.plot(x, group["actual_bpb"], marker="o", color="black", label="actual")
        ax.plot(x, group["predicted_bpb"], marker="s", color=cmap(0.82), label="predicted")
        is_holdout = group["eval_role"].eq("validation_1p2b").to_numpy(bool)
        ax.scatter(
            x[is_holdout],
            group.loc[is_holdout, "actual_bpb"],
            s=95,
            color=cmap(0.1),
            edgecolors="black",
            zorder=5,
            label="1.2B actual holdout",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(run_name)
        ax.set_xlabel("scale / target-budget multiplier")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("BPB")
    axes[0].legend(loc="best")
    fig.suptitle("Baseline mixture scale trajectories after 1.2B holdout fit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ensure_import_paths(args.packet_root, args.session7_code_dir)
    qsupport_module = importlib.import_module("search_qsupport_floor_candidates_20260423")
    benchmark = qsupport_module.QSupportFloorBenchmark(args.packet_root)
    patch_summary = patch_packet_from_registry(benchmark, args.registry_csv)
    weight_source_summary = summarize_packet_weight_source(benchmark)

    packet = benchmark.packet
    frame = packet.frame
    primary_indices = np.flatnonzero(packet.primary_mask)
    holdout_mask = (
        frame["scale"].eq("1_2b_24b").to_numpy(bool)
        & frame["path"].eq("qsplit_baselines3_holdout").to_numpy(bool)
        & frame["run_name"].isin(RUN_NAME_HOLDOUTS).to_numpy(bool)
        & packet.primary_mask
    )
    holdout_indices = np.flatnonzero(holdout_mask)
    if len(holdout_indices) == 0:
        raise ValueError("No 1.2B primary holdout rows found after registry patch.")
    train_indices = np.asarray([idx for idx in primary_indices if idx not in set(holdout_indices)], dtype=int)

    config = qsupport_config(qsupport_module, args.fit_maxiter)
    model = benchmark.fit(train_indices=train_indices, config=config)
    predictions = prediction_frame(benchmark, model, train_indices, holdout_indices)
    metrics = summarize_predictions(predictions)

    predictions.to_csv(args.out_dir / "qsupport_1p2b_holdout_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "qsupport_1p2b_holdout_metrics.csv", index=False)
    with (args.out_dir / "qsupport_1p2b_holdout_model.json").open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, default=lambda obj: np.asarray(obj).tolist())
    with (args.out_dir / "qsupport_1p2b_holdout_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "fit_maxiter": int(args.fit_maxiter),
                "patch_summary": asdict(patch_summary),
                "weight_source_summary": asdict(weight_source_summary),
                "train_rows": len(train_indices),
                "holdout_rows": len(holdout_indices),
                "holdout_run_names": sorted(
                    predictions.loc[predictions["eval_role"].eq("validation_1p2b"), "run_name"].astype(str).tolist()
                ),
                "metrics": metrics.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    plot_predicted_vs_actual(
        predictions,
        figures_dir / "qsupport_1p2b_holdout_predicted_vs_actual.png",
    )
    plot_baseline_trajectories(
        predictions,
        figures_dir / "qsupport_1p2b_holdout_baseline_trajectories.png",
    )

    print(json.dumps(asdict(patch_summary), indent=2))
    print(json.dumps(asdict(weight_source_summary), indent=2))
    print(metrics.to_string(index=False))
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
