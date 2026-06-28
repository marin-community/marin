#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "jax", "jaxlib", "matplotlib", "scikit-learn"]
# ///
"""Evaluate compact q/support after repairing packet weights and scale metadata."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"text.usetex": False})


MODEL_NAME = "qsupport_floor_ci_rpm100_gap015_corrected"
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
RUN_NAME_HOLDOUTS = {"baseline_proportional", "baseline_unimax"}
SCALE_ORDER = ("130m_2p6b", "60m_1p2b", "300m_6b", "520m_10p4b", "1_2b_24b")
NOMINAL_PARAMS = {
    "60m_1p2b": 60_000_000.0,
    "130m_2p6b": 130_000_000.0,
    "300m_6b": 300_000_000.0,
    "520m_10p4b": 520_000_000.0,
    "1_2b_24b": 1_200_000_000.0,
}
ACTUAL_NON_EMBEDDING_PARAMS = {
    "60m_1p2b": 58_998_528.0,
    "130m_2p6b": 22_813_184.0,
    "300m_6b": 102_648_576.0,
    "520m_10p4b": 339_788_800.0,
    "1_2b_24b": 906_037_248.0,
}
TIED_TOTAL_PARAMS = {
    "60m_1p2b": 157_499_136.0,
    "130m_2p6b": 88_480_256.0,
    "300m_6b": 201_149_184.0,
    "520m_10p4b": 471_122_944.0,
    "1_2b_24b": 1_168_705_536.0,
}
PARAM_AXES = {
    "nominal": NOMINAL_PARAMS,
    "non_embedding": ACTUAL_NON_EMBEDDING_PARAMS,
    "tied_total": TIED_TOTAL_PARAMS,
}


@dataclass(frozen=True)
class EvalBundle:
    model: dict[str, Any]
    predictions: pd.DataFrame
    metrics: pd.DataFrame


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
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_packet / "reference_outputs" / "qsupport_corrected_data_20260423",
    )
    parser.add_argument("--fit-maxiter", type=int, default=25)
    parser.add_argument("--scale-axis", choices=sorted(PARAM_AXES), default="non_embedding")
    return parser.parse_args()


def ensure_import_paths(packet_root: Path, session7_code_dir: Path) -> None:
    for path in (packet_root / "code", session7_code_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required code directory not found: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def apply_registry_labels(benchmark: Any, registry_csv: Path) -> dict[str, int]:
    packet = benchmark.packet
    frame = packet.frame
    registry = pd.read_csv(registry_csv)
    ready = registry.loc[
        registry["is_perplexity_ready"].astype(bool) & registry["objective_metric"].eq(PRIMARY_METRIC)
    ].copy()
    ready_lookup = {
        (
            str(row["scale"]),
            str(row["study_path"]),
            str(row["source_experiment"]),
            str(row["run_name"]),
            float(row["target_budget_multiplier"]),
        ): row
        for _, row in ready.iterrows()
    }
    updated = 0
    enabled = 0
    matched = 0
    for packet_index, packet_row in frame.iterrows():
        key = (
            str(packet_row["scale"]),
            str(packet_row["path"]),
            str(packet_row["source_experiment"]),
            str(packet_row["run_name"]),
            float(packet_row["target_budget_multiplier"]),
        )
        row = ready_lookup.get(key)
        if row is None:
            continue
        matched += 1
        was_primary = bool(packet.primary_mask[packet_index])
        old_value = float(packet.primary_y[packet_index]) if was_primary else np.nan
        new_value = float(row["objective_metric_value"])
        packet.primary_y[packet_index] = new_value
        packet.primary_mask[packet_index] = True
        frame.at[packet_index, PRIMARY_METRIC] = new_value
        frame.at[packet_index, "objective_metric_value"] = new_value
        if "is_perplexity_ready" in frame.columns:
            frame.at[packet_index, "is_perplexity_ready"] = True
        if "has_primary_label" in frame.columns:
            frame.at[packet_index, "has_primary_label"] = True
        if not was_primary:
            enabled += 1
        elif not np.isclose(old_value, new_value, rtol=0.0, atol=1e-12):
            updated += 1
    return {"registry_rows_matched": matched, "updated_primary_labels": updated, "newly_enabled_labels": enabled}


def apply_param_axis(benchmark: Any, scale_axis: str) -> dict[str, float | str]:
    law = benchmark.law
    packet = law.packet
    param_axis = PARAM_AXES[scale_axis]
    scale_labels = packet.scale_labels.astype(str)
    params = np.asarray([param_axis[label] for label in scale_labels], dtype=float)
    object.__setattr__(packet, "model_sizes", params.astype(np.int64))
    law.logN = np.log(params)
    match = law.subset_idx(
        scales=["130m_2p6b", "300m_6b", "520m_10p4b"],
        paths=["qsplit_representative12", "stratified"],
        fit_roles=["fit_region"],
        metric="primary",
    )
    law.muN = float(np.mean(law.logN[match]))
    law.stdN = float(np.std(law.logN[match])) or 1.0
    law.uN = (law.logN - law.muN) / law.stdN
    benchmark.base.u_n = np.asarray(law.uN, dtype=float)
    benchmark.base.u_d = np.asarray(law.uD, dtype=float)
    return {
        "scale_axis": scale_axis,
        "mu_log_params": law.muN,
        "std_log_params": law.stdN,
        **{f"{scale_axis}_params_{key}": value for key, value in param_axis.items()},
    }


def qsupport_config(module: Any, fit_maxiter: int, scale_axis: str) -> Any:
    return module.CandidateConfig(
        name=f"{MODEL_NAME}_{scale_axis}",
        title=f"Compact CI + q/support-conditioned floor, {scale_axis} scale, maxiter={fit_maxiter}",
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
    slope, intercept = (np.nan, np.nan)
    if len(actual_arr) >= 2 and actual_std > 0.0:
        slope, intercept = np.polyfit(actual_arr, pred_arr, 1)
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


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "train": predictions["eval_role"].eq("train"),
        "holdout": predictions["eval_role"].ne("train"),
        "seed7_fixed_520m": predictions["eval_role"].eq("seed7_holdout") & predictions["holdout_kind"].eq("fixed_520m"),
        "seed7_random_supplement": (
            predictions["eval_role"].eq("seed7_holdout") & predictions["holdout_kind"].eq("random_supplement")
        ),
        "validation_1p2b": predictions["eval_role"].eq("validation_1p2b"),
        "all_primary": predictions["eval_role"].isin(["train", "seed7_holdout", "validation_1p2b"]),
    }
    rows = []
    for name, mask in subsets.items():
        subset = predictions.loc[mask].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "subset": name,
                **metric_bundle(subset["actual_bpb"].to_numpy(float), subset["predicted_bpb"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(
    benchmark: Any,
    model: dict[str, Any],
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    role_name: str,
    holdout_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    packet = benchmark.packet
    primary_indices = np.flatnonzero(packet.primary_mask)
    pred = benchmark.predict_indices(model, primary_indices)
    frame = packet.frame.iloc[primary_indices].copy()
    frame["target_index"] = primary_indices
    frame["actual_bpb"] = packet.primary_y[primary_indices].astype(float)
    frame["predicted_bpb"] = pred.astype(float)
    frame["residual"] = frame["predicted_bpb"] - frame["actual_bpb"]
    frame["actual_non_embedding_params"] = [
        ACTUAL_NON_EMBEDDING_PARAMS[str(scale)] for scale in frame["scale"].astype(str)
    ]
    train_set = set(np.asarray(train_indices, dtype=int).tolist())
    holdout_set = set(np.asarray(holdout_indices, dtype=int).tolist())
    frame["eval_role"] = [
        role_name if int(idx) in holdout_set else "train" if int(idx) in train_set else "unused"
        for idx in frame["target_index"].to_numpy(int)
    ]
    if holdout_metadata is not None:
        meta_cols = [
            "target_index",
            "holdout_kind",
            "fixed_520m_bucket",
        ]
        available = [col for col in meta_cols if col in holdout_metadata.columns]
        meta = holdout_metadata[available].drop_duplicates("target_index")
        frame = frame.merge(meta, on="target_index", how="left", suffixes=("", "_holdout"))
    if "holdout_kind" not in frame.columns:
        frame["holdout_kind"] = np.where(frame["eval_role"].eq(role_name), role_name, "train")
    frame["holdout_kind"] = frame["holdout_kind"].fillna("train")
    return frame


def run_seed7_eval(benchmark: Any, module: Any, fit_maxiter: int) -> EvalBundle:
    import run_direct_multiscale_grp_components as direct_mod
    import run_transfer_benchmark_holdouts as bench_mod

    _train_examples, holdout_df, _examples = bench_mod.build_benchmark_split(
        benchmark.law,
        include_1_2b=True,
        seed=7,
    )
    train_indices = direct_mod.all_primary_train_indices(benchmark.law, holdout_df)
    holdout_indices = holdout_df["target_index"].to_numpy(int)
    model = benchmark.fit(train_indices=train_indices, config=qsupport_config(module, fit_maxiter, benchmark.scale_axis))
    predictions = prediction_frame(benchmark, model, train_indices, holdout_indices, "seed7_holdout", holdout_df)
    metrics = summarize_predictions(predictions)
    metrics.insert(0, "eval", "seed7")
    return EvalBundle(model=model, predictions=predictions, metrics=metrics)


def run_1p2b_holdout_eval(benchmark: Any, module: Any, fit_maxiter: int) -> EvalBundle:
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
    train_indices = np.asarray([idx for idx in primary_indices if idx not in set(holdout_indices)], dtype=int)
    model = benchmark.fit(train_indices=train_indices, config=qsupport_config(module, fit_maxiter, benchmark.scale_axis))
    predictions = prediction_frame(benchmark, model, train_indices, holdout_indices, "validation_1p2b")
    metrics = summarize_predictions(predictions)
    metrics.insert(0, "eval", "onepoint2b_holdout")
    return EvalBundle(model=model, predictions=predictions, metrics=metrics)


def plot_predicted_vs_actual(predictions: pd.DataFrame, out_path: Path, title: str, holdout_role: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    train = predictions.loc[predictions["eval_role"].eq("train")]
    holdout = predictions.loc[predictions["eval_role"].eq(holdout_role)]
    ax.scatter(
        train["actual_bpb"],
        train["predicted_bpb"],
        s=18,
        color="0.70",
        alpha=0.28,
        edgecolors="none",
        label=f"train ({len(train)})",
    )
    cmap = plt.get_cmap("RdYlGn_r")
    if holdout_role == "seed7_holdout":
        values = np.log10(holdout["actual_non_embedding_params"].to_numpy(float))
        colors = cmap((values - values.min()) / max(values.max() - values.min(), 1e-12))
        labels = holdout["holdout_kind"].astype(str)
        for kind in sorted(labels.unique()):
            mask = labels.eq(kind)
            ax.scatter(
                holdout.loc[mask, "actual_bpb"],
                holdout.loc[mask, "predicted_bpb"],
                s=np.where(kind == "fixed_520m", 70, 44),
                c=colors[mask.to_numpy(bool)],
                edgecolors="black",
                linewidths=0.5,
                label=f"holdout: {kind}",
                zorder=5,
            )
    else:
        for pos, (run_name, group) in enumerate(holdout.groupby("run_name", sort=True)):
            ax.scatter(
                group["actual_bpb"],
                group["predicted_bpb"],
                s=120,
                color=cmap(pos / max(1, len(RUN_NAME_HOLDOUTS) - 1)),
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
    subset_name = "holdout" if holdout_role == "seed7_holdout" else "validation_1p2b"
    row = metrics.loc[metrics["subset"].eq(subset_name)]
    if not row.empty:
        metric = row.iloc[0]
        metric_text = (
            f"{subset_name} RMSE={metric['rmse']:.4f}\n"
            f"mean residual={metric['mean_residual']:+.4f}\n"
            f"slope={metric['slope']:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            metric_text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual BPB")
    ax.set_ylabel("Predicted BPB")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fixed520m_multiplier(predictions: pd.DataFrame, out_path: Path) -> None:
    fixed = predictions.loc[
        predictions["eval_role"].eq("seed7_holdout") & predictions["holdout_kind"].eq("fixed_520m")
    ].copy()
    if fixed.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    means = fixed.groupby("target_budget_multiplier", as_index=False).agg(
        actual_mean=("actual_bpb", "mean"),
        predicted_mean=("predicted_bpb", "mean"),
    )
    for _, group in fixed.groupby("run_name"):
        ordered = group.sort_values("target_budget_multiplier")
        axes[0].plot(
            ordered["target_budget_multiplier"],
            ordered["actual_bpb"],
            color="0.72",
            alpha=0.55,
            linewidth=1.2,
        )
        axes[0].plot(
            ordered["target_budget_multiplier"],
            ordered["predicted_bpb"],
            color="#cf3b25",
            alpha=0.18,
            linewidth=1.2,
        )
    axes[0].plot(
        means["target_budget_multiplier"],
        means["actual_mean"],
        color="black",
        marker="o",
        linewidth=2.8,
        label="actual mean",
    )
    axes[0].plot(
        means["target_budget_multiplier"],
        means["predicted_mean"],
        color="#cf3b25",
        marker="s",
        linewidth=2.8,
        label="predicted mean",
    )
    axes[0].set_xlabel("target-budget multiplier")
    axes[0].set_ylabel("BPB")
    axes[0].set_title("Fixed 520M multiplier trajectories")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    ordered_multipliers = sorted(fixed["target_budget_multiplier"].astype(float).unique())
    data = [
        fixed.loc[np.isclose(fixed["target_budget_multiplier"].astype(float), mult), "residual"].to_numpy(float)
        for mult in ordered_multipliers
    ]
    axes[1].boxplot(data, tick_labels=[f"{mult:g}" for mult in ordered_multipliers])
    axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlabel("target-budget multiplier")
    axes[1].set_ylabel("prediction residual")
    axes[1].set_title("Residual by fixed-520M multiplier")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Compact q/support corrected-data diagnostics, seed 7")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_1p2b_trajectory(predictions: pd.DataFrame, out_path: Path) -> None:
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
        holdout = group["eval_role"].eq("validation_1p2b").to_numpy(bool)
        ax.scatter(
            x[holdout],
            group.loc[holdout, "actual_bpb"],
            s=95,
            color=cmap(0.1),
            edgecolors="black",
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(run_name)
        ax.set_xlabel("scale / multiplier")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("BPB")
    axes[0].legend(loc="best")
    fig.suptitle("Baseline trajectories with compact q/support 1.2B holdout fit")
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

    seed7_benchmark = qsupport_module.QSupportFloorBenchmark(args.packet_root)
    seed7_benchmark.scale_axis = args.scale_axis
    label_patch = apply_registry_labels(seed7_benchmark, args.registry_csv)
    scale_patch = apply_param_axis(seed7_benchmark, args.scale_axis)
    seed7 = run_seed7_eval(seed7_benchmark, qsupport_module, args.fit_maxiter)

    onepoint2b_benchmark = qsupport_module.QSupportFloorBenchmark(args.packet_root)
    onepoint2b_benchmark.scale_axis = args.scale_axis
    apply_registry_labels(onepoint2b_benchmark, args.registry_csv)
    apply_param_axis(onepoint2b_benchmark, args.scale_axis)
    onepoint2b = run_1p2b_holdout_eval(onepoint2b_benchmark, qsupport_module, args.fit_maxiter)

    seed7.predictions.to_csv(args.out_dir / "seed7_predictions.csv", index=False)
    seed7.metrics.to_csv(args.out_dir / "seed7_metrics.csv", index=False)
    onepoint2b.predictions.to_csv(args.out_dir / "onepoint2b_holdout_predictions.csv", index=False)
    onepoint2b.metrics.to_csv(args.out_dir / "onepoint2b_holdout_metrics.csv", index=False)
    pd.concat([seed7.metrics, onepoint2b.metrics], ignore_index=True).to_csv(args.out_dir / "metrics.csv", index=False)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "fit_maxiter": int(args.fit_maxiter),
                "label_patch": label_patch,
                "scale_patch": scale_patch,
                "seed7_model": {
                    "parameter_count": int(seed7.model["parameter_count"]),
                    "train_rmse": float(seed7.model["train_rmse"]),
                    "optimizer_success": bool(seed7.model["optimizer_success"]),
                    "optimizer_message": str(seed7.model["optimizer_message"]),
                    "floor_raw": np.asarray(seed7.model["floor_raw"], dtype=float).tolist(),
                },
                "onepoint2b_model": {
                    "parameter_count": int(onepoint2b.model["parameter_count"]),
                    "train_rmse": float(onepoint2b.model["train_rmse"]),
                    "optimizer_success": bool(onepoint2b.model["optimizer_success"]),
                    "optimizer_message": str(onepoint2b.model["optimizer_message"]),
                    "floor_raw": np.asarray(onepoint2b.model["floor_raw"], dtype=float).tolist(),
                },
            },
            handle,
            indent=2,
        )

    plot_predicted_vs_actual(
        seed7.predictions,
        figures_dir / "seed7_predicted_vs_actual_corrected.png",
        "Compact q/support corrected data: seed-7 holdout",
        "seed7_holdout",
    )
    plot_fixed520m_multiplier(
        seed7.predictions,
        figures_dir / "seed7_fixed520m_multiplier_corrected.png",
    )
    plot_predicted_vs_actual(
        onepoint2b.predictions,
        figures_dir / "onepoint2b_holdout_predicted_vs_actual_corrected.png",
        "Compact q/support corrected data: 1.2B holdout",
        "validation_1p2b",
    )
    plot_1p2b_trajectory(
        onepoint2b.predictions,
        figures_dir / "onepoint2b_baseline_trajectories_corrected.png",
    )

    print(pd.concat([seed7.metrics, onepoint2b.metrics], ignore_index=True).to_string(index=False))
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
