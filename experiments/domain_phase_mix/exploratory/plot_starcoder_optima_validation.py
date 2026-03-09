# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pandas", "wandb"]
# ///
"""Plot predicted-vs-actual BPB for validated StarCoder optima."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from experiments.domain_phase_mix.starcoder_metadata import (
    load_three_phase_starcoder_dataset,
    load_two_phase_starcoder_dataset,
)

OBJECTIVE_MODEL = "DS-RE-CEQ"
OBJECTIVE_MODE = "retrospective"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"


def _display_dataset_name(dataset: str) -> str:
    return {
        "two_phase_starcoder": "Two Phase StarCoder",
        "three_phase_starcoder": "Three Phase StarCoder",
    }.get(dataset, dataset)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("two_phase_starcoder", "three_phase_starcoder"),
        required=True,
    )
    parser.add_argument("--policy", type=str, default="feature_bayes_linear_observed")
    parser.add_argument("--metric-key", type=str, default=OBJECTIVE_METRIC)
    parser.add_argument("--wandb-entity", type=str, default="marin-community")
    parser.add_argument("--wandb-project", type=str, default="marin")
    parser.add_argument("--rl-rollout-report", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def _load_launch_plan(benchmark_output_dir: Path, dataset: str) -> dict[str, Any]:
    path = benchmark_output_dir / f"{dataset}_validation_launch_plan.json"
    return json.loads(path.read_text())


def _load_predicted_optima_frame(benchmark_output_dir: Path, *, dataset: str, policy: str) -> pd.DataFrame:
    frame = pd.read_csv(benchmark_output_dir / "predicted_optima.csv")
    frame = frame[
        (frame["dataset"] == dataset)
        & (frame["mode"] == OBJECTIVE_MODE)
        & (frame["policy"] == policy)
        & (frame["evaluation_model"] == OBJECTIVE_MODEL)
    ].copy()
    return frame[["subset_size", "predicted_objective"]].rename(columns={"predicted_objective": "predicted_bpb"})


def _load_regret_curve_frame(benchmark_output_dir: Path, *, dataset: str, policy: str) -> pd.DataFrame:
    frame = pd.read_csv(benchmark_output_dir / "curve_points.csv")
    frame = frame[
        (frame["dataset"] == dataset)
        & (frame["mode"] == OBJECTIVE_MODE)
        & (frame["policy"] == policy)
        & (frame["evaluation_model"] == OBJECTIVE_MODEL)
    ].copy()
    return frame[["subset_size", "regret@1_median"]].rename(columns={"regret@1_median": "regret_at_1"})


def _load_observed_bpb_summary(dataset: str) -> tuple[float, int]:
    if dataset == "two_phase_starcoder":
        spec, _ = load_two_phase_starcoder_dataset(target_col=OBJECTIVE_METRIC)
    elif dataset == "three_phase_starcoder":
        spec, _ = load_three_phase_starcoder_dataset(target_col=OBJECTIVE_METRIC)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return float(spec.y.min()), len(spec.y)


def _load_rl_rollout_summary(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text()
    schedule_section = re.search(
        r"The resulting executed StarCoder schedule for `r00` was:\s*(.*?)\s*The final programming BPB was:",
        text,
        flags=re.DOTALL,
    )
    if schedule_section is None:
        raise ValueError(f"Could not locate rollout schedule section in {report_path}")
    phase_matches = re.findall(r"phase_(\d+)\s*=\s*([0-9.]+)", schedule_section.group(1))
    if len(phase_matches) != 3:
        raise ValueError(f"Could not parse rollout phase weights from {report_path}")
    ordered = sorted(((int(phase_idx), float(value)) for phase_idx, value in phase_matches), key=lambda item: item[0])
    rollout_tuple = tuple(value for _, value in ordered)

    summary_match = re.search(
        r"The final programming BPB was:\s*[-*]\s*`([0-9.]+)`",
        text,
        flags=re.MULTILINE,
    )
    if summary_match is None:
        raise ValueError(f"Could not parse rollout BPB from {report_path}")

    return {
        "label": "RL rollout BPB (outcome_planner)",
        "validated_bpb": float(summary_match.group(1)),
        "rollout_tuple": rollout_tuple,
    }


def _fetch_actual_metric_map(
    *,
    run_names: list[str],
    metric_key: str,
    wandb_entity: str,
    wandb_project: str,
) -> dict[str, dict[str, Any]]:
    api = wandb.Api(overrides={"entity": wandb_entity, "project": wandb_project})
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"display_name": {"$regex": r"feature_bayes_linear_k.*_optimum$"}},
    )
    selected = {}
    target_names = set(run_names)
    for run in runs:
        if run.name not in target_names:
            continue
        selected[run.name] = {
            "actual_bpb": run.summary.get(metric_key),
            "wandb_state": run.state,
            "wandb_url": run.url,
        }
    return selected


def _format_starcoder_tuple(weight_config: dict[str, Any]) -> str:
    phase_weights = weight_config["phase_weights"]
    phase_names = sorted(phase_weights, key=lambda phase_name: int(phase_name.split("_")[-1]))
    weights = [float(phase_weights[phase_name]["starcoder"]) for phase_name in phase_names]
    return "(" + ", ".join(f"{weight:.4f}" for weight in weights) + ")"


def build_validation_plot_frame(
    *,
    launch_plan: dict[str, Any],
    predicted_optima: pd.DataFrame,
    regret_curve: pd.DataFrame,
    actual_metric_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    predicted_map = {
        int(row["subset_size"]): float(row["predicted_bpb"])
        for row in predicted_optima.to_dict(orient="records")
    }
    regret_map = {
        int(row["subset_size"]): float(row["regret_at_1"])
        for row in regret_curve.to_dict(orient="records")
    }
    name_prefix = str(launch_plan["name_prefix"])
    for run in launch_plan["runs"]:
        subset_size = int(run["subset_size"])
        full_run_name = f"{name_prefix}/{run['run_name']}"
        actual = actual_metric_map.get(full_run_name, {})
        rows.append(
            {
                "subset_size": subset_size,
                "run_name": str(run["run_name"]),
                "wandb_run_name": full_run_name,
                "predicted_optimum_tuple": _format_starcoder_tuple(run["weight_config"]),
                "predicted_bpb": predicted_map.get(subset_size),
                "actual_bpb": actual.get("actual_bpb"),
                "regret_at_1": regret_map.get(subset_size),
                "wandb_state": actual.get("wandb_state"),
                "wandb_url": actual.get("wandb_url"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("subset_size").reset_index(drop=True)
    return frame


def _plot_validation_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    best_observed_bpb: float,
    observed_count: int,
    rl_rollout_summary: dict[str, Any] | None,
    output_path: Path,
) -> None:
    fig, ax_bpb = plt.subplots(figsize=(10, 6))
    ax_regret = ax_bpb.twinx()

    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_bpb"],
        color="#1d4ed8",
        marker="o",
        linewidth=2.0,
        linestyle="--",
        label="Predicted BPB",
    )
    ax_bpb.plot(
        frame["subset_size"],
        frame["actual_bpb"],
        color="#c2410c",
        marker="o",
        linewidth=2.0,
        linestyle="-",
        label="Actual BPB",
    )
    ax_bpb.axhline(
        best_observed_bpb,
        color="#dc2626",
        linewidth=2.0,
        linestyle="--",
        label=f"Best observed BPB (among {observed_count} runs, {best_observed_bpb:.4f})",
    )
    if rl_rollout_summary is not None:
        rl_bpb = float(rl_rollout_summary["validated_bpb"])
        ax_bpb.axhline(
            rl_bpb,
            color="#7c3aed",
            linewidth=2.0,
            linestyle="--",
            label=f"{rl_rollout_summary['label']} ({rl_bpb:.4f})",
        )
    ax_regret.plot(
        frame["subset_size"],
        frame["regret_at_1"],
        color="#0f766e",
        marker="s",
        linewidth=2.0,
        linestyle="-",
        label="Regret@1",
    )

    ax_bpb.set_xlabel("Subset size")
    ax_bpb.set_ylabel("BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_bpb.set_title(f"{_display_dataset_name(dataset)} — predicted vs actual BPB at Feature Bayes Linear optima")
    ax_bpb.grid(True, which="major", axis="both", alpha=0.25)
    ax_bpb.set_xticks(frame["subset_size"].tolist())

    validated = frame[frame["actual_bpb"].notna()]
    if not validated.empty:
        min_idx = validated["actual_bpb"].idxmin()
        min_row = validated.loc[min_idx]
        ax_bpb.annotate(
            "\n".join(
                [
                    f"Min validated BPB = {min_row['actual_bpb']:.4f} at k={int(min_row['subset_size'])}",
                    f"Predicted optimum = {min_row['predicted_optimum_tuple']}",
                ]
            ),
            xy=(min_row["subset_size"], min_row["actual_bpb"]),
            xytext=(8, -18),
            textcoords="offset points",
            color="#c2410c",
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#c2410c", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#c2410c", "lw": 1.2},
        )
    if rl_rollout_summary is not None:
        rl_bpb = float(rl_rollout_summary["validated_bpb"])
        rl_tuple = tuple(float(value) for value in rl_rollout_summary["rollout_tuple"])
        tuple_text = "(" + ", ".join(f"{value:.4f}" for value in rl_tuple) + ")"
        ax_bpb.annotate(
            "\n".join(
                [
                    f"RL rollout BPB = {rl_bpb:.4f}",
                    f"Rollout actions = {tuple_text}",
                ]
            ),
            xy=(frame["subset_size"].iloc[-1], rl_bpb),
            xytext=(-8, -18),
            textcoords="offset points",
            color="#7c3aed",
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#7c3aed", "alpha": 0.9},
        )

    handles = ax_bpb.get_lines() + ax_regret.get_lines()
    labels = [handle.get_label() for handle in handles]
    ax_bpb.legend(handles, labels, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    benchmark_output_dir = args.benchmark_output_dir.resolve()
    launch_plan = _load_launch_plan(benchmark_output_dir, args.dataset)
    predicted_optima = _load_predicted_optima_frame(
        benchmark_output_dir,
        dataset=args.dataset,
        policy=args.policy,
    )
    regret_curve = _load_regret_curve_frame(
        benchmark_output_dir,
        dataset=args.dataset,
        policy=args.policy,
    )
    run_names = [f"{launch_plan['name_prefix']}/{run['run_name']}" for run in launch_plan["runs"]]
    actual_metric_map = _fetch_actual_metric_map(
        run_names=run_names,
        metric_key=args.metric_key,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    frame = build_validation_plot_frame(
        launch_plan=launch_plan,
        predicted_optima=predicted_optima,
        regret_curve=regret_curve,
        actual_metric_map=actual_metric_map,
    )
    best_observed_bpb, observed_count = _load_observed_bpb_summary(args.dataset)
    frame["best_observed_bpb"] = best_observed_bpb
    frame["observed_count"] = observed_count
    rl_rollout_summary = None
    if args.rl_rollout_report is not None:
        rl_rollout_summary = _load_rl_rollout_summary(args.rl_rollout_report.resolve())
        frame["rl_rollout_bpb"] = float(rl_rollout_summary["validated_bpb"])
        frame["rl_rollout_tuple"] = [json.dumps(list(rl_rollout_summary["rollout_tuple"]))] * len(frame)

    output_path = args.output_path
    if output_path is None:
        output_path = benchmark_output_dir / "plots" / f"{args.dataset}_{args.policy}_predicted_vs_actual_bpb.png"
    csv_output = args.csv_output
    if csv_output is None:
        csv_output = benchmark_output_dir / f"{args.dataset}_{args.policy}_predicted_vs_actual_bpb.csv"

    frame.to_csv(csv_output, index=False)
    _plot_validation_frame(
        frame,
        dataset=args.dataset,
        best_observed_bpb=best_observed_bpb,
        observed_count=observed_count,
        rl_rollout_summary=rl_rollout_summary,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
