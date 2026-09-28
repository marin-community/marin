# Copyright The Marin Authors
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from matplotlib.ticker import NullLocator

from experiments.domain_phase_mix.starcoder_metadata import (
    load_three_phase_starcoder_dataset,
    load_two_phase_starcoder_dataset,
)

OBJECTIVE_MODEL = "DS-RE-CEQ"
OBJECTIVE_MODE = "retrospective"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"


@dataclass(frozen=True)
class ObservedBpbSummary:
    """Best-observed BPB summary for one StarCoder reference set."""

    best_bpb: float
    observed_count: int
    best_tuple: tuple[float, ...]


def _display_dataset_name(dataset: str) -> str:
    return {
        "two_phase_starcoder": "Two Phase StarCoder",
        "three_phase_starcoder": "Three Phase StarCoder",
    }.get(dataset, dataset)


def _display_policy_name(policy: str) -> str:
    if "wsd_boundary_aligned" in policy:
        return "Feature Bayes Linear optima (WSD boundary-aligned)"
    if "wsds" in policy or "wsd" in policy:
        return "Feature Bayes Linear optima (WSD)"
    return "Feature Bayes Linear optima"


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
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--rl-rollout-report", type=Path, default=None)
    parser.add_argument("--observed-reference-csv", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--x-scale", choices=("linear", "log"), default="linear")
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


def _format_numeric_tuple(values: tuple[float, ...], *, trim_trailing_zeros: bool = False) -> str:
    if not trim_trailing_zeros:
        return "(" + ", ".join(f"{value:.4f}" for value in values) + ")"

    def _format_value(value: float) -> str:
        rounded = round(float(value), 4)
        text = f"{rounded:.4f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{text}.0"

    return "(" + ", ".join(_format_value(value) for value in values) + ")"


def _infer_starcoder_phase_columns(frame: pd.DataFrame) -> list[str]:
    phase_columns = []
    for column in frame.columns:
        match = re.fullmatch(r"phase_(\d+)_starcoder", column)
        if match is None:
            continue
        phase_columns.append((int(match.group(1)), column))
    if not phase_columns:
        raise ValueError("Could not infer any StarCoder phase columns from the observed reference frame")
    return [column for _, column in sorted(phase_columns)]


def _summarize_observed_bpb_frame(frame: pd.DataFrame) -> ObservedBpbSummary:
    metric_frame = frame[frame[OBJECTIVE_METRIC].notna()].copy()
    if metric_frame.empty:
        raise ValueError("Observed reference frame has no finite objective values")

    phase_columns = _infer_starcoder_phase_columns(metric_frame)
    best_index = metric_frame[OBJECTIVE_METRIC].astype(float).idxmin()
    best_row = metric_frame.loc[best_index]
    best_tuple = tuple(float(best_row[column]) for column in phase_columns)
    return ObservedBpbSummary(
        best_bpb=float(best_row[OBJECTIVE_METRIC]),
        observed_count=len(metric_frame),
        best_tuple=best_tuple,
    )


def _load_observed_bpb_summary(dataset: str, observed_reference_csv: Path | None = None) -> ObservedBpbSummary:
    if observed_reference_csv is not None:
        return _summarize_observed_bpb_frame(pd.read_csv(observed_reference_csv.resolve()))

    if dataset == "two_phase_starcoder":
        _, frame = load_two_phase_starcoder_dataset(target_col=OBJECTIVE_METRIC)
    elif dataset == "three_phase_starcoder":
        _, frame = load_three_phase_starcoder_dataset(target_col=OBJECTIVE_METRIC)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return _summarize_observed_bpb_frame(frame)


def _load_rl_rollout_summary(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text()
    schedule_section = None
    schedule_patterns = [
        r"The resulting executed StarCoder schedule for `r00` was:\s*(.*?)\s*The final programming BPB was:",
        r"The two-phase rollout schedule was:\s*(.*?)\s*Per-phase metrics:",
    ]
    for pattern in schedule_patterns:
        schedule_section = re.search(pattern, text, flags=re.DOTALL)
        if schedule_section is not None:
            break
    if schedule_section is None:
        raise ValueError(f"Could not locate rollout schedule section in {report_path}")

    phase_matches = re.findall(r"phase_(\d+)\s*=\s*([0-9.]+)", schedule_section.group(1))
    if len(phase_matches) < 2:
        raise ValueError(f"Could not parse rollout phase weights from {report_path}")
    ordered = sorted(((int(phase_idx), float(value)) for phase_idx, value in phase_matches), key=lambda item: item[0])
    rollout_tuple = tuple(value for _, value in ordered)

    summary_match = None
    summary_patterns = [
        r"The final programming BPB was:\s*[-*]\s*`([0-9.]+)`",
        r"final programming BPB:\s*`([0-9.]+)`",
    ]
    for pattern in summary_patterns:
        summary_match = re.search(pattern, text, flags=re.MULTILINE)
        if summary_match is not None:
            break
    if summary_match is None:
        raise ValueError(f"Could not parse rollout BPB from {report_path}")

    return {
        "label": "RL rollout BPB (outcome_planner)",
        "validated_bpb": float(summary_match.group(1)),
        "rollout_tuple": rollout_tuple,
    }


def _run_name_regex(run_names: list[str]) -> str:
    suffixes = sorted({re.escape(name.rsplit("/", 1)[-1]) for name in run_names})
    if not suffixes:
        raise ValueError("run_names must be non-empty")
    return rf"(?:{'|'.join(suffixes)})$"


def _fetch_actual_metric_map(
    *,
    dataset: str,
    run_names: list[str],
    metric_key: str,
    wandb_entity: str,
    wandb_project: str,
) -> dict[str, dict[str, Any]]:
    api = wandb.Api(overrides={"entity": wandb_entity, "project": wandb_project})
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"display_name": {"$regex": _run_name_regex(run_names)}},
    )
    target_suffixes = {name.rsplit("/", 1)[-1]: name for name in run_names}
    exact_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in run_names}
    suffix_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in run_names}

    def _candidate_payload(run: Any) -> dict[str, Any]:
        return {
            "actual_bpb": run.summary.get(metric_key),
            "wandb_state": run.state,
            "wandb_url": run.url,
            "wandb_run_name": run.name,
            "created_at": getattr(run, "created_at", "") or "",
        }

    target_names = set(run_names)
    for run in runs:
        candidate = _candidate_payload(run)
        if run.name in target_names:
            exact_candidates[run.name].append(candidate)
            continue

        suffix = run.name.rsplit("/", 1)[-1]
        target_name = target_suffixes.get(suffix)
        if target_name is not None and _wandb_run_matches_dataset(run.name, dataset):
            suffix_candidates[target_name].append(candidate)

    return _select_actual_metric_candidates(
        run_names=run_names,
        exact_candidates=exact_candidates,
        suffix_candidates=suffix_candidates,
    )


def _select_actual_metric_candidates(
    *,
    run_names: list[str],
    exact_candidates: dict[str, list[dict[str, Any]]],
    suffix_candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
        return (
            int(candidate.get("wandb_state") == "finished"),
            int(candidate.get("actual_bpb") is not None),
            str(candidate.get("created_at", "")),
        )

    selected = {}
    for target_name in run_names:
        candidates = exact_candidates.get(target_name) or suffix_candidates.get(target_name) or []
        if not candidates:
            continue
        selected[target_name] = max(candidates, key=_candidate_sort_key)
    return selected


def _wandb_run_matches_dataset(run_name: str, dataset: str) -> bool:
    if dataset == "two_phase_starcoder":
        return "two_phase" in run_name or "/t2s-" in run_name
    if dataset == "three_phase_starcoder":
        return "three_phase" in run_name or "/t3s-" in run_name
    return True


def _format_starcoder_tuple(weight_config: dict[str, Any]) -> str:
    phase_weights = weight_config["phase_weights"]
    phase_names = sorted(phase_weights, key=lambda phase_name: int(phase_name.split("_")[-1]))
    weights = [float(phase_weights[phase_name]["starcoder"]) for phase_name in phase_names]
    return _format_numeric_tuple(tuple(weights))


def build_validation_plot_frame(
    *,
    launch_plan: dict[str, Any],
    predicted_optima: pd.DataFrame,
    regret_curve: pd.DataFrame,
    actual_metric_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    predicted_map = {
        int(row["subset_size"]): float(row["predicted_bpb"]) for row in predicted_optima.to_dict(orient="records")
    }
    regret_map = {int(row["subset_size"]): float(row["regret_at_1"]) for row in regret_curve.to_dict(orient="records")}
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
    policy: str,
    x_scale: str,
    best_observed_bpb: float,
    observed_count: int,
    best_observed_tuple: tuple[float, ...],
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
    ax_bpb.set_title(f"{_display_dataset_name(dataset)} — predicted vs actual BPB at {_display_policy_name(policy)}")
    ax_bpb.grid(True, which="major", axis="both", alpha=0.25)
    ax_bpb.set_xscale(x_scale)
    subset_sizes = frame["subset_size"].astype(int).tolist()
    ax_bpb.set_xticks(subset_sizes)
    ax_bpb.set_xticklabels([str(value) for value in subset_sizes])
    if x_scale == "log":
        ax_bpb.xaxis.set_minor_locator(NullLocator())
    ax_bpb.annotate(
        "\n".join(
            [
                f"Best observed BPB = {best_observed_bpb:.4f}",
                f"Observed best = {_format_numeric_tuple(best_observed_tuple)}",
            ]
        ),
        xy=(frame["subset_size"].iloc[-1], best_observed_bpb),
        xytext=(-8, 10),
        textcoords="offset points",
        color="#dc2626",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#dc2626", "alpha": 0.9},
    )

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
        tuple_text = _format_numeric_tuple(rl_tuple, trim_trailing_zeros=True)
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
    actual_metric_map = (
        {}
        if args.skip_wandb
        else _fetch_actual_metric_map(
            dataset=args.dataset,
            run_names=run_names,
            metric_key=args.metric_key,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
    )
    frame = build_validation_plot_frame(
        launch_plan=launch_plan,
        predicted_optima=predicted_optima,
        regret_curve=regret_curve,
        actual_metric_map=actual_metric_map,
    )
    observed_summary = _load_observed_bpb_summary(args.dataset, observed_reference_csv=args.observed_reference_csv)
    frame["best_observed_bpb"] = observed_summary.best_bpb
    frame["observed_count"] = observed_summary.observed_count
    frame["best_observed_tuple"] = [_format_numeric_tuple(observed_summary.best_tuple)] * len(frame)
    rl_rollout_summary = None
    if args.rl_rollout_report is not None:
        rl_rollout_summary = _load_rl_rollout_summary(args.rl_rollout_report.resolve())
        frame["rl_rollout_bpb"] = float(rl_rollout_summary["validated_bpb"])
        frame["rl_rollout_tuple"] = [json.dumps(list(rl_rollout_summary["rollout_tuple"]))] * len(frame)

    output_path = args.output_path
    if output_path is None:
        suffix = "_logx" if args.x_scale == "log" else ""
        output_path = (
            benchmark_output_dir / "plots" / f"{args.dataset}_{args.policy}_predicted_vs_actual_bpb{suffix}.png"
        )
    csv_output = args.csv_output
    if csv_output is None:
        csv_output = benchmark_output_dir / f"{args.dataset}_{args.policy}_predicted_vs_actual_bpb.csv"

    frame.to_csv(csv_output, index=False)
    _plot_validation_frame(
        frame,
        dataset=args.dataset,
        policy=args.policy,
        x_scale=args.x_scale,
        best_observed_bpb=observed_summary.best_bpb,
        observed_count=observed_summary.observed_count,
        best_observed_tuple=observed_summary.best_tuple,
        rl_rollout_summary=rl_rollout_summary,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
