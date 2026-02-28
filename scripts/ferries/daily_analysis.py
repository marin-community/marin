#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch canonical daily ferry metrics from a W&B run.

Examples:
  uv run python scripts/ferries/daily_analysis.py \
    --run https://wandb.ai/marin-community/marin/runs/ferry_daily_125m_... \
    --format markdown

  uv run python scripts/ferries/daily_analysis.py \
    --run marin-community/marin/ferry_daily_125m_... \
    --format json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import wandb

DEFAULT_METRIC_KEYS = (
    "eval/paloma/c4_en/bpb",
    "eval/bpb",
    "eval/uncheatable_eval/bpb",
)
DEFAULT_THRESHOLDS = (5.0, 4.5, 4.0, 3.8)
TRAIN_LOSS_KEY = "train/loss"


@dataclass(frozen=True)
class AnalysisResult:
    run_path: str
    run_url: str
    run_name: str
    metrics: dict[str, float | int | str | None]
    learning_dynamics: dict[str, object]
    recent_ferries: list[dict[str, object]]
    analysis_observations: list[str]


def _normalize_run_path(run: str) -> str:
    """Normalize a W&B run reference into `entity/project/run_id_or_name`."""
    if "wandb.ai" not in run:
        parts = [p for p in run.strip("/").split("/") if p]
        if len(parts) != 3:
            raise ValueError(f"Expected run path `entity/project/run`, got: {run!r}")
        return "/".join(parts)

    parsed = urlparse(run)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    # URL form: /<entity>/<project>/runs/<run_id_or_name>
    if len(parts) >= 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    raise ValueError(f"Unrecognized W&B run URL format: {run!r}")


def _fmt_value(value: object, precision: int) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_train_loss_points(run: wandb.apis.public.Run) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in run.scan_history(keys=["_step", TRAIN_LOSS_KEY]):
        step = row.get("_step")
        loss = row.get(TRAIN_LOSS_KEY)
        if isinstance(step, (int, float)) and isinstance(loss, (int, float)):
            points.append((int(step), float(loss)))
    return points


def _sample_trajectory(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    if not points:
        return []
    if len(points) <= max_points:
        return points
    if max_points <= 1:
        return [points[-1]]

    sampled: list[tuple[int, float]] = []
    last_index = len(points) - 1
    for i in range(max_points):
        idx = round(i * last_index / (max_points - 1))
        sampled.append(points[idx])
    return sampled


def _compute_step_to_threshold(points: list[tuple[int, float]], threshold: float) -> int | None:
    for step, loss in points:
        if loss <= threshold:
            return step
    return None


def _compute_learning_dynamics(
    points: list[tuple[int, float]],
    *,
    thresholds: tuple[float, ...],
    window_points: int,
    trajectory_points: int,
) -> dict[str, object]:
    if not points:
        return {"available": False, "reason": f"No `{TRAIN_LOSS_KEY}` points found in run history."}

    losses = [loss for _, loss in points]
    early_window = losses[: min(window_points, len(losses))]
    late_window = losses[-min(window_points, len(losses)) :]

    step_to_threshold = {
        f"loss<={threshold:.1f}": _compute_step_to_threshold(points, threshold) for threshold in thresholds
    }

    late_volatility = statistics.pstdev(late_window) if len(late_window) >= 2 else None

    return {
        "available": True,
        "point_count": len(points),
        "train_loss_trajectory": _sample_trajectory(points, trajectory_points),
        "step_to_threshold": step_to_threshold,
        "early_loss_avg": statistics.fmean(early_window) if early_window else None,
        "late_loss_avg": statistics.fmean(late_window) if late_window else None,
        "late_loss_volatility_std": late_volatility,
        "window_points": min(window_points, len(losses)),
    }


def _fetch_recent_ferry_comparisons(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    exclude_run_name: str,
    max_runs: int,
    metric_keys: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    if max_runs <= 0:
        return []

    runs = api.runs(
        f"{entity}/{project}",
        filters={
            "state": "finished",
            "display_name": {"$regex": "^ferry_daily_"},
        },
        order="-created_at",
        per_page=max(20, max_runs * 4),
    )

    comparisons: list[dict[str, object]] = []
    for run in runs:
        if run.name == exclude_run_name:
            continue
        summary_metrics = {key: run.summary.get(key) for key in metric_keys}
        history_points = _extract_train_loss_points(run)
        dynamics = _compute_learning_dynamics(
            history_points,
            thresholds=thresholds,
            window_points=500,
            trajectory_points=10,
        )
        comparisons.append(
            {
                "run_name": run.name,
                "run_url": run.url,
                "metrics": summary_metrics,
                "learning_dynamics": {
                    "step_to_threshold": dynamics.get("step_to_threshold"),
                    "late_loss_avg": dynamics.get("late_loss_avg"),
                    "late_loss_volatility_std": dynamics.get("late_loss_volatility_std"),
                },
            }
        )
        if len(comparisons) >= max_runs:
            break
    return comparisons


def _build_analysis_observations(
    *,
    metric_keys: tuple[str, ...],
    current_metrics: dict[str, object],
    current_dynamics: dict[str, object],
    recent_ferries: list[dict[str, object]],
) -> list[str]:
    observations: list[str] = []
    if not recent_ferries:
        return observations

    for key in metric_keys:
        current_val = _as_float(current_metrics.get(key))
        recent_vals = [_as_float(entry["metrics"].get(key)) for entry in recent_ferries]
        recent_vals = [value for value in recent_vals if value is not None]
        if current_val is None or not recent_vals:
            continue

        best_recent = min(recent_vals)
        mean_recent = statistics.fmean(recent_vals)
        delta_best = current_val - best_recent
        delta_mean = current_val - mean_recent

        observations.append(
            "final metric "
            f"`{key}`: current `{current_val:.5f}` vs best recent `{best_recent:.5f}` "
            f"(delta `{delta_best:+.5f}`), recent mean `{mean_recent:.5f}` "
            f"(delta `{delta_mean:+.5f}`)"
        )

    current_step_map = current_dynamics.get("step_to_threshold")
    if isinstance(current_step_map, dict):
        for threshold_key, current_step_raw in current_step_map.items():
            current_step = _as_float(current_step_raw)
            recent_steps: list[float] = []
            for entry in recent_ferries:
                behavior = entry.get("learning_dynamics", {})
                if not isinstance(behavior, dict):
                    continue
                step_map = behavior.get("step_to_threshold")
                if not isinstance(step_map, dict):
                    continue
                step_val = _as_float(step_map.get(threshold_key))
                if step_val is not None:
                    recent_steps.append(step_val)

            if current_step is None and recent_steps:
                observations.append(
                    f"behavior `{threshold_key}`: current run did not hit threshold; "
                    f"{len(recent_steps)}/{len(recent_ferries)} recent runs did"
                )
            elif current_step is not None and recent_steps:
                mean_recent_step = statistics.fmean(recent_steps)
                observations.append(
                    f"behavior `{threshold_key}` step: current `{int(current_step)}` vs recent mean "
                    f"`{mean_recent_step:.0f}` (delta `{current_step - mean_recent_step:+.0f}`)"
                )

    current_late = _as_float(current_dynamics.get("late_loss_avg"))
    recent_late = [_as_float(entry["learning_dynamics"].get("late_loss_avg")) for entry in recent_ferries]
    recent_late = [value for value in recent_late if value is not None]
    if current_late is not None and recent_late:
        mean_recent_late = statistics.fmean(recent_late)
        observations.append(
            "behavior `late_loss_avg`: "
            f"current `{current_late:.5f}` vs recent mean `{mean_recent_late:.5f}` "
            f"(delta `{current_late - mean_recent_late:+.5f}`)"
        )

    current_vol = _as_float(current_dynamics.get("late_loss_volatility_std"))
    recent_vol = [_as_float(entry["learning_dynamics"].get("late_loss_volatility_std")) for entry in recent_ferries]
    recent_vol = [value for value in recent_vol if value is not None]
    if current_vol is not None and recent_vol:
        mean_recent_vol = statistics.fmean(recent_vol)
        observations.append(
            "behavior `late_loss_volatility_std`: "
            f"current `{current_vol:.5f}` vs recent mean `{mean_recent_vol:.5f}` "
            f"(delta `{current_vol - mean_recent_vol:+.5f}`)"
        )

    return observations


def analyze_run(
    run_ref: str,
    metric_keys: tuple[str, ...],
    *,
    thresholds: tuple[float, ...],
    window_points: int,
    trajectory_points: int,
    recent_ferries: int,
) -> AnalysisResult:
    run_path = _normalize_run_path(run_ref)
    entity, project, _run_name = run_path.split("/", 2)
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    metrics: dict[str, float | int | str | None] = {key: run.summary.get(key) for key in metric_keys}
    history_points = _extract_train_loss_points(run)
    learning_dynamics = _compute_learning_dynamics(
        history_points,
        thresholds=thresholds,
        window_points=window_points,
        trajectory_points=trajectory_points,
    )
    comparisons = _fetch_recent_ferry_comparisons(
        api,
        entity=entity,
        project=project,
        exclude_run_name=run.name,
        max_runs=recent_ferries,
        metric_keys=metric_keys,
        thresholds=thresholds,
    )
    observations = _build_analysis_observations(
        metric_keys=metric_keys,
        current_metrics=metrics,
        current_dynamics=learning_dynamics,
        recent_ferries=comparisons,
    )
    return AnalysisResult(
        run_path=run_path,
        run_url=run.url,
        run_name=run.name,
        metrics=metrics,
        learning_dynamics=learning_dynamics,
        recent_ferries=comparisons,
        analysis_observations=observations,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="W&B run URL or run path (`entity/project/run_id_or_name`).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. `markdown` is intended for copy/paste into the ferry log.",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        dest="metric_keys",
        help=(
            "Metric key to extract. Repeat to override defaults. "
            "Defaults to eval/paloma/c4_en/bpb, eval/bpb, eval/uncheatable_eval/bpb."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=5,
        help="Decimal precision for numeric output (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="Loss threshold for step-to-threshold analysis. Repeatable.",
    )
    parser.add_argument(
        "--window-points",
        type=int,
        default=500,
        help="Point count for early/late averages and late volatility (default: 500).",
    )
    parser.add_argument(
        "--trajectory-points",
        type=int,
        default=20,
        help="How many trajectory points to print (default: 20).",
    )
    parser.add_argument(
        "--recent-ferries",
        type=int,
        default=3,
        help="How many recent completed daily ferries to compare (default: 3, set 0 to disable).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    metric_keys = tuple(args.metric_keys) if args.metric_keys else DEFAULT_METRIC_KEYS

    thresholds = tuple(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS

    try:
        result = analyze_run(
            args.run,
            metric_keys,
            thresholds=thresholds,
            window_points=args.window_points,
            trajectory_points=args.trajectory_points,
            recent_ferries=args.recent_ferries,
        )
    except Exception as exc:
        print(f"Failed to analyze run: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(
            json.dumps(
                {
                    "run_path": result.run_path,
                    "run_name": result.run_name,
                    "run_url": result.run_url,
                    "metrics": result.metrics,
                    "learning_dynamics": result.learning_dynamics,
                    "recent_ferries": result.recent_ferries,
                    "analysis_observations": result.analysis_observations,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"Run: `{result.run_name}`")
    print(f"W&B: {result.run_url}")
    print("Metrics:")
    for key in metric_keys:
        print(f"- `{key}`: `{_fmt_value(result.metrics.get(key), args.precision)}`")

    dynamics = result.learning_dynamics
    if dynamics.get("available"):
        print("Learning dynamics (optional, if interesting):")
        trajectory = dynamics["train_loss_trajectory"]
        print(f"- `train-loss trajectory` ({len(trajectory)} sampled points):")
        for step, loss in trajectory:
            print(f"  - step `{step}`: loss `{_fmt_value(loss, args.precision)}`")
        print("- `step-to-threshold`:")
        for threshold_key, step in dynamics["step_to_threshold"].items():
            print(f"  - `{threshold_key}`: `{_fmt_value(step, args.precision)}`")
        print(
            f"- `early loss avg` (first {dynamics['window_points']} points): "
            f"`{_fmt_value(dynamics['early_loss_avg'], args.precision)}`"
        )
        print(
            f"- `late loss avg` (last {dynamics['window_points']} points): "
            f"`{_fmt_value(dynamics['late_loss_avg'], args.precision)}`"
        )
        print(
            "- `late-stage volatility` (std over last "
            f"{dynamics['window_points']} points): "
            f"`{_fmt_value(dynamics['late_loss_volatility_std'], args.precision)}`"
        )
    else:
        print("Learning dynamics (optional, if interesting):")
        print(f"- unavailable: {dynamics.get('reason', 'no reason provided')}")

    if result.recent_ferries:
        print(f"Recent completed ferries (optional context, latest {len(result.recent_ferries)}):")
        for entry in result.recent_ferries:
            print(f"- `{entry['run_name']}`: {entry['run_url']}")
            for key in metric_keys:
                print(f"  - `{key}`: `{_fmt_value(entry['metrics'].get(key), args.precision)}`")
            behavior = entry["learning_dynamics"]
            print(
                "  - behavior: "
                f"step<=4.0 `{_fmt_value(behavior['step_to_threshold'].get('loss<=4.0'), args.precision)}`, "
                f"late_avg `{_fmt_value(behavior['late_loss_avg'], args.precision)}`, "
                f"late_std `{_fmt_value(behavior['late_loss_volatility_std'], args.precision)}`"
            )

    print("Analysis (optional, if interesting):")
    if result.analysis_observations:
        for observation in result.analysis_observations:
            print(f"- {observation}")
    else:
        print("- no comparative observations available (missing recent runs or required metrics)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
