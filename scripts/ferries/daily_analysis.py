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
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import wandb

DEFAULT_METRIC_KEYS = (
    "eval/paloma/c4_en/bpb",
    "eval/bpb",
    "eval/uncheatable_eval/bpb",
)


@dataclass(frozen=True)
class AnalysisResult:
    run_path: str
    run_url: str
    run_name: str
    metrics: dict[str, float | int | str | None]


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


def analyze_run(run_ref: str, metric_keys: tuple[str, ...]) -> AnalysisResult:
    run_path = _normalize_run_path(run_ref)
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    metrics: dict[str, float | int | str | None] = {key: run.summary.get(key) for key in metric_keys}
    return AnalysisResult(
        run_path=run_path,
        run_url=run.url,
        run_name=run.name,
        metrics=metrics,
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
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    metric_keys = tuple(args.metric_keys) if args.metric_keys else DEFAULT_METRIC_KEYS

    try:
        result = analyze_run(args.run, metric_keys)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
