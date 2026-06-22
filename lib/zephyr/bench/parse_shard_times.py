# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse per-shard execution times (scatter vs reduce) from a normalize coord's logs.

Reads `iris job logs <coord> --max-lines 0` and associates each
`Shard N done in Xs` with the worker's most recent `Executing stage stageK`
line, bucketing into scatter (stage0) vs reduce (stage1). Reports per-stage
sum/median/count — the clean shard-level execution-time metric (independent of
worker idle/scheduling, unlike worker-uptime sum_task_wall).

Usage:
  uv run python lib/zephyr/bench/parse_shard_times.py <normalize-coord-job-id> [--config ...]
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess

_STAGE1_RE = re.compile(r"Executing stage stage1")
_DONE_RE = re.compile(r"Shard \d+ done in ([0-9.]+)s")


def _classify(lines: list[str]) -> dict[str, list[float]]:
    """Bucket each `Shard N done in Xs` into scatter/reduce using the stage barrier.

    Stages are barriers, so every scatter `done` precedes the first stage1 marker
    and every reduce `done` follows it — robust to missed early per-worker logs.
    """
    times: dict[str, list[float]] = {"stage0": [], "stage1": []}
    reduce_started = False
    for line in lines:
        if _STAGE1_RE.search(line):
            reduce_started = True
        d = _DONE_RE.search(line)
        if d:
            times["stage1" if reduce_started else "stage0"].append(float(d.group(1)))
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coord_job_id", help="normalize coord job id, OR a path to a streamed log file with --file")
    ap.add_argument("--config", default="lib/iris/config/marin.yaml")
    ap.add_argument("--file", action="store_true", help="treat coord_job_id as a local log file path")
    args = ap.parse_args()

    if args.file:
        with open(args.coord_job_id) as f:
            lines = f.read().splitlines()
    else:
        out = subprocess.run(
            [
                "uv",
                "run",
                "iris",
                "--config",
                args.config,
                "job",
                "logs",
                args.coord_job_id,
                "--max-lines",
                "0",
                "--no-tail",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        ).stdout
        lines = out.splitlines()

    times = _classify(lines)

    def stat(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        return (
            f"n={len(xs):3d} sum={sum(xs):8.1f}s median={statistics.median(xs):6.1f}s mean={statistics.mean(xs):6.1f}s"
        )

    s0, s1 = times["stage0"], times["stage1"]
    print(f"scatter(stage0): {stat(s0)}")
    print(f"reduce (stage1): {stat(s1)}")
    print(f"TOTAL shard-seconds: {sum(s0) + sum(s1):.1f}  (scatter={sum(s0):.1f} reduce={sum(s1):.1f})")


if __name__ == "__main__":
    main()
