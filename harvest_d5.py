#!/usr/bin/env python
"""Summarize a grug MoE run's json_logger metrics from `iris job logs`.

Usage: harvest_d5.py <job-path> [--raw <saved-log-file>]
Prints MFU percentiles over the steady window, the drop series, and the loss tail.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys

METRIC_LINE = re.compile(r'(\{"tracker": "json_logger".*)$')


def log_lines(job: str, raw: str | None) -> list[str]:
    if raw:
        with open(raw) as handle:
            return handle.readlines()
    proc = subprocess.run(
        # --max-lines matters: the server default is 1000 lines, which silently truncates a
        # 120-step run to its first ~20 steps and biases every "steady state" number.
        [".venv/bin/iris", "--cluster=marin", "job", "logs", job, "--max-lines", "400000"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--raw")
    parser.add_argument("--warmup", type=int, default=10, help="steps to drop before percentiles")
    args = parser.parse_args()

    steps: dict[int, dict[str, float]] = {}
    for line in log_lines(args.job, args.raw):
        match = METRIC_LINE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "log":
            continue
        step = payload.get("step")
        metrics = payload.get("metrics") or {}
        if step is None:
            continue
        entry = steps.setdefault(int(step), {})
        for key in ("throughput/mfu", "throughput/tokens_per_second", "throughput/duration",
                    "moe/drop_fraction", "train/loss"):
            if key in metrics:
                entry[key.split("/")[-1]] = float(metrics[key])

    if not steps:
        print("no metric rows found", file=sys.stderr)
        return 1

    ordered = sorted(steps)
    mfus = [steps[s]["mfu"] for s in ordered if s >= args.warmup and "mfu" in steps[s]]
    print(f"steps {ordered[0]}..{ordered[-1]} ({len(ordered)} rows), mfu samples past warmup {len(mfus)}")
    if len(mfus) >= 2:
        quantiles = statistics.quantiles(mfus, n=10)
        print(
            f"MFU  p10 {quantiles[0]:.3f}  p50 {statistics.median(mfus):.3f}  p90 {quantiles[8]:.3f}  "
            f"mean {statistics.mean(mfus):.3f}  sd {statistics.stdev(mfus):.3f}"
        )
    toks = [steps[s].get("tokens_per_second") for s in ordered if steps[s].get("tokens_per_second")]
    durs = [steps[s].get("duration") for s in ordered if steps[s].get("duration")]
    if toks:
        print(f"tok/s median {statistics.median(toks):,.0f}   step time median {statistics.median(durs):.3f}s")
    drops = [(s, steps[s]["drop_fraction"]) for s in ordered if "drop_fraction" in steps[s]]
    if drops:
        print("drop_fraction series (every 10th step, plus the last 5):")
        sample = [d for d in drops if d[0] % 10 == 0] + drops[-5:]
        print("  " + "  ".join(f"{s}:{v:.4f}" for s, v in sample))
    losses = [(s, steps[s]["loss"]) for s in ordered if "loss" in steps[s]]
    if losses:
        print(f"loss: first {losses[0][1]:.4f} @ {losses[0][0]}, last {losses[-1][1]:.4f} @ {losses[-1][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
