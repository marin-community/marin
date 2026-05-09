# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-region throughput metrics emitted by multiswarm workers.

Reads `gs://marin-{region}/experiments/swe_zero_100b/throughput/{worker}_{iter}.json`
files (written by `_emit_throughput_metric` in run_swe_zero_swarm_multiregion.py)
and computes total tokens/sec across the pipeline + per-region breakdown for an
arbitrary window.

Usage::

    python scripts/throughput_stats.py [--window-min 10] [--regions us-east5,us-east1,us-west4,us-central1]

Output is a JSON object:

    {
        "window_min": 10,
        "since_ts": <unix>,
        "by_region": {
            "us-east5": {"rollouts": ..., "tokens": ..., "tokens_per_sec": ..., "n_iters": ..., "n_workers": ...},
            ...
        },
        "total": {"rollouts": ..., "tokens": ..., "tokens_per_sec": ..., "n_iters": ...}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import fsspec

DEFAULT_REGIONS = ["us-east5", "us-east1", "us-west4", "us-central1"]


def aggregate(regions: list[str], window_min: int) -> dict:
    now = time.time()
    cutoff = now - window_min * 60
    by_region: dict[str, dict] = {}

    fs = fsspec.filesystem("gs")
    total_rollouts = 0
    total_tokens = 0
    total_iters = 0

    for region in regions:
        prefix = f"gs://marin-{region}/experiments/swe_zero_100b/throughput/"
        try:
            files = fs.ls(prefix)
        except Exception:
            files = []

        rollouts = 0
        tokens = 0
        n_iters = 0
        workers: set[int] = set()
        for f in files:
            if not f.endswith(".json"):
                continue
            # Cheap mtime filter via blob stat
            try:
                info = fs.info(f)
            except Exception:
                continue
            mtime = info.get("mtime") or info.get("updated")
            if mtime is None:
                continue
            mtime_ts = mtime.timestamp() if hasattr(mtime, "timestamp") else mtime
            if mtime_ts < cutoff:
                continue
            try:
                with fs.open(f, "rb") as fh:
                    metric = json.load(fh)
            except Exception:
                continue
            if int(metric.get("ts", 0)) < cutoff:
                continue
            rollouts += int(metric.get("rollouts_produced", 0))
            tokens += int(metric.get("trajectory_tokens_estimate", 0))
            n_iters += 1
            workers.add(int(metric.get("worker_id", -1)))

        by_region[region] = {
            "rollouts": rollouts,
            "tokens": tokens,
            "tokens_per_sec": tokens / (window_min * 60) if window_min > 0 else 0,
            "n_iters": n_iters,
            "n_workers": len(workers),
        }
        total_rollouts += rollouts
        total_tokens += tokens
        total_iters += n_iters

    return {
        "window_min": window_min,
        "since_ts": cutoff,
        "by_region": by_region,
        "total": {
            "rollouts": total_rollouts,
            "tokens": total_tokens,
            "tokens_per_sec": total_tokens / (window_min * 60) if window_min > 0 else 0,
            "n_iters": total_iters,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-min", type=int, default=10)
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--format", choices=["json", "human"], default="human")
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    result = aggregate(regions, args.window_min)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"=== throughput over last {args.window_min} min ===")
        print(f"{'region':<14} {'tokens/s':>10} {'rollouts':>10} {'iters':>7} {'workers':>8}")
        for region, m in result["by_region"].items():
            print(
                f"{region:<14} {m['tokens_per_sec']:>10,.0f} {m['rollouts']:>10,} {m['n_iters']:>7} {m['n_workers']:>8}"
            )
        t = result["total"]
        print(f"{'TOTAL':<14} {t['tokens_per_sec']:>10,.0f} {t['rollouts']:>10,} {t['n_iters']:>7} -")
    return 0


if __name__ == "__main__":
    sys.exit(main())
