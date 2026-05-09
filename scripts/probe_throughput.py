# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External throughput probe — measures actual tokens/sec by diffing
`rollouts_resume.json` file sizes across two GCS snapshots N seconds apart.

Bypasses the in-process metric aggregator's blind spot (workers preempted
mid-iter never emit `[METRICS]`). This script only reads GCS metadata, so it
works without modifying or restarting any running jobs.

Conversion constants are calibrated against round-3 aggregation
(8.2M rollouts, 74.4B Qwen3 tokens, ~228 GB raw JSON):

    bytes_per_rollout = 28_000   # ~28 KB raw JSON per rollout
    tokens_per_rollout = 9_073   # 74.4B / 8.2M

These are averages — large rollouts vary 5-50 KB.

Usage::

    python scripts/probe_throughput.py [--interval 60] [--regions ...]

Output JSON (or human-readable):
    {
      "interval_s": 60,
      "by_region": {
        "us-east1": {
          "delta_bytes": 1.2e9,
          "active_shards": 5,
          "tokens_per_sec": 6_500_000,
          ...
        }, ...
      },
      "total": {"tokens_per_sec": ..., ...}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import fsspec

DEFAULT_REGIONS = ["us-east5", "us-east1", "us-west4", "us-central1"]
BYTES_PER_ROLLOUT = 28_000
TOKENS_PER_ROLLOUT = 9_073


def snapshot(regions: list[str], fs: fsspec.AbstractFileSystem) -> dict[str, dict[str, int]]:
    """Return {region: {shard_path: size_bytes}} for every rollouts_resume.json
    in BOTH the native per_pr/ dir AND the cross-region imported/ dir.

    A region's tokens/sec credit goes to the COMPUTE region (where the work
    physically happens), not the claim region — so an imported file at
    gs://marin-us-east5/.../imported/shard_X/ counts toward us-east5 even
    though shard_X is part of us-central1's claim partition.
    """
    out: dict[str, dict[str, int]] = {}
    for region in regions:
        sizes: dict[str, int] = {}
        for sub in ("per_pr", "imported"):
            prefix = f"marin-{region}/experiments/swe_zero_100b/{sub}"
            try:
                paths = fs.glob(f"{prefix}/shard_*/rollouts_resume.json")
            except Exception:
                paths = []
            for p in paths:
                try:
                    info = fs.info(p)
                    sizes[p] = int(info.get("size", 0))
                except Exception:
                    continue
        out[region] = sizes
    return out


def diff(t0: dict[str, dict[str, int]], t1: dict[str, dict[str, int]]) -> dict[str, dict]:
    by_region: dict[str, dict] = {}
    for region in t0:
        s0 = t0[region]
        s1 = t1.get(region, {})
        delta_bytes = 0
        active = 0
        for path, size1 in s1.items():
            size0 = s0.get(path, 0)
            d = size1 - size0
            if d > 0:
                delta_bytes += d
                active += 1
        by_region[region] = {
            "delta_bytes": delta_bytes,
            "active_shards": active,
        }
    return by_region


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="seconds between snapshots")
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--format", choices=["json", "human"], default="human")
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    fs = fsspec.filesystem("gs")

    print("snapshot 1/2 (t=0)...", file=sys.stderr)
    t0 = snapshot(regions, fs)
    print(f"  scanned {sum(len(s) for s in t0.values())} rollouts_resume.json files", file=sys.stderr)
    print(f"sleeping {args.interval}s...", file=sys.stderr)
    time.sleep(args.interval)
    print(f"snapshot 2/2 (t={args.interval})...", file=sys.stderr)
    t1 = snapshot(regions, fs)

    deltas = diff(t0, t1)
    total_bytes = sum(d["delta_bytes"] for d in deltas.values())
    total_active = sum(d["active_shards"] for d in deltas.values())
    interval = args.interval

    for _region, d in deltas.items():
        bps = d["delta_bytes"] / interval
        rps = bps / BYTES_PER_ROLLOUT
        tps = rps * TOKENS_PER_ROLLOUT
        d["bytes_per_sec"] = bps
        d["rollouts_per_sec"] = rps
        d["tokens_per_sec"] = tps

    total_bps = total_bytes / interval
    total_rps = total_bps / BYTES_PER_ROLLOUT
    total_tps = total_rps * TOKENS_PER_ROLLOUT

    result = {
        "interval_s": interval,
        "by_region": deltas,
        "total": {
            "delta_bytes": total_bytes,
            "active_shards": total_active,
            "bytes_per_sec": total_bps,
            "rollouts_per_sec": total_rps,
            "tokens_per_sec": total_tps,
        },
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== throughput over {interval}s window (GCS file-size diff) ===")
        print(f"{'region':<14} {'tokens/s':>14} {'rollouts/s':>11} {'MB/s':>8} {'active':>7}")
        for region, d in deltas.items():
            print(
                f"{region:<14} {d['tokens_per_sec']:>14,.0f} {d['rollouts_per_sec']:>11,.1f} "
                f"{d['bytes_per_sec']/1e6:>8,.1f} {d['active_shards']:>7}"
            )
        print(f"{'TOTAL':<14} {total_tps:>14,.0f} {total_rps:>11,.1f} " f"{total_bps/1e6:>8,.1f} {total_active:>7}")
        print(
            f"\n(constants: {BYTES_PER_ROLLOUT} B/rollout, "
            f"{TOKENS_PER_ROLLOUT} tokens/rollout - calibrated against round-3)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
