# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit untouched PRs across all 1260 shards.

For each shard, compute:
  - source_prs = PRs listed in sampling_plan.json
  - touched_prs = unique instance_ids in rollouts.json
  - untouched = source_prs - touched_prs

Writes per-shard counts to gs://marin-us-east5/datasets/swe-zero-untouched-prs.json.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

REGIONS = {0: "us-east5", 1: "us-east1", 2: "us-west4", 3: "us-central1"}
TOTAL = 1260


def audit_one(idx: int) -> dict:
    region = REGIONS[idx % 4]
    base = f"gs://marin-{region}/experiments/swe_zero_100b/per_pr/shard_{idx:03d}_of_1260"
    fs = fsspec.filesystem("gs")
    plan_path = f"{base}/sampling_plan.json"
    rolls_path = f"{base}/rollouts.json"
    if not fs.exists(plan_path):
        return {"shard": idx, "status": "no_plan", "source": 0, "touched": 0, "untouched": []}
    with fsspec.open(plan_path, "rb") as f:
        plan = json.load(f)
    source_ids = {p.get("instance_id") for p in (plan if isinstance(plan, list) else plan.get("prs", []))}
    source_ids.discard(None)
    touched_ids = set()
    if fs.exists(rolls_path):
        with fsspec.open(rolls_path, "rb") as f:
            rolls = json.load(f)
        if isinstance(rolls, list):
            for r in rolls:
                if isinstance(r, dict):
                    iid = r.get("instance_id")
                    if iid:
                        touched_ids.add(iid)
    untouched = sorted(source_ids - touched_ids)
    return {
        "shard": idx,
        "status": "ok",
        "source": len(source_ids),
        "touched": len(touched_ids),
        "untouched": untouched,
    }


def main():
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(audit_one, i): i for i in range(TOTAL)}
        for j, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception as e:
                r = {"shard": futures[fut], "status": f"error: {e}", "source": 0, "touched": 0, "untouched": []}
            results.append(r)
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{TOTAL} shards audited", flush=True)
    results.sort(key=lambda r: r["shard"])
    total_source = sum(r["source"] for r in results)
    total_touched = sum(r["touched"] for r in results)
    total_untouched = sum(len(r["untouched"]) for r in results)
    summary = {
        "n_shards": TOTAL,
        "total_source_prs": total_source,
        "total_touched_prs": total_touched,
        "total_untouched_prs": total_untouched,
        "coverage_pct": (total_touched / total_source * 100) if total_source else 0,
        "by_partition": {},
        "shards": results,
    }
    for p in range(4):
        sub = [r for r in results if r["shard"] % 4 == p]
        summary["by_partition"][f"P{p}_{REGIONS[p]}"] = {
            "n_shards": len(sub),
            "source": sum(r["source"] for r in sub),
            "touched": sum(r["touched"] for r in sub),
            "untouched": sum(len(r["untouched"]) for r in sub),
        }
    out_path = "gs://marin-us-east5/datasets/swe-zero-untouched-prs.json"
    with fsspec.open(out_path, "w") as f:
        json.dump(summary, f)
    print(f"\nWrote {out_path}")
    print(
        f"source={total_source:,} touched={total_touched:,} "
        f"untouched={total_untouched:,} coverage={summary['coverage_pct']:.2f}%"
    )
    for k, v in summary["by_partition"].items():
        print(f"  {k}: source={v['source']} touched={v['touched']} untouched={v['untouched']}")


if __name__ == "__main__":
    sys.exit(main())
