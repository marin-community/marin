# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit-v4: enumerate shards with stale `_done` markers.

A shard is "stale-done" when `_done` exists in its regional bucket but at least
one PR in the shard's `sampling_plan.json` (under `languages.<lang>.instance_ids`)
has fewer than 100 rollouts in `rollouts.json`.

Reading sampling_plan.json (rather than slicing the upstream HF dataset) avoids
two pitfalls: (1) HF dataset revision drift since the workers ran, (2) the
worker's actual PR set is language-filtered and not a clean `[idx::1260]` stride
of the dataset.

Output: gs://marin-us-east5/datasets/swe-zero-stale-done-shards.json
{
  "n_shards": 1260,
  "n_done": <count of shards with _done marker>,
  "n_stale_done": <count flagged as stale>,
  "n_legitimately_done": <count where _done is correct>,
  "n_undone": <count of shards without _done>,
  "stale_done_shard_indices": [int, ...],
  "by_partition": {
    "P0_us-east5": {n_done, n_stale_done, n_legitimately_done, n_undone},
    "P1_us-east1": {...},
    ...
  }
}
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logger = logging.getLogger(__name__)

REGIONS = {0: "us-east5", 1: "us-east1", 2: "us-west4", 3: "us-central1"}
TOTAL_SHARDS = 1260
TARGET_ROLLOUTS_PER_PR = 100


def _shard_path(idx: int) -> str:
    region = REGIONS[idx % 4]
    return f"gs://marin-{region}/experiments/swe_zero_100b/per_pr/shard_{idx:03d}_of_{TOTAL_SHARDS:03d}"


def _read_source_prs_from_plan(plan_path: str, fs: fsspec.AbstractFileSystem) -> list[str]:
    """Extract per-shard PR list from sampling_plan.json.

    Schema: {"languages": {<lang>: {"instance_ids": [...], ...}, "__shard__": {...}}}
    Concatenate `instance_ids` across all language sub-dicts (excluding `__shard__`).
    """
    if not fs.exists(plan_path):
        return []
    with fsspec.open(plan_path, "rb") as f:
        plan = json.load(f)
    langs = plan.get("languages", {}) if isinstance(plan, dict) else {}
    ids: list[str] = []
    for lang_key, lang_val in langs.items():
        if lang_key == "__shard__" or not isinstance(lang_val, dict):
            continue
        for iid in lang_val.get("instance_ids", []) or []:
            if iid:
                ids.append(iid)
    return ids


def _audit_shard(idx: int) -> dict:
    base = _shard_path(idx)
    fs = fsspec.filesystem("gs")
    done_marker = f"{base}/_done"
    rolls_path = f"{base}/rollouts.json"
    plan_path = f"{base}/sampling_plan.json"

    has_done = fs.exists(done_marker)
    source_prs = _read_source_prs_from_plan(plan_path, fs)

    # Count rollouts per instance_id from rollouts.json (streaming-friendly).
    counts: Counter = Counter()
    if fs.exists(rolls_path):
        with fsspec.open(rolls_path, "rb") as f:
            rolls = json.load(f)
        if isinstance(rolls, list):
            for r in rolls:
                if isinstance(r, dict):
                    iid = r.get("instance_id")
                    if iid:
                        counts[iid] += 1
        del rolls  # free the big list before next shard

    # Coverage on this shard's source PR set
    n_at_target = sum(1 for iid in source_prs if counts.get(iid, 0) >= TARGET_ROLLOUTS_PER_PR)
    n_missing = sum(1 for iid in source_prs if counts.get(iid, 0) == 0)
    n_partial = len(source_prs) - n_at_target - n_missing
    is_complete = bool(source_prs) and n_at_target == len(source_prs)
    is_stale_done = has_done and not is_complete

    return {
        "shard": idx,
        "region": REGIONS[idx % 4],
        "n_source": len(source_prs),
        "n_at_target": n_at_target,
        "n_partial": n_partial,
        "n_missing": n_missing,
        "has_done": has_done,
        "has_plan": bool(source_prs),
        "is_complete": is_complete,
        "is_stale_done": is_stale_done,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_audit_shard, i): i for i in range(TOTAL_SHARDS)}
        for j, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "shard": futures[fut],
                    "region": REGIONS[futures[fut] % 4],
                    "error": str(e),
                    "is_stale_done": False,
                    "has_done": False,
                    "is_complete": False,
                }
            results.append(r)
            if (j + 1) % 100 == 0:
                logger.info("  %d/%d shards audited", j + 1, TOTAL_SHARDS)

    results.sort(key=lambda r: r["shard"])
    n_done = sum(1 for r in results if r.get("has_done"))
    n_stale = sum(1 for r in results if r.get("is_stale_done"))
    n_legit_done = sum(1 for r in results if r.get("has_done") and r.get("is_complete"))
    n_undone = sum(1 for r in results if not r.get("has_done"))
    stale_indices = [r["shard"] for r in results if r.get("is_stale_done")]

    by_partition: dict[str, dict] = {}
    for p in range(4):
        sub = [r for r in results if r["shard"] % 4 == p]
        by_partition[f"P{p}_{REGIONS[p]}"] = {
            "n_shards": len(sub),
            "n_done": sum(1 for r in sub if r.get("has_done")),
            "n_stale_done": sum(1 for r in sub if r.get("is_stale_done")),
            "n_legitimately_done": sum(1 for r in sub if r.get("has_done") and r.get("is_complete")),
            "n_undone": sum(1 for r in sub if not r.get("has_done")),
        }

    summary = {
        "n_shards": TOTAL_SHARDS,
        "n_done": n_done,
        "n_stale_done": n_stale,
        "n_legitimately_done": n_legit_done,
        "n_undone": n_undone,
        "stale_done_shard_indices": stale_indices,
        "by_partition": by_partition,
        "shards": results,
    }

    out_path = "gs://marin-us-east5/datasets/swe-zero-stale-done-shards.json"
    with fsspec.open(out_path, "w") as f:
        json.dump(summary, f)
    logger.info("Wrote %s", out_path)

    print("=" * 70)
    print("STALE _done AUDIT SUMMARY")
    print(f"  shards total: {TOTAL_SHARDS}")
    print(f"  shards with _done: {n_done}")
    print(f"  shards stale-done (need cleanup): {n_stale}")
    print(f"  shards legitimately done: {n_legit_done}")
    print(f"  shards without _done: {n_undone}")
    print()
    for k, v in by_partition.items():
        print(
            f"  {k}: done={v['n_done']} stale={v['n_stale_done']} "
            f"legit_done={v['n_legitimately_done']} undone={v['n_undone']}"
        )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
