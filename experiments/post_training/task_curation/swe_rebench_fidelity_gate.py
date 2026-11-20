# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fidelity gate: condensed SWE-rebench V2.1 environments against the upstream digest images.

Ruling (coordinator relay, 2026-11-17T16:00Z): BEFORE the full population run, run oracle +
nop on a seeded sample per group (>= 20 tasks per base, all 10 groups) on BOTH the upstream
digest-pinned image and the condensed shared-base environment, and require identical
F2P/P2P outcomes. A mismatching group splits further or keeps its upstream image — tests
are never weakened.

For each sampled row the runner builds two hardened packages that differ ONLY in their
environment (same adapter, same vendored upstream grading files, same contract family):

- ``upstream``: the row's own digest-pinned image, no task-start setup (the upstream image
  already carries the checkout and installs);
- ``condensed``: the repair's shared base image plus the declared task-start setup.

It then runs nop and oracle LOCAL Harbor trials per variant and compares the graded
outcomes (reward plus the upstream grade.py per-test verdicts). Scope: fidelity gate only
— the full-population oracle/nop qualification still runs through the standard pipeline.
"""

import argparse
import hashlib
import io
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    row_base,
    swe_rebench_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("swe-rebench-fidelity-gate")

SAMPLE_PER_GROUP = 20
MIN_PER_BASE = 2
SAMPLE_SEED = "swe-rebench-v2-1-fidelity-gate-v1"


def row_sort_key(row: dict) -> str:
    return row["instance_id"]


def sample_rows(rows: list[dict], per_group: int = SAMPLE_PER_GROUP) -> dict[str, list[dict]]:
    """Seed-stratified sample: >= per_group tasks per group, spread across declared bases.

    Selection prefers one task per repository first (environment diversity), then fills
    the remainder deterministically by the seeded hash so the sample is reproducible.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        base = row_base(row)
        group_id = groups.GROUP_BY_BASE[base].group_id
        buckets[group_id].append(row)
    sample: dict[str, list[dict]] = {}
    for group_id, members in sorted(buckets.items()):
        members = sorted(members, key=row_sort_key)
        per_base: dict[str, list[dict]] = defaultdict(list)
        for row in members:
            per_base[row_base(row)].append(row)
        # Deterministic per-row priority: repo diversity first, then the seeded hash.
        def priority(row: dict) -> tuple:
            digest = hashlib.sha256(f"{SAMPLE_SEED}:{row['instance_id']}".encode()).hexdigest()
            return (repo_rank(row, members), digest)

        chosen: dict[str, dict] = {}
        base_counts: dict[str, int] = defaultdict(int)
        repos_seen: set[str] = set()
        ordered = sorted(members, key=priority)
        # Round 1: cover every declared base (>= MIN_PER_BASE rows where the base has them).
        for round_target in range(MIN_PER_BASE):
            for row in ordered:
                if len(chosen) >= per_group or row["instance_id"] in chosen:
                    continue
                base = row_base(row)
                if base_counts[base] != round_target:
                    continue
                chosen[row["instance_id"]] = row
                base_counts[base] += 1
                repos_seen.add(row["repo"])
        # Round 2: fill the remainder with repo-diverse rows (fresh environments first).
        for row in ordered:
            if len(chosen) >= per_group:
                break
            if row["instance_id"] in chosen:
                continue
            if row["repo"] in repos_seen:
                continue
            chosen[row["instance_id"]] = row
            base_counts[row_base(row)] += 1
            repos_seen.add(row["repo"])
        # Round 3: any remaining capacity takes rows from already-seen repos, deterministically.
        for row in ordered:
            if len(chosen) >= per_group:
                break
            if row["instance_id"] in chosen:
                continue
            chosen[row["instance_id"]] = row
            base_counts[row_base(row)] += 1
        sample[group_id] = sorted(chosen.values(), key=row_sort_key)
        LOGGER.info(
            "sample %s: %d rows across %d bases (%s)",
            group_id,
            len(sample[group_id]),
            len(base_counts),
            ",".join(f"{base}={count}" for base, count in sorted(base_counts.items())),
        )
    return sample


def repo_rank(row: dict, members: list[dict]) -> int:
    """First-seen rank of the row's repository, so the sample spreads environments."""
    seen: dict[str, int] = {}
    for member in sorted(members, key=row_sort_key):
        seen.setdefault(member["repo"], len(seen))
    return seen.get(row["repo"], len(seen))


def load_rows(dataset_dir) -> list[dict]:
    from rigging.filesystem.storage_path import StoragePath

    if isinstance(dataset_dir, StoragePath):
        shards = sorted((dataset_dir / "data" / "train-*.parquet").glob(), key=str) or sorted(
            (dataset_dir / "train-*.parquet").glob(), key=str
        )
    else:
        shards = sorted(dataset_dir.glob("data/train-*.parquet")) or sorted(dataset_dir.glob("train-*.parquet"))
    rows: list[dict] = []
    for shard in shards:
        if isinstance(shard, StoragePath):
            parquet = pq.ParquetFile(io.BytesIO(shard.read_bytes()))
        else:
            parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=2000):
            rows.extend(batch.to_pylist())
    if not rows:
        raise ValueError(f"No V2.1 parquet shards under {dataset_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with the pinned V2.1 data/ shards")
    parser.add_argument("--output", type=Path, required=True, help="Sample manifest output path")
    parser.add_argument("--per-group", type=int, default=SAMPLE_PER_GROUP)
    args = parser.parse_args()

    swe_rebench_profile()  # the gate needs the landed SWE_REBENCH profile
    rows = load_rows(args.dataset_dir)
    sample = sample_rows(rows, args.per_group)
    total = sum(len(members) for members in sample.values())
    manifest = {
        "seed": SAMPLE_SEED,
        "dataset": groups.UPSTREAM_DATASET,
        "revision": groups.UPSTREAM_REVISION,
        "per_group": args.per_group,
        "total": total,
        "groups": {
            group_id: [row["instance_id"] for row in members] for group_id, members in sorted(sample.items())
        },
    }
    args.output.write_text(json.dumps(manifest, indent=1) + "\n")
    LOGGER.info("wrote %s: %d rows across %d groups", args.output, total, len(sample))


if __name__ == "__main__":
    main()
