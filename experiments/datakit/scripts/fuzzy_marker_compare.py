# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two co-partitioned duplicate-marker trees, per source.

Both trees name one file per normalized shard, so the comparison is a shard-wise
walk with no shuffle. Reports, for every source, how many documents each tree
removes and how much the two agree, which is what says whether a rule change
moved recall or only moved which document of a pair survives.

    uv run iris --cluster=cw-us-east-02a job run --no-wait --priority batch \
        --cpu 16 --memory 64GB -- python experiments/datakit/scripts/fuzzy_marker_compare.py \
            --prefix s3://.../marin \
            --baseline datakit/verify_fuzzy_dups_c757e4f0 \
            --candidate user/rav/dedup/cluster_text/v3/verified \
            --out s3://.../user/rav/dedup/compare/v1
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)


def _ids(path: str) -> set[str]:
    if not StoragePath(path).exists():
        return set()
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows == 0:
            return set()
        return set(parquet.read(columns=["id"]).column("id").to_pylist())


def _compare_group(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in task["entries"]:
        baseline = _ids(entry["baseline"])
        candidate = _ids(entry["candidate"])
        rows.append(
            {
                "source_tag": entry["source_tag"],
                "baseline": len(baseline),
                "candidate": len(candidate),
                "both": len(baseline & candidate),
                "baseline_only": len(baseline - candidate),
                "candidate_only": len(candidate - baseline),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--baseline", required=True, help="Marker tree to compare against")
    parser.add_argument("--candidate", required=True, help="New marker tree")
    parser.add_argument("--manifest", required=True, help="cluster_text manifest.json naming every shard")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards-per-task", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--worker-cpu", type=float, default=16)
    parser.add_argument("--worker-ram", default="96g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()

    manifest = json.loads(StoragePath(args.manifest).read_bytes())
    entries = [
        {
            "source_tag": shard["source_tag"],
            "baseline": prefix_join(
                prefix_join(prefix_join(args.prefix, args.baseline), f"outputs/{shard['source_tag']}"),
                shard["basename"],
            ),
            "candidate": prefix_join(
                prefix_join(prefix_join(args.prefix, args.candidate), f"outputs/{shard['source_tag']}"),
                shard["basename"],
            ),
        }
        for shard in manifest["shards"]
    ]
    tasks = [
        {"entries": entries[start : start + args.shards_per_task]}
        for start in range(0, len(entries), args.shards_per_task)
    ]
    logger.info("Comparing %d shards in %d tasks", len(entries), len(tasks))

    context = ZephyrContext(
        name="fuzzy-marker-compare",
        resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk="64g"),
        max_workers=args.max_workers,
    )
    outcome = context.execute(
        Dataset.from_list(tasks).map(_compare_group),
        verbose=True,
        map_task_resources=ResourceConfig(cpu=1, ram="6g", disk="16g"),
    )

    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"baseline": 0, "candidate": 0, "both": 0, "baseline_only": 0, "candidate_only": 0}
    )
    for result in outcome.results:
        for row in result:
            entry = per_source[row["source_tag"]]
            for key in entry:
                entry[key] += row[key]

    totals = {key: sum(entry[key] for entry in per_source.values()) for key in next(iter(per_source.values()))}
    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "shards": len(entries),
        "elapsed_seconds": time.monotonic() - started,
        "totals": totals,
        "sources": dict(sorted(per_source.items())),
    }
    StoragePath(prefix_join(args.out, "compare.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Totals: %s", json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
