# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare what two duplicate rules remove from each store bucket.

The store partitions on ``(domain cluster, quality)``, and mixture weights are
set per bucket, so a corpus-wide drop rate cannot say whether a rule change is
safe. A rule that takes 15% evenly is a different corpus from one that takes 15%
concentrated in the buckets the mixture leans on.

Buckets are computed the way the store computes them: the domain cluster from
the assignment artifact's ``cluster_<view>`` column, and the quality bucket by
cutting the score with the calibration for that document's content type. Every
tree is measured over the same shards in the same pass, so the difference
between rules is a difference in the rules, not in coverage.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority interactive --cpu 4 --memory 16GB \
        -- python experiments/datakit/scripts/dedup_drop_by_bucket.py \
            --markers new=s3://.../verified/v11-c075-restored \
                      old=datakit/verify_fuzzy_dups_c757e4f0 \
            --out s3://.../user/rav/dedup/reports/by_bucket.json
"""

import argparse
import collections
import json
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.verify_fuzzy_dups import VerifiedFuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit import hero_data
from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import calibration_edges_by_content_type
from experiments.datakit.reference_pipeline import SPLIT

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/drop_by_bucket"
DEFAULT_CALIBRATION_KEY = "default"
CLUSTER_VIEW = 40


def read_columns(path: str, columns: list[str]):
    with StoragePath(path).open("rb") as handle:
        return pq.read_table(handle, columns=columns)


def marker_ids(attr_dir: str, basename: str) -> set[str]:
    """Ids one tree marked in this shard; sparse, so absent means none."""
    if not attr_dir:
        return set()
    try:
        return set(read_columns(prefix_join(attr_dir, basename), ["id"]).column("id").to_pylist())
    except FileNotFoundError:
        return set()


def count_shard(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Documents and per-tree removals, keyed by (cluster, quality)."""
    edges = spec["edges"]
    documents: collections.Counter = collections.Counter()
    dropped = {name: collections.Counter() for name in spec["trees"]}
    read = 0

    for shard in spec["shards"]:
        basename = shard["basename"]
        try:
            quality = read_columns(prefix_join(shard["quality_dir"], basename), ["id", "score"])
            content = read_columns(prefix_join(shard["content_dir"], basename), ["id", "content_type"])
            cluster = read_columns(prefix_join(shard["cluster_dir"], basename), ["id", spec["cluster_column"]])
        except FileNotFoundError:
            continue
        ids = quality.column("id").to_pylist()
        if not (len(ids) == content.num_rows == cluster.num_rows):
            raise RuntimeError(f"{basename}: quality/content/cluster row mismatch -- co-partitioning broken")
        read += 1

        scores = np.asarray(quality.column("score").to_pylist(), dtype=np.float64)
        types = np.asarray(content.column("content_type").to_pylist(), dtype=object)
        buckets = np.empty(len(scores), dtype=np.int32)
        for kind in set(types.tolist()):
            selector = types == kind
            buckets[selector] = np.digitize(scores[selector], edges.get(kind, edges[DEFAULT_CALIBRATION_KEY]))
        clusters = cluster.column(spec["cluster_column"]).to_pylist()

        for position in range(len(ids)):
            documents[(int(clusters[position]), int(buckets[position]))] += 1
        for name, attr_dirs in spec["trees"].items():
            marked = marker_ids(attr_dirs.get(shard["source_key"], ""), basename)
            if not marked:
                continue
            for position, doc_id in enumerate(ids):
                if doc_id in marked:
                    dropped[name][(int(clusters[position]), int(buckets[position]))] += 1

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards", read)
    yield {
        "documents": {f"{c}:{q}": n for (c, q), n in documents.items()},
        "dropped": {name: {f"{c}:{q}": n for (c, q), n in counts.items()} for name, counts in dropped.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", nargs="+", required=True, help="NAME=ROOT per marker tree to compare")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-view", type=int, default=CLUSTER_VIEW)
    parser.add_argument("--shards-per-task", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=8)
    parser.add_argument("--worker-ram", default="48g")
    parser.add_argument("--task-cpu", type=float, default=4)
    parser.add_argument("--task-ram", default="8g")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prefix = hero_data.MANIFEST_PREFIX
    roots = dict(pair.split("=", 1) for pair in args.markers)
    # Each tree names its own attribute directory per source, so resolve through
    # the artifact rather than assuming a layout: the two rules were produced by
    # different code and key the focus crawl differently.
    trees: dict[str, dict[str, str]] = {}
    for name, root in roots.items():
        path = root if "://" in root else prefix_join(prefix, root)
        artifact = read_artifact(path, VerifiedFuzzyDupsAttrData)
        trees[name] = {key: entry.attr_dir for key, entry in artifact.sources.items()}
        logger.info("%s: %d sources, rule=%s", name, len(artifact.sources), artifact.rule or artifact.verification)

    edges = calibration_edges_by_content_type(
        hero_data.quality_calibration(hero_data.NEMOTRON_88K),
        expected_sha256=hero_data.NEMOTRON_88K.calibration_sha256,
    )

    shards: list[dict[str, str]] = []
    for name in hero_data.source_names():
        tokenize = read_artifact(hero_data.tokenized(name).output_path, TokenizedAttrData)
        source_key = tokenize.source_keys.get(SPLIT)
        tokenize_dir = tokenize.output_dirs.get(SPLIT)
        if source_key is None or tokenize_dir is None:
            continue
        quality_dir = hero_data.quality(name).output_path
        content_dir = hero_data.content_type(name).output_path
        cluster_dir = read_artifact(hero_data.assigned_clusters(name).output_path, AssignmentAttrData).output_dir
        for path in StoragePath(f"{tokenize_dir.rstrip('/')}/*.parquet").glob():
            shards.append(
                {
                    "basename": str(path).rsplit("/", 1)[-1],
                    "source_key": source_key,
                    "quality_dir": quality_dir,
                    "content_dir": content_dir,
                    "cluster_dir": cluster_dir,
                }
            )
    logger.info("Measuring %d shards across %d trees", len(shards), len(trees))

    batches = [
        {
            "shards": shards[start : start + args.shards_per_task],
            "trees": trees,
            "edges": edges,
            "cluster_column": f"cluster_{args.cluster_view}",
        }
        for start in range(0, len(shards), args.shards_per_task)
    ]
    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk="64g")
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="16g")
    context = ZephyrContext(name="dedup-drop-by-bucket", resources=worker, max_workers=args.max_workers)
    outcome = context.execute(Dataset.from_list(batches).flat_map(count_shard), verbose=True, map_task_resources=task)

    documents: collections.Counter = collections.Counter()
    dropped = {name: collections.Counter() for name in trees}
    for result in outcome.results:
        if not isinstance(result, dict):
            continue
        documents.update(result["documents"])
        for name, counts in result["dropped"].items():
            dropped[name].update(counts)

    by_quality: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_cluster: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for key, n in documents.items():
        c, q = key.split(":")
        by_quality["documents"][q] += n
        by_cluster["documents"][c] += n
    for name, counts in dropped.items():
        for key, n in counts.items():
            c, q = key.split(":")
            by_quality[name][q] += n
            by_cluster[name][c] += n

    payload = {
        "trees": {name: roots[name] for name in trees},
        "cluster_view": args.cluster_view,
        "documents": sum(documents.values()),
        "totals": {name: sum(counts.values()) for name, counts in dropped.items()},
        "by_quality": {k: dict(v) for k, v in by_quality.items()},
        "by_cluster": {k: dict(v) for k, v in by_cluster.items()},
        "by_bucket": {
            "documents": dict(documents),
            **{name: dict(counts) for name, counts in dropped.items()},
        },
    }
    StoragePath(args.out).write_bytes(json.dumps(payload, indent=1).encode())

    total = sum(documents.values())
    logger.info("documents %d", total)
    for name in trees:
        logger.info(
            "%s removes %d (%.2f%% of corpus)", name, payload["totals"][name], 100 * payload["totals"][name] / total
        )
    logger.info("--- by quality bucket ---")
    for q in sorted(by_quality["documents"], key=int):
        docs = by_quality["documents"][q]
        rates = "  ".join(f"{name}={100 * by_quality[name][q] / docs:5.2f}%" for name in trees)
        logger.info("q%s: %14d docs  %s", q, docs, rates)
    logger.info("--- by domain cluster (10 largest) ---")
    for c, docs in sorted(by_cluster["documents"].items(), key=lambda kv: -kv[1])[:10]:
        rates = "  ".join(f"{name}={100 * by_cluster[name][c] / docs:5.2f}%" for name in trees)
        logger.info("cluster %-4s %12d docs  %s", c, docs, rates)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
