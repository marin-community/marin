# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure what fuzzy dedup removes from each quality bucket.

A corpus-wide drop rate says how much the rule takes. It does not say what it
takes. If removals concentrate in the top bucket the rule is eating the data the
store exists to keep, and if they concentrate in the bottom bucket it is doing
the filtering stage's work for it. Either way the composition shifts, and the
mixing weights downstream are set per bucket.

Buckets are computed exactly as the store computes them: the same calibration,
cut per content type, so a score means the same thing here as it does there.
Sampled by shard, and each shard is read once for every threshold, so the
comparison between thresholds is over identical documents.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority interactive --cpu 8 --memory 64GB \
        -- python experiments/datakit/scripts/dedup_drop_by_quality.py \
            --markers 0.60=s3://.../verified/v11-c060 0.75=s3://.../verified/v11-c075 \
            --shards 400 --out s3://.../user/rav/dedup/reports/by_quality.json
"""

import argparse
import collections
import concurrent.futures
import json
import logging
import random

import numpy as np
import pyarrow.parquet as pq
from marin.execution.artifact import read_artifact
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit import hero_data
from experiments.datakit.cluster.quality.fast_transformer.artifact import calibration_edges_by_content_type
from experiments.datakit.reference_pipeline import SPLIT

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_KEY = "default"
READ_CONCURRENCY = 32


def read_columns(path: str, columns: list[str]):
    with StoragePath(path).open("rb") as handle:
        return pq.read_table(handle, columns=columns)


def marker_ids(root: str, source_tag: str, basename: str) -> set[str]:
    """Ids this run marked in one shard; a missing file means no duplicates."""
    path = prefix_join(root, f"outputs/{source_tag}/{basename}")
    try:
        return set(read_columns(path, ["id"]).column("id").to_pylist())
    except FileNotFoundError:
        return set()


def shard_counts(spec: dict, roots: dict[str, str], edges: dict) -> dict | None:
    """Documents and removals per quality bucket for one shard."""
    try:
        quality = read_columns(prefix_join(spec["quality_dir"], spec["basename"]), ["id", "score"])
        content = read_columns(prefix_join(spec["content_dir"], spec["basename"]), ["id", "content_type"])
    except FileNotFoundError:
        return None
    ids = quality.column("id").to_pylist()
    if len(ids) != content.num_rows:
        raise RuntimeError(f"{spec['basename']}: quality/content-type row mismatch -- co-partitioning broken")

    scores = np.asarray(quality.column("score").to_pylist(), dtype=np.float64)
    types = np.asarray(content.column("content_type").to_pylist(), dtype=object)
    buckets = np.empty(len(scores), dtype=np.int32)
    for kind in set(types.tolist()):
        selector = types == kind
        buckets[selector] = np.digitize(scores[selector], edges.get(kind, edges[DEFAULT_CALIBRATION_KEY]))

    documents = collections.Counter(buckets.tolist())
    dropped = {name: collections.Counter() for name in roots}
    for name, root in roots.items():
        marked = marker_ids(root, spec["source_tag"], spec["basename"])
        if not marked:
            continue
        for position, doc_id in enumerate(ids):
            if doc_id in marked:
                dropped[name][int(buckets[position])] += 1
    return {"documents": documents, "dropped": dropped, "source": spec["source_name"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", nargs="+", required=True, help="NAME=ROOT for each marker tree to measure")
    parser.add_argument("--cluster-text", required=True, help="Root holding the manifest that names every shard")
    parser.add_argument("--shards", type=int, default=400, help="Shards sampled across the corpus")
    parser.add_argument(
        "--restrict-to",
        help=(
            "JSON list of tag/basename shards to sample from. Required when a marker tree is "
            "incomplete: a missing file reads as no duplicates, so an unfinished run scores as a "
            "clean one and the comparison measures coverage instead of the rule."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()

    roots = dict(pair.split("=", 1) for pair in args.markers)
    edges = calibration_edges_by_content_type(
        hero_data.quality_calibration(hero_data.NEMOTRON_88K),
        expected_sha256=hero_data.NEMOTRON_88K.calibration_sha256,
    )
    logger.info("bucket edges per content type: %s", {k: [round(e, 3) for e in v] for k, v in edges.items()})

    manifest = json.loads(StoragePath(prefix_join(args.cluster_text, "manifest.json")).read_bytes())

    # The manifest names shards by normalized source key; the quality and
    # content-type trees are addressed per source name. Only tokenize carries
    # the mapping, so read it once per source.
    def resolve(name: str) -> tuple[str, str, str, str] | None:
        tokenize = read_artifact(hero_data.tokenized(name).output_path, TokenizedAttrData)
        key = tokenize.source_keys.get(SPLIT)
        if key is None:
            return None
        return key, name, hero_data.quality(name).output_path, hero_data.content_type(name).output_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        resolved = [r for r in pool.map(resolve, hero_data.source_names()) if r]
    by_key = {key: (name, quality_dir, content_dir) for key, name, quality_dir, content_dir in resolved}
    logger.info("Resolved %d of %d sources", len(by_key), len(hero_data.source_names()))

    specs = []
    for shard in manifest["shards"]:
        entry = by_key.get(shard["source_key"])
        if entry is None:
            continue
        name, quality_dir, content_dir = entry
        specs.append(
            {
                "source_name": name,
                "source_tag": shard["source_tag"],
                "basename": shard["basename"],
                "quality_dir": quality_dir,
                "content_dir": content_dir,
            }
        )

    if args.restrict_to:
        allowed = set(json.loads(StoragePath(args.restrict_to).read_bytes()))
        specs = [s for s in specs if f"{s['source_tag']}/{s['basename']}" in allowed]
        logger.info("Restricted to %d shards every marker tree covers", len(specs))
        if not specs:
            raise ValueError("no sampled shard is covered by every marker tree")

    rng = random.Random(args.seed)
    sample = rng.sample(specs, min(args.shards, len(specs)))
    logger.info("Measuring %d of %d shards over %d marker trees", len(sample), len(specs), len(roots))

    documents: collections.Counter = collections.Counter()
    dropped = {name: collections.Counter() for name in roots}
    per_source: dict[str, dict] = collections.defaultdict(lambda: {"documents": 0, **{n: 0 for n in roots}})
    measured = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        for outcome in pool.map(lambda spec: shard_counts(spec, roots, edges), sample):
            if outcome is None:
                continue
            measured += 1
            documents.update(outcome["documents"])
            per_source[outcome["source"]]["documents"] += sum(outcome["documents"].values())
            for name in roots:
                dropped[name].update(outcome["dropped"][name])
                per_source[outcome["source"]][name] += sum(outcome["dropped"][name].values())

    total = sum(documents.values())
    payload = {
        "shards_measured": measured,
        "shards_sampled": len(sample),
        "documents": total,
        "buckets": sorted(documents),
        "by_bucket": {
            str(bucket): {
                "documents": documents[bucket],
                **{name: dropped[name][bucket] for name in roots},
            }
            for bucket in sorted(documents)
        },
        "by_source": {k: v for k, v in sorted(per_source.items())},
    }
    StoragePath(args.out).write_bytes(json.dumps(payload, indent=1).encode())

    logger.info("%d shards, %d documents", measured, total)
    for bucket in sorted(documents):
        share = 100 * documents[bucket] / total if total else 0
        rates = "  ".join(f"{name}={100 * dropped[name][bucket] / documents[bucket]:5.2f}%" for name in roots)
        logger.info("q%d: %12d docs (%5.2f%% of corpus)  %s", bucket, documents[bucket], share, rates)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
