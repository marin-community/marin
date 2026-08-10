# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the shape of the fuzzy-duplicate clusters that a dedup run produced.

The attribute tree records one row for each member of a non-singleton cluster.
Cluster identity is the connected-components label, which is a uniformly
distributed hash, so ``dup_cluster_id % modulus == residue`` selects a uniform
random sample of clusters and keeps every member of each selected cluster. That
gives exact sizes for the sampled clusters from one pass over the tree.

The scan reports the size distribution, how far each cluster spreads across
sources, and the clusters that hold no canonical member. It reads three columns
and no document text.

Run it in region, on CPU nodes::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \
        --priority interactive --cpu 60 --memory 200g \
        -- python experiments/datakit/scripts/analyze_dedup_clusters.py \
            --attrs s3://marin-us-east-02a/marin/datakit/dedup_709f5997/outputs_it20 \
            --out s3://marin-us-east-02a/marin/user/rav/dedup-quality/it20_clusters.json
"""

import argparse
import collections
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

COLUMNS = ["id", "dup_cluster_id", "is_cluster_canonical"]
DEFAULT_MODULUS = 1024
DEFAULT_WORKERS = 48
SIZE_BINS = [2, 3, 4, 5, 10, 20, 50, 100, 1000, 10_000, 100_000]
EXAMPLE_CLUSTERS = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attrs", required=True, help="Attribute tree root holding source_NNN directories")
    parser.add_argument("--out", required=True, help="Path for the JSON summary")
    parser.add_argument("--modulus", type=int, default=DEFAULT_MODULUS, help="Sample one cluster in this many")
    parser.add_argument("--residue", type=int, default=0, help="Cluster-id residue to keep")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent file readers")
    args = parser.parse_args(argv)
    if args.modulus < 1:
        parser.error("--modulus must be at least 1")
    if not 0 <= args.residue < args.modulus:
        parser.error("--residue must be inside the modulus")
    return args


def _bin_label(size: int) -> str:
    previous = None
    for edge in SIZE_BINS:
        if size < edge:
            return f"{previous}-{edge - 1}" if previous else f"<{edge}"
        previous = edge
    return f">={SIZE_BINS[-1]}"


def scan_file(fs, path: str, modulus: int, residue: int) -> tuple[int, int, list[tuple[str, str, bool, str]]]:
    """Return (rows, canonicals, sampled member tuples) for one attribute shard."""
    source_tag = path.rsplit("/", 2)[-2]
    with fs.open(path, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=COLUMNS)
    rows = table.num_rows
    if rows == 0:
        return 0, 0, []
    ids = table.column("id").to_pylist()
    clusters = table.column("dup_cluster_id").to_pylist()
    canonical = table.column("is_cluster_canonical").to_pylist()
    canonicals = sum(1 for flag in canonical if flag)
    sampled = [
        (clusters[i], ids[i], canonical[i], source_tag)
        for i in range(rows)
        # Cluster ids are decimal 128-bit hashes, so the residue is a uniform sample.
        if int(clusters[i]) % modulus == residue
    ]
    return rows, canonicals, sampled


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)

    fs, root = url_to_fs(args.attrs.rstrip("/"))
    paths = sorted(p for p in fs.find(root) if p.endswith(".parquet"))
    logger.info("Scanning %d attribute shards under %s", len(paths), root)

    total_rows = 0
    total_canonicals = 0
    members: dict[str, list[tuple[str, bool, str]]] = collections.defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_file, fs, path, args.modulus, args.residue) for path in paths]
        for done, future in enumerate(futures, start=1):
            rows, canonicals, sampled = future.result()
            total_rows += rows
            total_canonicals += canonicals
            for cluster, doc_id, is_canonical, source_tag in sampled:
                members[cluster].append((doc_id, is_canonical, source_tag))
            if done % 5000 == 0 or done == len(paths):
                logger.info("Read %d/%d shards; %d sampled clusters so far", done, len(paths), len(members))

    sizes = [len(m) for m in members.values()]
    sources_per_cluster = [len({tag for _, _, tag in m}) for m in members.values()]
    canonicals_per_cluster = [sum(1 for _, flag, _ in m if flag) for m in members.values()]
    size_histogram = collections.Counter(_bin_label(size) for size in sizes)
    spread_histogram = collections.Counter(sources_per_cluster)
    canonical_histogram = collections.Counter(canonicals_per_cluster)

    largest = sorted(members.items(), key=lambda kv: -len(kv[1]))[:EXAMPLE_CLUSTERS]
    summary = {
        "attrs_root": root,
        "shards": len(paths),
        "cluster_member_rows_total": total_rows,
        "canonical_rows_total": total_canonicals,
        "noncanonical_rows_total": total_rows - total_canonicals,
        "sample": {"modulus": args.modulus, "residue": args.residue, "clusters": len(members), "members": sum(sizes)},
        "cluster_size": {
            "mean": statistics.mean(sizes) if sizes else 0,
            "median": statistics.median(sizes) if sizes else 0,
            "p90": statistics.quantiles(sizes, n=10)[-1] if len(sizes) > 10 else None,
            "p99": statistics.quantiles(sizes, n=100)[-1] if len(sizes) > 100 else None,
            "max": max(sizes) if sizes else 0,
            "histogram": dict(sorted(size_histogram.items(), key=lambda kv: len(kv[0]))),
        },
        "sources_per_cluster": {str(k): v for k, v in sorted(spread_histogram.items())},
        "canonicals_per_cluster": {str(k): v for k, v in sorted(canonical_histogram.items())},
        "clusters_without_canonical": canonical_histogram.get(0, 0),
        "largest_clusters": [
            {
                "cluster_id": cluster,
                "size": len(m),
                "sources": sorted({tag for _, _, tag in m}),
                "canonicals": sum(1 for _, flag, _ in m if flag),
                "member_ids": [doc for doc, _, _ in m[:10]],
            }
            for cluster, m in largest
        ],
    }

    StoragePath(args.out).write_bytes(json.dumps(summary, indent=2).encode())
    logger.info("Wrote summary to %s", args.out)
    logger.info(
        "Rows %d (canonical %d). Sampled %d clusters, %d members; mean size %.2f, max %d, no-canonical clusters %d",
        total_rows,
        total_canonicals,
        len(members),
        sum(sizes),
        summary["cluster_size"]["mean"],
        summary["cluster_size"]["max"],
        summary["clusters_without_canonical"],
    )


if __name__ == "__main__":
    main()
