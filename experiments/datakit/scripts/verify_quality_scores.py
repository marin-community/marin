# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile a whole scoring run's written shards against its manifest.

``score_corpus.py verify`` answers this for one source. A repair run covers 150 of
them, so this drives that same check across every source in a manifest and
pools the result, rather than reimplementing the reconciliation.

The invariant that matters is per shard: the join is an inner join and every embedded
document has a ``chunk_index == 0`` token row, so ``score_rows == embed_rows`` exactly.
A shard scoring *fewer* rows than it embedded is the signature of a join that quietly
did not finish, which is how the first corpus run lost 36.3M documents. Zero-row embed
shards are legitimate -- a shard fully removed by global dedup embeds nothing -- and are
counted separately rather than treated as a shortfall.

Pooled mean and standard deviation are exact (combined from per-source moments); the
bucket shares are exact because the per-source counts are counts. Percentiles are not
poolable from summaries and are reported per source.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --cpu 8 --memory 32g --disk 32g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.verify_quality_scores --manifest s3://.../manifest_depth2
"""

import argparse
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    DEFAULT_CALIBRATION,
    DEFAULT_MANIFEST,
    read_manifest,
    verify_source,
)

logger = logging.getLogger(__name__)

# The 142-source corpus run, for comparison. These are the numbers a repair run's
# distribution is judged against; a trace/safety corpus may legitimately differ.
CORPUS_REFERENCE = {
    "score_mean": 0.5353,
    "score_std": 0.1654,
    "p1": 0.1494,
    "p50": 0.5378,
    "p99": 0.8913,
    "bucket_shares": [0.070, 0.251, 0.346, 0.232, 0.101],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--verify-threads", type=int, default=32)
    ap.add_argument("--source-threads", type=int, default=16)
    ap.add_argument("--report", default="", help="write the pooled report here as JSON")
    args = ap.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    manifest = read_manifest(args.manifest)
    sources = sorted(set(manifest.get_column("source").to_list()))
    logger.info("verifying %d sources from %s", len(sources), args.manifest)

    def one(source: str) -> dict:
        return verify_source(args.manifest, source, args.calibration, args.verify_threads)

    with ThreadPoolExecutor(max_workers=args.source_threads) as pool:
        reports = list(pool.map(one, sources))

    score_rows = sum(r["score_rows"] for r in reports)
    embed_rows = sum(r["embed_rows"] for r in reports)
    counts = np.sum([r["bucket_counts"] for r in reports], axis=0)
    # Exact pooled moments: E[x] and E[x^2] are additive over disjoint groups.
    weighted = [(r["score_rows"], r["score_mean"], r["score_std"]) for r in reports if r["score_rows"]]
    total = sum(n for n, _, _ in weighted)
    mean = sum(n * m for n, m, _ in weighted) / total if total else None
    second = sum(n * (s * s + m * m) for n, m, s in weighted) / total if total else None
    std = math.sqrt(max(second - mean * mean, 0.0)) if total else None

    mismatched = [r for r in reports if r["score_rows"] != r["embed_rows"]]
    report = {
        "manifest": args.manifest,
        "sources": len(reports),
        "score_rows": score_rows,
        "embed_rows": embed_rows,
        "shortfall": embed_rows - score_rows,
        "sources_with_row_mismatch": [
            {"source": r["source"], "score_rows": r["score_rows"], "embed_rows": r["embed_rows"]} for r in mismatched
        ],
        "normalize_docs": sum(r["normalize_docs"] or 0 for r in reports),
        "sources_not_covering_normalize": [
            {"source": r["source"], "score_rows": r["score_rows"], "normalize_docs": r["normalize_docs"]}
            for r in reports
            if not r["covers_normalize"]
        ],
        "duplicate_ids": sum(r["duplicate_ids"] for r in reports),
        "sources_with_duplicate_ids": [
            {"source": r["source"], "duplicate_ids": r["duplicate_ids"]} for r in reports if r["duplicate_ids"]
        ],
        "shards_below_embed_rows": sum(r["shards_below_embed_rows"] for r in reports),
        "missing_score_files": sum(r["missing_score_files"] for r in reports),
        "empty_score_files": sum(r["empty_score_files"] for r in reports),
        "manifest_shards": sum(r["manifest_shards"] for r in reports),
        "model_tags": sorted({tag for r in reports for tag in r["model_tags"]}),
        "pooled_score_mean": mean,
        "pooled_score_std": std,
        "bucket_edges": reports[0]["bucket_edges"] if reports else [],
        "bucket_counts": counts.tolist(),
        "bucket_shares": (counts / max(score_rows, 1)).tolist(),
        "corpus_reference": CORPUS_REFERENCE,
        "per_source": reports,
    }
    summary = {k: v for k, v in report.items() if k != "per_source"}
    logger.info("VERIFY %s", json.dumps(summary, indent=2))
    if args.report:
        fsspec.filesystem("s3").pipe_file(args.report.removeprefix("s3://"), json.dumps(report, indent=2).encode())
        logger.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
