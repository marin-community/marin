# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repack the Focus Crawl's fuzzy candidates onto its current shard layout.

The pinned candidate artifact addresses the Focus Crawl by its pre-#8111
extraction, whose 4,573 shards were repacked into the 333 shards the normalized
tree carries today. Duplicate attributes are sparse, so a consumer that joins by
shard basename finds nothing and reads it as "no duplicates" rather than as an
error: the v10 cluster-text build skipped all 333 shards and 12.2 million
clustered documents silently.

:func:`repack_fuzzy_dups_source` re-shards the candidate rows by ``id`` with the
same deterministic rule normalization uses, so it never reads the text. It
returns a complete candidate artifact -- the legacy key replaced by the current
one, every other source untouched -- which drops straight into the
``--candidates`` of a cluster-text build.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority production --cpu 8 --memory 32GB \
        -- python experiments/datakit/scripts/repack_focus_candidates.py \
            --prefix s3://.../marin --candidates datakit/dedup_709f5997 \
            --out s3://.../user/rav/dedup/candidates_repacked/v1
"""

import argparse
import logging

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.repack_fuzzy_dups import repack_fuzzy_dups_source
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)

LEGACY_SOURCE_KEY = "data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9/outputs/main"
NORMALIZED_SOURCE = "normalized/common-crawl-focus-2026-22_17ce32f9"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True, help="Pinned fuzzy candidate artifact, relative to prefix")
    parser.add_argument("--normalized", default=NORMALIZED_SOURCE, help="Normalized source whose layout to match")
    parser.add_argument("--legacy-source-key", default=LEGACY_SOURCE_KEY)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=8)
    parser.add_argument("--worker-ram", default="64g")
    parser.add_argument("--worker-disk", default="128g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    candidates = read_artifact(prefix_join(args.prefix, args.candidates), FuzzyDupsAttrData)
    normalized = read_artifact(prefix_join(args.prefix, args.normalized), NormalizedData)
    logger.info(
        "Repacking %s onto the layout of %s (%d candidate sources)",
        args.legacy_source_key,
        args.normalized,
        len(candidates.sources),
    )

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    result = repack_fuzzy_dups_source(
        candidates=candidates,
        legacy_source_key=args.legacy_source_key,
        normalized=normalized,
        output_path=args.out,
        max_workers=args.max_workers,
        worker_resources=worker,
    )
    write_artifact(result, args.out)
    logger.info("Wrote repacked candidate artifact with %d sources to %s", len(result.sources), args.out)


if __name__ == "__main__":
    main()
