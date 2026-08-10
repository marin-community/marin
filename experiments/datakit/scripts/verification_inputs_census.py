# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Size the inputs a full fuzzy-verification run must read, and the blast radius
of clusters that hold no canonical member.

Verification joins normalized text, MinHash buckets, and candidate attributes,
so the MinHash tree is a third read alongside the corpus. The verifier also
rejects a cluster whose first record is not canonical, thus the count of such
clusters and the documents inside them decides whether that path needs a
fallback before a full run.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import DEFAULT_SCALE, select_sources, zephyr_datakit_steps

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, help="Sampled cluster workload JSON")
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)

    workload = json.loads(StoragePath(args.workload).read_bytes())
    clusters = workload["clusters"]
    without = {c: m for c, m in clusters.items() if not any(x["canonical"] for x in m)}
    members_total = sum(len(m) for m in clusters.values())
    members_without = sum(len(m) for m in without.values())
    logger.info(
        "CANONICAL-LESS: %d of %d sampled clusters (%.4f%%); %d of %d sampled members (%.4f%%); largest %d",
        len(without),
        len(clusters),
        100.0 * len(without) / max(1, len(clusters)),
        members_without,
        members_total,
        100.0 * members_without / max(1, members_total),
        max((len(m) for m in without.values()), default=0),
    )

    sources = select_sources(None)
    steps = zephyr_datakit_steps(sources, DEFAULT_SCALE)

    def measure(name: str) -> int:
        try:
            attrs = read_artifact(steps.minhash[name].output_path, MinHashAttrData)
            fs, root = url_to_fs(str(attrs.attr_dir).rstrip("/"))
            return sum(entry.get("size") or 0 for entry in fs.find(root, detail=True).values())
        except Exception as e:
            logger.warning("Cannot measure MinHash for %s: %s", name, e)
            return 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sizes = list(pool.map(measure, sorted(steps.minhash)))
    logger.info("MINHASH TREE: %.2f TiB across %d sources", sum(sizes) / 1024**4, len(sizes))


if __name__ == "__main__":
    main()
