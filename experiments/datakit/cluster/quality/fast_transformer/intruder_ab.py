# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge two quality scorers by the intruder test, holding domain fixed.

Builds a :class:`~experiments.datakit.cluster.intruder.BucketPool` per scorer from
its quality buckets over the same documents, and asks which scorer's buckets a panel
finds more coherent.

Buckets are named ``<group>|q<bucket>`` and the pool is stratified on ``<group>``, so
the in-group documents and the intruder always come from the *same* domain and differ
only in assigned quality. Without that, a panel can pick out the intruder by topic —
and since quality correlates with domain, a scorer that merely sorted documents by
domain would look maximally coherent. The stratified form asks the question the store
actually needs: inside one partition, does the quality axis separate documents?

Documents are shuffled before the pool sees them. ``BucketPool`` treats each bucket's
head as a uniform sample, which is only true if the caller shuffled — and the scored
shards arrive in corpus order, which correlates with everything.
"""

import argparse
import logging
import random

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.intruder import (
    DEFAULT_PANEL_SIZE,
    IN_GROUP_COUNT,
    Bucket,
    BucketPool,
    default_panel,
    openrouter_panel,
    run_intruder_test,
)

logger = logging.getLogger(__name__)

DEFAULT_GROUPING = "domain_id"
# Documents per (group, bucket). Four is the floor a trial needs; this leaves room
# for distinct trials without holding the whole evaluation set in memory.
DEFAULT_PER_BUCKET = 64
MIN_BUCKET_DOCS = IN_GROUP_COUNT  # a trial needs this many in-group documents
# Bands for domain-quantile bucketing: matches the deployed 5-bucket scale so the
# two bucketings are compared at the same granularity.
N_BUCKETS = 5
BUCKETINGS = ("global", "domain-quantile")


def _read_dir(path: str, columns: list[str]) -> dict[str, list]:
    out: dict[str, list] = {c: [] for c in columns}
    shards = sorted(str(m) for m in StoragePath(f"{path.rstrip('/')}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no parquet shards under {path}")
    for shard in shards:
        with StoragePath(shard).open("rb") as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        for c in columns:
            out[c].extend(table.column(c).to_pylist())
    return out


def _domain_quantile_buckets(scores: list[float], groups: list, n_buckets: int) -> list[int]:
    """Rank each document into ``n_buckets`` equal bands *within its own group*.

    The deployed alternative applies one global set of cutpoints to every document,
    which lets a whole domain land in one band — v0 puts 61.3% of maths and 0.0% of
    safety in its top two buckets. Ranking within a group makes each band hold the
    same share of every domain by construction, so no domain can be excluded.

    The cost is that a band no longer means an absolute quality level: the top band
    of a uniformly poor domain is still poor. Whether such bands are *coherent* —
    whether a panel can tell one from another — is what the intruder test measures,
    and it is not answered by the construction.
    """
    by_group: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        by_group.setdefault(str(group), []).append(i)
    buckets = [0] * len(scores)
    for members in by_group.values():
        order = sorted(members, key=lambda i: scores[i])
        for rank, i in enumerate(order):
            # Equal-count bands; the last band absorbs the rounding remainder.
            buckets[i] = min(n_buckets - 1, rank * n_buckets // len(order))
    return buckets


def build_pool(
    name: str,
    docs_dir: str,
    scored_root: str,
    *,
    grouping: str,
    per_bucket: int,
    seed: int,
    bucketing: str = "global",
    adjacent_only: bool = False,
    n_buckets: int = N_BUCKETS,
) -> BucketPool:
    """A scorer's ``(group, bucket)`` bucketing over the evaluation documents.

    ``bucketing="global"`` uses the scorer's own ``quality_bucket``, produced by the
    deployed global cutpoints. ``bucketing="domain-quantile"`` re-buckets by rank
    within each group. The second makes two scorers directly comparable — their
    bands hold identical population shares — which the first does not: a scorer
    whose top band holds 2.4% of documents is contrasted against a far-away
    remainder, while one holding 23% is contrasted against its near neighbours, and
    that difference alone moves intruder detection.
    """
    meta = _read_dir(docs_dir, ["id", "text", grouping])
    text_by_id = dict(zip(meta["id"], meta["text"], strict=True))
    group_by_id = dict(zip(meta["id"], meta[grouping], strict=True))

    scored = _read_dir(f"{scored_root.rstrip('/')}/outputs/main", ["id", "score", "quality_bucket"])
    keep = [i for i, doc_id in enumerate(scored["id"]) if doc_id in text_by_id]
    ids = [scored["id"][i] for i in keep]
    if bucketing == "domain-quantile":
        assigned = _domain_quantile_buckets([scored["score"][i] for i in keep], [group_by_id[d] for d in ids], n_buckets)
    else:
        assigned = [scored["quality_bucket"][i] for i in keep]

    members: dict[str, list[str]] = {}
    for doc_id, bucket in zip(ids, assigned, strict=True):
        members.setdefault(f"{group_by_id[doc_id]}|q{bucket}", []).append(text_by_id[doc_id])

    rng = random.Random(seed)
    buckets = []
    for key, docs in members.items():
        if len(docs) < MIN_BUCKET_DOCS:
            continue
        rng.shuffle(docs)  # BucketPool's head-as-uniform-sample contract
        buckets.append(Bucket(key, docs[:per_bucket]))
    logger.info("intruder_ab: %s -> %d (group, bucket) cells from %d scored docs", name, len(buckets), len(scored["id"]))

    if adjacent_only:
        return _adjacent_pool(name, buckets)
    return BucketPool(name, buckets, stratum_of=lambda key: key.rsplit("|q", 1)[0])


def _adjacent_pool(name: str, buckets: list[Bucket]) -> BucketPool:
    """A pool whose every trial pairs neighbouring quality buckets.

    Unconstrained, a trial draws its intruder from any other bucket in the group,
    and how hard that is depends on how far apart the two buckets are. That favours
    a lopsided bucketing: the deployed model holds 63.5% of each domain in one
    bucket and its extreme buckets are tiny (10th/90th percentile cell-size ratio
    0.009 against 0.088), so its trials often pit a large middle bucket against a
    handful of outliers, which are conspicuous for being outliers rather than for
    the bucket being coherent.

    Confining each trial to an adjacent pair asks both bucketings the same
    question — can the panel tell one quality level from the next — so neither
    profits from the shape of its own distribution.
    """
    by_group: dict[str, dict[int, Bucket]] = {}
    for bucket in buckets:
        group, level = bucket.name.rsplit("|q", 1)
        by_group.setdefault(group, {})[int(level)] = bucket

    paired: list[Bucket] = []
    for group, levels in by_group.items():
        for level in sorted(levels):
            if level + 1 not in levels:
                continue
            # One stratum per adjacent pair; a cell appears in up to two of them.
            for member in (level, level + 1):
                paired.append(Bucket(f"{group}|p{level}|q{member}", levels[member].docs))
    logger.info("intruder_ab: %s -> %d adjacent-pair cells", name, len(paired))
    return BucketPool(name, paired, stratum_of=lambda key: key.rsplit("|q", 1)[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", required=True, help="evaluation set (id, text, grouping column)")
    parser.add_argument("--lhs", required=True, metavar="NAME=PATH", help="first scorer's output root")
    parser.add_argument("--rhs", required=True, metavar="NAME=PATH", help="second scorer's output root")
    parser.add_argument("--grouping", default=DEFAULT_GROUPING, help="column held fixed within a trial")
    parser.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET)
    parser.add_argument(
        "--bucketing",
        choices=BUCKETINGS,
        default="global",
        help="global: the scorer's own cutpoints. domain-quantile: equal-count bands within each group.",
    )
    parser.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    parser.add_argument(
        "--panel-model",
        default=None,
        metavar="OPENROUTER_MODEL",
        help="judge via OpenRouter with this model instead of local headless claude "
        "(needs OR_INTRUDER_key; use when CLI seats would starve the operator's own usage)",
    )
    parser.add_argument("--target-trials", type=int, default=120)
    parser.add_argument("--max-trials", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--adjacent-only",
        action="store_true",
        help="confine every trial to neighbouring quality buckets, so a lopsided bucketing gains nothing",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    lhs_name, lhs_path = args.lhs.split("=", 1)
    rhs_name, rhs_path = args.rhs.split("=", 1)
    common = {
        "docs_dir": args.docs_dir,
        "grouping": args.grouping,
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "bucketing": args.bucketing,
        "adjacent_only": args.adjacent_only,
    }
    lhs = build_pool(lhs_name, scored_root=lhs_path, **common)
    rhs = build_pool(rhs_name, scored_root=rhs_path, **common)

    panel = openrouter_panel(args.panel_model, args.panel_size) if args.panel_model else default_panel(args.panel_size)
    result = run_intruder_test(
        lhs,
        rhs,
        panel=panel,
        target_trials=args.target_trials,
        max_trials=args.max_trials,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        seed=args.seed,
    )
    print(f"\ngrouping held fixed: {args.grouping}   bucketing: {args.bucketing}   chance: {result.chance_level}")
    print(f"decision: {result.decision}")
    print(f"  {result.lhs_name:14} {result.lhs_accuracy:.3f}  {np.round(result.lhs_interval, 3)}")
    print(f"  {result.rhs_name:14} {result.rhs_accuracy:.3f}  {np.round(result.rhs_interval, 3)}")
    print(f"  difference interval: {np.round(result.difference_interval, 3)}")
    by_side = ", ".join(f"{side} {n}" for side, n in sorted(result.abstained_by_side.items())) or "none"
    print(f"  trials/side: {result.n_trials_per_side}   abstained: {result.n_abstained} ({by_side})")
    # A lopsided split means one side was scored on an easier, self-selected
    # subset of its trials, which voids the comparison regardless of the intervals.
    counts = sorted(result.abstained_by_side.values())
    if len(counts) == 2 and counts[1] > 3 * max(counts[0], 1):
        print("  WARNING: abstentions are lopsided — treat this comparison as void, not as evidence")
    print("\nPanel seats are the same model, so this measures self-consistency, not")
    print("agreement across independent judges.")


if __name__ == "__main__":
    main()
