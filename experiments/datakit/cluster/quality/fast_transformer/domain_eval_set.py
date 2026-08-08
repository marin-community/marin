# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw a domain-stratified evaluation set from a clustered sample.

A quality scorer never has to be coherent across the whole corpus: the store
partitions on ``cluster=<C>/quality=<Q>``, so quality is only ever read *inside* a
domain. Judging it therefore needs documents grouped by domain, which a clustered
sample supplies directly (``domain_id`` plus the finer ``cluster_5000``).

Equal quota per domain, not proportional to domain size. A proportional draw would
hand most of the evaluation to whichever domains are large, and the failure being
measured is precisely that some domains are treated worse than others — each needs
enough documents to estimate its own bucket mix.

*Within* a domain the draw is proportional to where that domain's documents actually
live. Counting first makes that exact: a census of ``domain_id`` over every shard
gives each shard's share of each domain, and the quota is split along those shares.
A greedy alternative — walk shards and take rows until quotas fill — is what this
replaced, and it drew 86% of an 80k set from a single source while reporting every
domain at full quota, because the first shards visited answered for everything.

Three passes, all cheap except the last: census ``domain_id``, allocate quota, then
read text only for the chosen rows. :func:`composition` reports what was actually
drawn, so a skewed draw is visible without a separate audit.
"""

import argparse
import collections
import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_PER_DOMAIN = 2_000
DOMAIN_COLUMN = "domain_id"
# Only the columns every shard carries. A clustered sample keeps each source's own
# extra fields, so schemas differ shard to shard (``teacher``, ``author``,
# ``agency``, ``channel_id``, ...) and even ``source`` is absent from some. The
# source label comes from the shard's directory instead, which the tree's layout
# guarantees.
CARRIED_COLUMNS = ("id", "text", "domain_id", "cluster_5000")
SHARD_ROWS = 5_000


def _shard_paths(root: str) -> list[str]:
    return sorted(str(m) for m in StoragePath(f"{root.rstrip('/')}/**/*.parquet").glob())


def _census(shards: list[str]) -> dict[str, collections.Counter]:
    """Rows per domain in every shard. Reads one int column, nothing else."""
    counts: dict[str, collections.Counter] = {}
    for i, shard in enumerate(shards, 1):
        with StoragePath(shard).open("rb") as handle:
            domains = pq.ParquetFile(handle).read(columns=[DOMAIN_COLUMN]).column(DOMAIN_COLUMN).to_pylist()
        counts[shard] = collections.Counter(domains)
        if i % 100 == 0:
            logger.info("domain_eval_set: census %d/%d shards", i, len(shards))
    return counts


def _allocate(census: dict[str, collections.Counter], per_domain: int) -> dict[str, dict[int, int]]:
    """Split each domain's quota across shards in proportion to where it lives.

    Largest-remainder rounding, so the per-domain allocations sum to exactly the
    quota rather than drifting with the number of shards.
    """
    totals: collections.Counter = collections.Counter()
    for counter in census.values():
        totals.update(counter)

    plan: dict[str, dict[int, int]] = {shard: {} for shard in census}
    for domain, total in totals.items():
        quota = min(per_domain, total)
        holders = [(shard, c[domain]) for shard, c in census.items() if c[domain]]
        exact = [(shard, quota * n / total) for shard, n in holders]
        floors = {shard: int(v) for shard, v in exact}
        remainder = quota - sum(floors.values())
        for shard, _ in sorted(exact, key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remainder]:
            floors[shard] += 1
        for shard, n in floors.items():
            capped = min(n, dict(holders)[shard])
            if capped:
                plan[shard][domain] = capped
        if quota < per_domain:
            logger.warning("domain_eval_set: domain %s holds only %d rows (quota %d)", domain, total, per_domain)
    logger.info(
        "domain_eval_set: %d domains, %d rows allocated", len(totals), sum(sum(d.values()) for d in plan.values())
    )
    return plan


def composition(drawn: list[tuple]) -> dict:
    """What the draw actually contains, from ``(domain, source)`` pairs.

    Takes pairs rather than whole records on purpose: holding 80k documents' text
    just to summarize their provenance is about a gigabyte, and it is the reason an
    earlier version of this was killed by the OOM reaper after the draw had already
    finished.
    """
    per_domain = collections.Counter(domain for domain, _ in drawn)
    sources = collections.Counter(source for _, source in drawn)
    by_domain_sources = collections.defaultdict(set)
    for domain, source in drawn:
        by_domain_sources[domain].add(source)
    counts = [len(v) for v in by_domain_sources.values()]
    return {
        "rows": len(drawn),
        "domains": len(per_domain),
        "docs_per_domain_min": min(per_domain.values()) if per_domain else 0,
        "docs_per_domain_max": max(per_domain.values()) if per_domain else 0,
        "distinct_sources": len(sources),
        "largest_source_share": (max(sources.values()) / len(drawn)) if drawn else 0.0,
        "sources_per_domain_min": min(counts) if counts else 0,
        "sources_per_domain_median": int(np.median(counts)) if counts else 0,
        "single_source_domains": sum(1 for v in counts if v == 1),
    }


def build_eval_set(*, sample_root: str, out_dir: str, per_domain: int = DEFAULT_PER_DOMAIN, seed: int = 42) -> dict:
    """Write the domain-stratified evaluation shards; returns the composition report."""
    shards = _shard_paths(sample_root)
    if not shards:
        raise ValueError(f"no parquet shards under {sample_root}")
    logger.info("domain_eval_set: %d shards under %s", len(shards), sample_root)

    plan = _allocate(_census(shards), per_domain)
    rng = np.random.default_rng(seed)
    root_prefix = sample_root.rstrip("/") + "/"
    out_root = out_dir.rstrip("/")

    drawn: list[tuple] = []  # (domain, source) only — see composition()
    buffer: list[dict] = []
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        with StoragePath(f"{out_root}/part-{shard_index:05d}.parquet").open("wb") as handle:
            pq.write_table(pa.Table.from_pylist(buffer), handle)
        shard_index += 1
        buffer = []

    for shard, wanted in plan.items():
        if not wanted:
            continue
        with StoragePath(shard).open("rb") as handle:
            table = pq.ParquetFile(handle).read(columns=list(CARRIED_COLUMNS))
        columns = table.to_pydict()
        source = shard[len(root_prefix) :].rsplit("/", 1)[0] if shard.startswith(root_prefix) else ""
        by_domain: dict[int, list[int]] = collections.defaultdict(list)
        for row, domain in enumerate(columns[DOMAIN_COLUMN]):
            by_domain[domain].append(row)
        for domain, take in wanted.items():
            candidates = by_domain[domain]
            chosen = rng.choice(len(candidates), size=min(take, len(candidates)), replace=False)
            for pick in chosen:
                record = {name: columns[name][candidates[pick]] for name in CARRIED_COLUMNS}
                record["source"] = source
                drawn.append((record[DOMAIN_COLUMN], source))
                buffer.append(record)
                if len(buffer) >= SHARD_ROWS:
                    flush()
    flush()

    report = composition(drawn)
    logger.info("domain_eval_set: wrote %d rows to %s", report["rows"], out_root)
    logger.info("domain_eval_set: composition %s", json.dumps(report, default=str))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True, help="clustered sample tree (parquet with domain_id)")
    parser.add_argument("--out-dir", required=True, help="directory the scorer-ready shards are written to")
    parser.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN, help="documents per domain")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    build_eval_set(sample_root=args.sample_root, out_dir=args.out_dir, per_domain=args.per_domain, seed=args.seed)


if __name__ == "__main__":
    main()
