# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the real fuzzy verifier over a bounded slice of production candidates.

Verification cannot be sliced by source: most clusters span several sources, and
a source slice loses their canonical members. It can be sliced by cluster. This
driver keeps every source registered and narrows the candidate tree instead, so
each selected cluster stays complete while the untouched shards drop out. The
verifier skips a shard whose candidate file is absent, so the run pays for the
selected shards only.

It measures what a full run needs to be sized: CPU seconds per candidate member,
peak worker memory with the document store resident, and the accepted fraction.

Run it in region::

    uv run iris --cluster=marin job run --no-wait --target-cluster cw-us-east-02a \
        --priority interactive --cpu 4 --memory 16g \
        -- python experiments/datakit/scripts/verify_fuzzy_dups_smoke.py \
            --workload s3://.../workload_m16384.json \
            --candidates s3://.../dedup_709f5997 --candidate-attrs-subdir outputs_it20 \
            --scratch s3://.../dedup-quality/smoke_candidates \
            --output s3://.../dedup-quality/smoke_verified
"""

import argparse
import collections
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    FuzzyVerificationStoreConfig,
    verify_fuzzy_dups,
)
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import DEFAULT_SCALE, select_sources, zephyr_datakit_steps
from experiments.datakit.reports.dedup import COUNTER_PREFIX, dedup_report

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = ["id", "dup_cluster_id", "is_cluster_canonical"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--candidates", required=True, help="Fuzzy-dedup output root")
    parser.add_argument("--candidate-attrs-subdir", default="outputs_it20")
    parser.add_argument("--scratch", required=True, help="Where the narrowed candidate tree is written")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-sources",
        default="",
        help=(
            "Comma-separated source names to leave out of verification. Their documents keep no verified "
            "marker, so the store retains them, and clusters reaching into them verify against what remains."
        ),
    )
    parser.add_argument("--max-clusters", type=int, default=20_000)
    parser.add_argument("--max-shards", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--worker-cpu", type=float, default=8.0)
    parser.add_argument("--worker-ram", default="64g")
    parser.add_argument("--worker-disk", default="64g")
    parser.add_argument("--task-cpu", type=float, default=2.0)
    parser.add_argument("--task-ram", default="16g")
    return parser.parse_args(argv)


def _select_clusters(workload: dict, max_clusters: int, max_shards: int) -> dict[str, list[dict]]:
    """Pick complete clusters that fit inside a shard budget."""
    usable = {c: m for c, m in workload["clusters"].items() if len(m) >= 2}
    by_footprint = sorted(usable, key=lambda c: (len({(x["source_tag"], x["shard"]) for x in usable[c]}), c))
    chosen: dict[str, list[dict]] = {}
    shards: set[tuple[str, str]] = set()
    for cluster in by_footprint:
        needed = {(m["source_tag"], m["shard"]) for m in usable[cluster]}
        if len(shards | needed) > max_shards:
            continue
        shards |= needed
        chosen[cluster] = usable[cluster]
        if len(chosen) >= max_clusters:
            break
    for cluster in by_footprint:
        if len(chosen) >= max_clusters:
            break
        if cluster not in chosen and {(m["source_tag"], m["shard"]) for m in usable[cluster]} <= shards:
            chosen[cluster] = usable[cluster]
    return chosen


def _write_narrowed_tree(
    chosen: dict[str, list[dict]], source_attrs: str, scratch: str, workers: int
) -> tuple[int, int]:
    """Copy only the rows of the selected clusters, keeping the shard layout.

    Returns the row and canonical counts of the slice, which the report needs:
    the candidate artifact's own counters describe the whole corpus, so using
    them here would divide this slice's verified duplicates by 5.95B members.
    """
    wanted: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for members in chosen.values():
        for member in members:
            wanted[(member["source_tag"], member["shard"])].add(member["id"])

    fs, root = url_to_fs(source_attrs.rstrip("/"))
    _, scratch_root = url_to_fs(scratch.rstrip("/"))

    def narrow(item: tuple[tuple[str, str], set[str]]) -> tuple[int, int]:
        (tag, shard), ids = item
        with fs.open(f"{root}/{tag}/{shard}", "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=CANDIDATE_COLUMNS)
        mask = pa.array([doc_id in ids for doc_id in table.column("id").to_pylist()])
        kept = table.filter(mask)
        if kept.num_rows == 0:
            return 0, 0
        target = f"{scratch_root}/{tag}/{shard}"
        fs.makedirs(f"{scratch_root}/{tag}", exist_ok=True)
        with fs.open(target, "wb") as handle:
            pq.write_table(kept, handle)
        return kept.num_rows, sum(kept.column("is_cluster_canonical").to_pylist())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(narrow, wanted.items()))
    rows = sum(row_count for row_count, _ in counts)
    canonicals = sum(canonical_count for _, canonical_count in counts)
    logger.info("Narrowed tree: %d rows (%d canonical) across %d shards", rows, canonicals, len(wanted))
    return rows, canonicals


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)

    workload = json.loads(StoragePath(args.workload).read_bytes())
    chosen = _select_clusters(workload, args.max_clusters, args.max_shards)
    members = sum(len(m) for m in chosen.values())
    logger.info("Selected %d complete clusters holding %d members", len(chosen), members)

    source_attrs = f"{args.candidates.rstrip('/')}/{args.candidate_attrs_subdir}"
    rows, canonicals = _write_narrowed_tree(chosen, source_attrs, args.scratch, workers=32)
    if rows == 0:
        raise RuntimeError("narrowed candidate tree is empty")
    # The trees are co-partitioned, so verification can keep only the shard
    # triples the selected clusters occupy. Selection guarantees each cluster
    # is complete inside them.
    shard_basenames = frozenset(member["shard"] for members in chosen.values() for member in members)
    logger.info("Verifying %d shard triple(s) instead of the whole corpus", len(shard_basenames))

    excluded = {name.strip() for name in args.exclude_sources.split(",") if name.strip()}
    sources = select_sources(None)
    unknown = excluded - set(sources)
    if unknown:
        raise KeyError(f"unknown sources to exclude: {sorted(unknown)}")
    steps = zephyr_datakit_steps(sources, DEFAULT_SCALE)
    sources = {name: step for name, step in sources.items() if name not in excluded}
    normalized_sources = {name: read_artifact(step.output_path, NormalizedData) for name, step in sources.items()}
    minhash_sources = {name: read_artifact(steps.minhash[name].output_path, MinHashAttrData) for name in sources}
    if excluded:
        logger.info("Excluding %d source(s) from verification: %s", len(excluded), sorted(excluded))

    original = read_artifact(args.candidates, FuzzyDupsAttrData)
    kept_keys = {datakit_source_key(str(data.main_output_dir)) for data in normalized_sources.values()}
    narrowed = {}
    excluded_tags = set()
    for source_key, entry in original.sources.items():
        tag = str(entry.attr_dir).rstrip("/").rsplit("/", 1)[-1]
        if source_key not in kept_keys:
            excluded_tags.add(tag)
            continue
        narrowed[source_key] = FuzzyDupsPerSource(attr_dir=f"{args.scratch.rstrip('/')}/{tag}")
    candidates = original.model_copy(update={"sources": narrowed})
    if excluded_tags:
        touched = sum(1 for members in chosen.values() if any(m["source_tag"] in excluded_tags for m in members))
        logger.info(
            "Excluded source tags %s appear in %d of %d selected clusters",
            sorted(excluded_tags),
            touched,
            len(chosen),
        )

    missing = {datakit_source_key(str(d.main_output_dir)) for d in normalized_sources.values()} - set(candidates.sources)
    if missing:
        raise ValueError(f"candidate tree lacks {len(missing)} source(s), first: {sorted(missing)[:3]}")

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.worker_disk)
    started = time.monotonic()
    verified = verify_fuzzy_dups(
        normalized_sources=normalized_sources,
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=args.output,
        verification_params=FuzzyVerificationParams(),
        local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
        store_config=FuzzyVerificationStoreConfig(
            recovery_timeout=1_800,
            ready_timeout=1_800,
            lookup_batch_size=128,
            shards_per_worker=1,
        ),
        max_output_shards=args.max_workers,
        max_workers=args.max_workers,
        worker_resources=worker,
        coordinator_resources=ResourceConfig(cpu=1, ram="4g"),
        map_task_resources=task,
        reduce_task_resources=task,
        shard_basenames=shard_basenames,
    )
    elapsed = time.monotonic() - started

    counters = dict(sorted(verified.counters.items()))
    cpu_time = counters.get("zephyr/worker/cpu_time", 0.0)
    candidate_members = counters.get("dedup/fuzzy/verification/candidate_members", 0)
    logger.info("SMOKE elapsed=%.1fs members=%s", elapsed, candidate_members)
    if candidate_members:
        logger.info(
            "SMOKE cost: %.2f CPU-ms per member; %.1f text chars per member",
            1000.0 * cpu_time / candidate_members,
            counters.get("dedup/fuzzy/verification/candidate_text_chars", 0) / candidate_members,
        )
    logger.info("SMOKE peak worker RSS: %.2f GiB", counters.get("zephyr/worker/mem_peak_bytes", 0) / 1024**3)
    logger.info("SMOKE counters:\n%s", json.dumps(counters, indent=2))
    StoragePath(f"{args.output.rstrip('/')}/smoke_counters.json").write_bytes(
        json.dumps({"elapsed_seconds": elapsed, "clusters": len(chosen), "counters": counters}, indent=2).encode()
    )

    # Scope the candidate counters to this slice so the report's ratios describe
    # what actually ran rather than the whole corpus.
    sliced = candidates.model_copy(
        update={
            "counters": {
                f"{COUNTER_PREFIX}/cluster_members": rows,
                f"{COUNTER_PREFIX}/canonicals": canonicals,
                f"{COUNTER_PREFIX}/singletons_skipped": 0,
            }
        }
    )
    report = dedup_report(f"{args.output.rstrip('/')}/report", candidates=sliced, verified=verified)
    logger.info("SMOKE report: %s", report.html_path)
    logger.info("SMOKE report stats:\n%s", json.dumps(report.stats, indent=2, default=str))


if __name__ == "__main__":
    main()
