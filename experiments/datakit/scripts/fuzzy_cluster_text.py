# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize fuzzy-duplicate candidate text grouped by cluster.

The production verifier re-joins candidate attributes to normalized text and
serves the text through worker memory stores on every run, which makes each
algorithm change cost a full corpus pass. Paying the shuffle once and storing
the text already grouped by cluster turns duplicate detection into an
embarrassingly parallel map over cluster groups.

Every row carries a document ID and a file index. The manifest maps each file
index to its normalized source and shard name, which is enough to write the
co-partitioned marker tree.

Connected components are heavily skewed. A cluster above ``--max-cluster-size``
is split with one MinHash permutation of each document's text. Two documents
with Jaccard J share their minimum n-gram hash with probability J. This is an
operational bound, not a full-recall containment partition: a short excerpt can
meet the containment rule while it has low Jaccard similarity with its source.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority batch --cpu 16 --memory 96GB \
        -- python experiments/datakit/scripts/fuzzy_cluster_text.py \
            --prefix s3://.../marin --candidates datakit/dedup_709f5997 \
            --large-clusters s3://.../user/rav/dedup/large_clusters/v1/large_clusters.parquet \
            --out s3://.../user/rav/dedup/cluster_text/v6 \
            --output-shards 8192 --shards-per-task 96 --max-workers 96
"""

import argparse
import dataclasses
import json
import logging
import os
from collections.abc import Iterator, Mapping
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.copartitioned import CopartitionedSource, build_copartitioned_shards
from marin.execution.artifact import read_record
from marin.processing.classification.deduplication.cluster_text import (
    CLUSTER_TEXT_SUBDIRECTORY,
    MAXIMUM_VERIFICATION_TEXT_CHARS,
    ClusterTextManifest,
    ClusterTextShard,
    resolve_data_path,
    write_cluster_text_manifest,
    write_cluster_text_success,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_text"
_SHARED_OVERSIZED_KEY = "fuzzy_cluster_text_oversized"
SPLIT_NGRAM_SIZE = 5
DEFAULT_MAX_SHARD_FAILURES = 20

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("cluster_key", pa.string(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("text_truncated", pa.bool_(), nullable=False),
        pa.field("file_idx", pa.int32(), nullable=False),
    ]
)


@dataclasses.dataclass(frozen=True)
class TextShard:
    """One normalized shard and the candidate attributes beside it."""

    file_idx: int
    normalized_path: str
    candidate_path: str
    source_key: str
    source_tag: str
    basename: str


def _split_hash(text: str) -> int:
    """One MinHash permutation over the document's word n-grams.

    Returns the minimum 64-bit hash of the case-folded word n-grams. Two
    documents with Jaccard J share this value with probability exactly J, which
    is what makes it a partition key that keeps duplicates together where a
    split on the document ID would scatter them. Python's own string hash is
    salted per process and would place the same document differently on
    different workers, so this uses dupekit's fixed xxh3.
    """
    tokens = text.casefold().split()
    if len(tokens) < SPLIT_NGRAM_SIZE:
        return dupekit.hash_xxh3_64(" ".join(tokens).encode("utf-8", "surrogatepass"))
    shingles = [
        " ".join(tokens[start : start + SPLIT_NGRAM_SIZE]).encode("utf-8", "surrogatepass")
        for start in range(len(tokens) - SPLIT_NGRAM_SIZE + 1)
    ]
    return min(dupekit.hash_xxh3_64_batch(shingles))


def _read_table(path: str, columns: list[str]) -> pa.Table | None:
    """Read selected columns, or None when the file holds no rows.

    A shard with no candidates is written as an empty Parquet file whose schema
    carries no columns at all, so selecting by name raises there.
    """
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows == 0:
            return None
        return parquet.read(columns=columns)


def _join_shard_group(shards: list[TextShard]) -> Iterator[dict[str, Any]]:
    """Join a group of shards in one task.

    The shuffle reads every map output once per reduce partition, so its cost
    scales with the product of the two counts. Grouping input shards cuts the
    map side of that product without changing the result.
    """
    oversized: dict[str, int] = zephyr_worker_ctx().get_shared(_SHARED_OVERSIZED_KEY)
    for shard in shards:
        yield from _join_shard(shard, oversized)


def _join_shard(shard: TextShard, oversized: Mapping[str, int]) -> Iterator[dict[str, Any]]:
    """Join one shard's candidate rows to their normalized text.

    Only a fifth of a normalized shard is a candidate in a typical source, so
    the row selection happens in Arrow and only the selected text crosses into
    Python. Converting every row first was the dominant cost of this stage.
    """
    if not StoragePath(shard.candidate_path).exists():
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidate_shards_missing", 1)
        return
    candidates = _read_table(shard.candidate_path, ["id", "dup_cluster_id"])
    if candidates is None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidate_shards_empty", 1)
        return
    attributes = {
        candidate_id: str(cluster_id)
        for candidate_id, cluster_id in zip(
            candidates.column("id").to_pylist(),
            candidates.column("dup_cluster_id").to_pylist(),
            strict=True,
        )
    }
    if len(attributes) != candidates.num_rows:
        raise ValueError(f"{shard.candidate_path} contains duplicate candidate IDs")
    wanted = candidates.column("id").combine_chunks()

    emitted = 0
    previous_id: str | None = None
    previous_text: str | None = None
    with StoragePath(shard.normalized_path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(columns=["id", "text"]):
            mask = pc.is_in(batch.column("id"), value_set=wanted)
            selected = batch.filter(mask)
            if selected.num_rows:
                ids = selected.column("id").to_pylist()
                texts = selected.column("text").to_pylist()
                for record_id, text in zip(ids, texts, strict=True):
                    raw_text = text or ""
                    cluster_id = attributes.pop(record_id, None)
                    if cluster_id is None:
                        if record_id != previous_id or raw_text != previous_text:
                            raise ValueError(f"Repeated normalized ID {record_id!r} has inconsistent text")
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/repeated_normalized_ids", 1)
                        continue
                    previous_id = record_id
                    previous_text = raw_text
                    text = raw_text
                    text_truncated = len(text) > MAXIMUM_VERIFICATION_TEXT_CHARS
                    if text_truncated:
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/oversized_documents", 1)
                        text = ""
                    splits = oversized.get(cluster_id, 1)
                    cluster_key = cluster_id
                    if splits > 1:
                        split_index = _split_hash(text) % splits
                        cluster_key = f"{cluster_id}:{split_index:04d}"
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/split_members", 1)
                    yield {
                        "cluster_key": cluster_key,
                        "dup_cluster_id": cluster_id,
                        "id": record_id,
                        "text": text,
                        "text_truncated": text_truncated,
                        "file_idx": shard.file_idx,
                    }
                    emitted += 1

    if attributes:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidates_without_text", len(attributes))
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_with_missing_text", 1)
        raise ValueError(
            f"{shard.candidate_path} holds {len(attributes)} IDs absent from {shard.normalized_path} "
            f"against {emitted} joined, first {sorted(attributes)[:3]!r}"
        )
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/members", emitted)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_joined", 1)


def group_key_of(cluster_key: str, groups: int) -> int:
    """Stable output group for one cluster group.

    Partitions on the *split* key, not the cluster ID. The largest component
    holds 831 million members: routing all of its splits to one file would
    hand one reduce task the whole component and undo the split.

    Keep ``groups`` well above the reduce-task count. Zephyr hashes this key
    again to pick a reduce task, so one group per task is balls-into-bins:
    ``e**-1`` of the tasks get nothing and the unlucky ones get five or more
    groups. The v5 run measured 37% idle reducers and a 31 GB partition
    against a 5.6 GB mean. Eight groups per task flattens that tail.
    """
    return dupekit.hash_xxh3_64(cluster_key.encode("utf-8")) % groups


def cluster_sort_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    """Keep a cluster contiguous and put its longest documents first."""
    return record["cluster_key"], -len(record["text"]), record["id"]


def _write_group(group: int, records: Iterator[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    path = prefix_join(output_dir, f"part-{group:06d}.parquet")
    result = write_parquet_file(records, path, schema=_TEXT_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_written", result["count"])
    return {**result, "group": group}


def build_shards(prefix: str, candidates: str, output_path: str) -> list[TextShard]:
    """Pair every normalized shard with its candidate attribute shard.

    Candidate source keys identify the normalized inputs. Candidate attributes
    can omit empty shards, but they cannot contain a shard absent from the
    normalized source.
    """
    candidate_path = resolve_data_path(prefix, candidates)
    record = read_record(candidate_path)
    if record is None or record.result is None:
        raise FileNotFoundError(f"No candidate artifact payload at {candidate_path}")
    candidate_artifact = FuzzyDupsAttrData.model_validate(record.result)
    entries, _ = build_copartitioned_shards(
        sources=[
            CopartitionedSource(source_key=source_key, input_dir=prefix_join(prefix, source_key))
            for source_key in sorted(candidate_artifact.sources)
        ],
        output_path=output_path,
    )

    expected_by_source: dict[str, set[str]] = {}
    for entry in entries:
        expected_by_source.setdefault(entry.source_key, set()).add(entry.basename)
    for source_key, source in candidate_artifact.sources.items():
        candidate_dir = resolve_data_path(prefix, source.attr_dir)
        candidate_paths = StoragePath(prefix_join(candidate_dir, "*.parquet")).glob()
        candidate_basenames = {os.path.basename(str(path)) for path in candidate_paths}
        extra = candidate_basenames - expected_by_source[source_key]
        if extra:
            raise ValueError(f"Candidate source {source_key!r} has unexpected shards: {sorted(extra)!r}")

    return [
        TextShard(
            file_idx=entry.file_idx,
            normalized_path=entry.input_path,
            candidate_path=prefix_join(
                resolve_data_path(prefix, candidate_artifact.sources[entry.source_key].attr_dir), entry.basename
            ),
            source_key=entry.source_key,
            source_tag=entry.source_tag,
            basename=entry.basename,
        )
        for entry in entries
    ]


def load_oversized(large_clusters_path: str, max_cluster_size: int) -> tuple[dict[str, int], int]:
    """Return split counts and the estimated member count for oversized clusters."""
    summary_path = StoragePath(prefix_join(str(StoragePath(large_clusters_path).parent), "summary.json"))
    if not summary_path.exists():
        raise FileNotFoundError(f"Large-cluster planner summary is absent: {summary_path}")
    summary = json.loads(summary_path.read_bytes())
    minimum_size = int(summary["minimum_size"])
    if minimum_size > max_cluster_size:
        raise ValueError(
            f"Large-cluster plan starts at {minimum_size} members, above the materializer cap {max_cluster_size}"
        )
    with StoragePath(large_clusters_path).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["dup_cluster_id", "size"])
    oversized = {
        str(cluster_id): -(-int(size) // max_cluster_size)
        for cluster_id, size in zip(
            table.column("dup_cluster_id").to_pylist(), table.column("size").to_pylist(), strict=True
        )
        if size > max_cluster_size
    }
    members = sum(int(size) for size in table.column("size").to_pylist() if size > max_cluster_size)
    return oversized, members


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--large-clusters", required=True, help="large_clusters.parquet from fuzzy_large_clusters.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cluster-size", type=int, default=100_000)
    parser.add_argument("--output-shards", type=int, default=4096, help="Reduce tasks")
    parser.add_argument("--groups-per-shard", type=int, default=8, help="Output files assigned to each reduce task")
    parser.add_argument("--shards-per-task", type=int, default=8, help="Input shards joined by one map task")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="128g")
    parser.add_argument("--worker-disk", default="512g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="12g", help="Map task memory")
    parser.add_argument("--task-disk", default="48g")
    parser.add_argument("--reduce-task-ram", default="26g", help="Reduce holds a partition and its sort spill")
    parser.add_argument("--max-shard-failures", type=int, default=DEFAULT_MAX_SHARD_FAILURES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shards = build_shards(args.prefix, args.candidates, args.out)
    oversized, oversized_cluster_members = load_oversized(args.large_clusters, args.max_cluster_size)
    logger.info(
        "Grouping %d shards; %d clusters exceed %d members and will be split",
        len(shards),
        len(oversized),
        args.max_cluster_size,
    )

    manifest = ClusterTextManifest(
        candidates=resolve_data_path(args.prefix, args.candidates),
        max_cluster_size=args.max_cluster_size,
        output_shards=args.output_shards,
        groups_per_shard=args.groups_per_shard,
        split_ngram_size=SPLIT_NGRAM_SIZE,
        oversized_clusters=oversized,
        oversized_cluster_members=oversized_cluster_members,
        shards=[
            ClusterTextShard(
                file_idx=shard.file_idx,
                source_key=shard.source_key,
                source_tag=shard.source_tag,
                basename=shard.basename,
            )
            for shard in shards
        ],
    )
    write_cluster_text_manifest(args.out, manifest)

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    reduce_task = ResourceConfig(cpu=args.task_cpu, ram=args.reduce_task_ram, disk=args.task_disk)
    context = ZephyrContext(
        name="fuzzy-cluster-text",
        resources=worker,
        max_workers=args.max_workers,
        max_shard_failures=args.max_shard_failures,
    )
    context.put(_SHARED_OVERSIZED_KEY, oversized)
    shard_groups = [
        shards[start : start + args.shards_per_task] for start in range(0, len(shards), args.shards_per_task)
    ]
    logger.info("Map side: %d tasks of up to %d shards", len(shard_groups), args.shards_per_task)

    key_groups = args.output_shards * args.groups_per_shard
    logger.info("Reduce side: %d output files across %d reduce tasks", key_groups, args.output_shards)
    pipeline = (
        Dataset.from_list(shard_groups)
        .flat_map(_join_shard_group)
        .group_by(
            key=lambda record: group_key_of(record["cluster_key"], key_groups),
            reducer=lambda group, records: _write_group(
                group, records, prefix_join(args.out, CLUSTER_TEXT_SUBDIRECTORY)
            ),
            sort_by=cluster_sort_key,
            num_output_shards=args.output_shards,
        )
    )
    outcome = context.execute(pipeline, verbose=True, map_task_resources=task, reduce_task_resources=reduce_task)

    payload = {
        "manifest": prefix_join(args.out, "manifest.json"),
        "shards": len(shards),
        "oversized_clusters": len(oversized),
        "oversized_cluster_members": oversized_cluster_members,
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=2).encode())
    write_cluster_text_success(args.out)
    logger.info("Wrote %s", prefix_join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
