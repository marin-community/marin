# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize fuzzy-duplicate candidate text grouped by cluster.

The production verifier re-joins candidate attributes to normalized text and
serves the text through worker memory stores on every run, which makes each
algorithm change cost a full corpus pass. Paying the shuffle once and storing
the text already grouped by cluster turns duplicate detection into an
embarrassingly parallel map over cluster groups.

Every row carries the provenance needed to rebuild a co-partitioned attribute
tree from anything computed on top of it: the normalized shard is named by
``source_key`` and ``basename``, the row inside it by ``id``, and ``file_idx``
indexes the manifest this job writes beside the data.

Connected components are heavily skewed. A cluster above ``--max-cluster-size``
is split in two levels. The first is a single MinHash permutation of the
document's own text: two documents with Jaccard J share their minimum n-gram
hash with probability J, so it keeps duplicate pairs together where a split on
the document ID would scatter them. The second is a uniform split on the
content ID, and it exists because the first cannot bound a group at all when
the cluster is mostly one duplicate group -- identical documents share a
minimum hash by construction. Both are deterministic functions of the document.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority batch --cpu 16 --memory 96GB \
        -- python experiments/datakit/scripts/fuzzy_cluster_text.py \
            --prefix s3://.../marin --candidates datakit/dedup_709f5997 \
            --verified datakit/verify_fuzzy_dups_c757e4f0 \
            --large-clusters s3://.../user/rav/dedup/large_clusters/v1/large_clusters.parquet \
            --out s3://.../user/rav/dedup/cluster_text/v5
"""

import argparse
import dataclasses
import json
import logging
from collections.abc import Iterator
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    ClusterDocument,
    find_duplicates,
)
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_text"
_SHARED_SHARDS_KEY = "fuzzy_cluster_text_shards"
_SHARED_OVERSIZED_KEY = "fuzzy_cluster_text_oversized"
SPLIT_NGRAM_SIZE = 5
# Identical documents share a minimum n-gram hash, which is exactly what keeps
# duplicates in one group -- and exactly why that hash cannot bound a group
# that is mostly one duplicate group. The largest component produced a split of
# 1,684,557 members against a 100,000 cap. A uniform sub-split on the content ID
# caps it: every sub-group still solves internally, so the cost is one extra
# survivor per sub-group, against 1.68 million removals.
SPLIT_SUBDIVISIONS = 16

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("cluster_key", pa.string(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("split_index", pa.int32(), nullable=False),
        pa.field("sub_index", pa.int32(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("file_idx", pa.int32(), nullable=False),
        pa.field("source_tag", pa.string(), nullable=False),
        pa.field("basename", pa.string(), nullable=False),
        pa.field("row_index", pa.int32(), nullable=False),
        pa.field("is_cluster_canonical", pa.bool_(), nullable=False),
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
        return dupekit.hash_xxh3_64(" ".join(tokens).encode("utf-8", "ignore"))
    shingles = [
        " ".join(tokens[start : start + SPLIT_NGRAM_SIZE]).encode("utf-8", "ignore")
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
    for shard in shards:
        yield from _join_shard(shard)


def _join_shard(shard: TextShard) -> Iterator[dict[str, Any]]:
    """Join one shard's candidate rows to their normalized text.

    Only a fifth of a normalized shard is a candidate in a typical source, so
    the row selection happens in Arrow and only the selected text crosses into
    Python. Converting every row first was the dominant cost of this stage.
    """
    if not StoragePath(shard.candidate_path).exists():
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidate_shards_missing", 1)
        return
    oversized: dict[str, int] = zephyr_worker_ctx().get_shared(_SHARED_OVERSIZED_KEY)

    candidates = _read_table(shard.candidate_path, ["id", "dup_cluster_id", "is_cluster_canonical"])
    if candidates is None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidate_shards_empty", 1)
        return
    attributes = {
        candidate_id: (str(cluster_id), bool(canonical))
        for candidate_id, cluster_id, canonical in zip(
            candidates.column("id").to_pylist(),
            candidates.column("dup_cluster_id").to_pylist(),
            candidates.column("is_cluster_canonical").to_pylist(),
            strict=True,
        )
    }
    wanted = candidates.column("id").combine_chunks()

    emitted = 0
    offset = 0
    with StoragePath(shard.normalized_path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(columns=["id", "text"]):
            mask = pc.is_in(batch.column("id"), value_set=wanted)
            selected = pc.indices_nonzero(mask).to_pylist()
            if selected:
                ids = batch.column("id").take(pa.array(selected)).to_pylist()
                texts = batch.column("text").take(pa.array(selected)).to_pylist()
                for position, record_id, text in zip(selected, ids, texts, strict=True):
                    attribute = attributes.pop(record_id, None)
                    if attribute is None:
                        # A repeated normalized ID is byte-identical text under
                        # one content hash; the first occurrence answers the join.
                        continue
                    cluster_id, canonical = attribute
                    text = text or ""
                    splits = oversized.get(cluster_id, 1)
                    split_index = 0
                    sub_index = 0
                    cluster_key = cluster_id
                    if splits > 1:
                        split_index = _split_hash(text) % splits
                        sub_index = dupekit.hash_xxh3_64(record_id.encode()) % SPLIT_SUBDIVISIONS
                        cluster_key = f"{cluster_id}:{split_index:04d}:{sub_index:02d}"
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/split_members", 1)
                    yield {
                        "cluster_key": cluster_key,
                        "dup_cluster_id": cluster_id,
                        "split_index": split_index,
                        "sub_index": sub_index,
                        "id": record_id,
                        "text": text,
                        "file_idx": shard.file_idx,
                        "source_tag": shard.source_tag,
                        "basename": shard.basename,
                        "row_index": offset + position,
                        "is_cluster_canonical": canonical,
                    }
                    emitted += 1
            offset += batch.num_rows

    if attributes:
        # The candidate tree names the Focus Crawl by a different extraction,
        # so a shard of it can carry IDs this normalized tree never held.
        # Counting them keeps one mismatched source from ending the run.
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/candidates_without_text", len(attributes))
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_with_missing_text", 1)
        if len(attributes) > max(emitted, 1):
            raise ValueError(
                f"{shard.candidate_path} holds {len(attributes)} IDs absent from {shard.normalized_path} "
                f"against {emitted} joined, first {sorted(attributes)[:3]!r}"
            )
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/members", emitted)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_joined", 1)


def output_shard_of(cluster_key: str, shards: int) -> int:
    """Stable output shard for one cluster group.

    Partitions on the *split* key, not the cluster ID. The largest component
    holds 831 million members: routing all of its splits to one file would
    hand one reduce task the whole component and undo the split.
    """
    return dupekit.hash_xxh3_64(cluster_key.encode("utf-8")) % shards


_MARKER_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
        pa.field("dup_cluster_key", pa.string(), nullable=False),
        pa.field("dup_representative_id", pa.string(), nullable=False),
        pa.field("dup_representative_source_tag", pa.string(), nullable=False),
        pa.field("dup_containment", pa.float32(), nullable=False),
        pa.field("dup_jaccard", pa.float32(), nullable=False),
        pa.field("dup_novel_tokens", pa.int32(), nullable=False),
    ]
)


def _solve_cluster(members: list[dict[str, Any]], params: ClusterDedupParams) -> Iterator[dict[str, Any]]:
    """Emit one marker per member that a longer survivor already holds."""
    documents = [
        ClusterDocument(id=row["id"], text=row["text"], file_idx=row["file_idx"], source_tag=row["source_tag"])
        for row in members
    ]
    for removal in find_duplicates(documents, params):
        member = members[removal.member_index]
        representative = members[removal.representative_index]
        yield {
            "file_idx": member["file_idx"],
            "id": member["id"],
            "dup_doc": True,
            "dup_cluster_key": member["cluster_key"],
            "dup_representative_id": representative["id"],
            "dup_representative_source_tag": representative["source_tag"],
            "dup_containment": removal.containment,
            "dup_jaccard": removal.jaccard,
            "dup_novel_tokens": removal.novel_tokens,
        }


def _write_group(
    shard: int,
    records: Iterator[dict[str, Any]],
    output_dir: str,
    params: ClusterDedupParams | None,
) -> Iterator[dict[str, Any]]:
    """Write one grouped text file, and solve its clusters when asked.

    Solving here would save a pass, but the reducer's memory is already spoken
    for by the shuffle merge, and holding a cluster's text on top of that is
    what kills the stage. Grouping alone keeps this task I/O bound; the solver
    runs afterwards as a map over the grouped files, where it gets the whole
    machine.
    """
    path = prefix_join(output_dir, f"part-{shard:05d}.parquet")
    written = 0
    clusters = 0
    duplicates = 0
    markers: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    current: str | None = None

    def flush() -> None:
        nonlocal clusters, duplicates
        if params is None or len(pending) < 2:
            pending.clear()
            return
        clusters += 1
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/cluster_size/{_size_bin(len(pending))}", 1)
        for marker in _solve_cluster(pending, params):
            markers.append(marker)
            duplicates += 1
        pending.clear()

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal written, current
        for record in records:
            written += 1
            if params is not None:
                if record["cluster_key"] != current:
                    flush()
                    current = record["cluster_key"]
                pending.append(record)
            yield {field.name: record[field.name] for field in _TEXT_SCHEMA}
        flush()

    result = write_parquet_file(rows(), path, schema=_TEXT_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_written", written)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/clusters", clusters)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicates", duplicates)
    del result
    yield from markers


_SIZE_BINS = (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)


def _size_bin(size: int) -> str:
    for edge in _SIZE_BINS:
        if size < edge:
            return f"{edge:06d}"
    return "999999"


def _write_markers(file_idx: int, records: Iterator[dict[str, Any]], output_path: str) -> dict[str, Any]:
    """Write one shard's markers into the co-partitioned attribute tree."""
    shards: dict[int, TextShard] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
    shard = shards[file_idx]
    path = prefix_join(prefix_join(output_path, f"outputs/{shard.source_tag}"), shard.basename)
    written = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal written
        for record in records:
            written += 1
            yield {field.name: record[field.name] for field in _MARKER_SCHEMA}

    result = write_parquet_file(rows(), path, schema=_MARKER_SCHEMA)
    return {**result, "file_idx": file_idx, "markers": written}


def build_shards(prefix: str, candidates: str, verified: str) -> list[TextShard]:
    """Pair every normalized shard with its candidate attribute shard.

    Source order and tags come from the verified artifact, so ``file_idx``
    reproduces the numbering the production verifier used.
    """
    candidate_artifact = json.loads(
        StoragePath(prefix_join(prefix_join(prefix, candidates), ".artifact.json")).read_bytes()
    )["result"]
    verified_artifact = json.loads(
        StoragePath(prefix_join(prefix_join(prefix, verified), ".artifact.json")).read_bytes()
    )["result"]

    # The two artifacts share every source key but one: the candidate tree
    # names the Focus Crawl by its pre-#8111 extraction. Pair the leftovers.
    candidate_sources = dict(candidate_artifact["sources"])
    only_candidate = sorted(set(candidate_sources) - set(verified_artifact["sources"]))
    only_verified = sorted(set(verified_artifact["sources"]) - set(candidate_sources))
    if len(only_candidate) != len(only_verified) or len(only_candidate) > 1:
        raise ValueError(f"cannot pair sources: candidate_only={only_candidate!r}, verified_only={only_verified!r}")
    for verified_key, candidate_key in zip(only_verified, only_candidate, strict=True):
        candidate_sources[verified_key] = candidate_sources[candidate_key]

    fs, _ = url_to_fs(prefix)
    shards: list[TextShard] = []
    for source_key, entry in sorted(verified_artifact["sources"].items(), key=lambda item: item[1]["source_tag"]):
        normalized_dir = prefix_join(prefix, source_key)
        candidate_dir = prefix_join(prefix, str(candidate_sources[source_key]["attr_dir"]))
        _, root = url_to_fs(normalized_dir)
        names = sorted(
            str(path).rsplit("/", 1)[-1] for path in fs.ls(root, detail=False) if str(path).endswith(".parquet")
        )
        for name in names:
            shards.append(
                TextShard(
                    file_idx=len(shards),
                    normalized_path=prefix_join(normalized_dir, name),
                    candidate_path=prefix_join(candidate_dir, name),
                    source_key=source_key,
                    source_tag=entry["source_tag"],
                    basename=name,
                )
            )
    return shards


def load_oversized(large_clusters_path: str, max_cluster_size: int) -> dict[str, int]:
    """Clusters above the cap, mapped to the number of splits they need."""
    with StoragePath(large_clusters_path).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["dup_cluster_id", "size"])
    return {
        cluster_id: -(-size // max_cluster_size)
        for cluster_id, size in zip(
            table.column("dup_cluster_id").to_pylist(), table.column("size").to_pylist(), strict=True
        )
        if size > max_cluster_size
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verified", required=True, help="Verified artifact that pins the source order")
    parser.add_argument("--large-clusters", required=True, help="large_clusters.parquet from fuzzy_large_clusters.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cluster-size", type=int, default=100_000)
    parser.add_argument("--output-shards", type=int, default=4096)
    parser.add_argument("--shards-per-task", type=int, default=8, help="Input shards joined by one map task")
    parser.add_argument("--limit-shards", type=int, default=0, help="Join only the first N shards; for a smoke run")
    parser.add_argument("--solve-inline", action="store_true", help="Also solve clusters in the reducer")
    parser.add_argument("--minimum-containment", type=float, default=0.98)
    parser.add_argument("--no-zero-novel-tokens", action="store_true")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="128g")
    parser.add_argument("--worker-disk", default="512g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="12g", help="Map task memory")
    parser.add_argument("--task-disk", default="48g")
    parser.add_argument("--reduce-task-ram", default="26g", help="Reduce holds a partition and its sort spill")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shards = build_shards(args.prefix, args.candidates, args.verified)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    oversized = load_oversized(args.large_clusters, args.max_cluster_size)
    logger.info(
        "Grouping %d shards; %d clusters exceed %d members and will be split",
        len(shards),
        len(oversized),
        args.max_cluster_size,
    )

    manifest = {
        "version": "v1",
        "candidates": args.candidates,
        "verified": args.verified,
        "max_cluster_size": args.max_cluster_size,
        "output_shards": args.output_shards,
        "duplicate_rule": None,
        "split_ngram_size": SPLIT_NGRAM_SIZE,
        "split_subdivisions": SPLIT_SUBDIVISIONS,
        "oversized_clusters": oversized,
        "shards": [dataclasses.asdict(shard) for shard in shards],
    }
    StoragePath(prefix_join(args.out, "manifest.json")).write_bytes(json.dumps(manifest).encode())

    params = (
        ClusterDedupParams(
            minimum_containment=args.minimum_containment,
            accept_zero_novel_tokens=not args.no_zero_novel_tokens,
        )
        if args.solve_inline
        else None
    )
    logger.info("Duplicate rule: %s", params.model_dump_json() if params else "grouping only")
    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    reduce_task = ResourceConfig(cpu=args.task_cpu, ram=args.reduce_task_ram, disk=args.task_disk)
    context = ZephyrContext(name="fuzzy-cluster-text", resources=worker, max_workers=args.max_workers)
    context.put(_SHARED_SHARDS_KEY, shards)
    context.put(_SHARED_OVERSIZED_KEY, oversized)
    shard_groups = [
        shards[start : start + args.shards_per_task] for start in range(0, len(shards), args.shards_per_task)
    ]
    logger.info("Map side: %d tasks of up to %d shards", len(shard_groups), args.shards_per_task)

    grouped = Dataset.from_list(shard_groups).flat_map(_join_shard_group).group_by(
        key=lambda record: output_shard_of(record["cluster_key"], args.output_shards),
        reducer=lambda shard, records: _write_group(shard, records, prefix_join(args.out, "text"), params),
        sort_by=lambda record: (record["cluster_key"], record["id"]),
        num_output_shards=args.output_shards,
    )
    pipeline = (
        grouped.group_by(
            key=lambda record: record["file_idx"],
            reducer=lambda file_idx, records: _write_markers(file_idx, records, prefix_join(args.out, "verified")),
            sort_by=lambda record: record["id"],
            num_output_shards=min(len(shards), args.output_shards),
        )
        if params is not None
        else grouped
    )
    outcome = context.execute(pipeline, verbose=True, map_task_resources=task, reduce_task_resources=reduce_task)

    markers = sum(result.get("markers", 0) for result in outcome.results if isinstance(result, dict))
    del markers
    payload = {
        "manifest": prefix_join(args.out, "manifest.json"),
        "params": params.model_dump(mode="json") if params else None,
        "shards": len(shards),
        "oversized_clusters": len(oversized),
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Wrote %s", prefix_join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
