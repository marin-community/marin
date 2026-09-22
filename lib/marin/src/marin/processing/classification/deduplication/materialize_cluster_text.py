# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize fuzzy candidates with the production text and split policy.

Oversized components use a text MinHash partition and a document-ID subdivision.
The split can separate duplicates. Documents retain at most 64 Mi characters.
"""

import dataclasses
import hashlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import CopartitionedSource, build_copartitioned_shards
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact, read_record
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_text import (
    CLUSTER_TEXT_MANIFEST_FILENAME,
    CLUSTER_TEXT_SUBDIRECTORY,
    ClusterTextData,
    ClusterTextManifest,
    ClusterTextParams,
    ClusterTextShard,
    resolve_data_path,
    write_cluster_text_manifest,
    write_cluster_text_success,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.large_clusters import LargeClusterPlan

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_text"
_SHARED_OVERSIZED_KEY = "fuzzy_cluster_text_oversized"
DEFAULT_MAX_SHARD_FAILURES = 20

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("cluster_key", pa.string(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
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


def _split_hash(text: str, ngram_size: int) -> int:
    """Minimum xxh3 hash of the case-folded word n-grams."""
    tokens = text.casefold().split()
    if len(tokens) < ngram_size:
        return dupekit.hash_xxh3_64(" ".join(tokens).encode("utf-8", "ignore"))
    shingles = [
        " ".join(tokens[start : start + ngram_size]).encode("utf-8", "ignore")
        for start in range(len(tokens) - ngram_size + 1)
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


def _join_shard_group(shards: list[TextShard], params: ClusterTextParams) -> Iterator[dict[str, Any]]:
    """Group map inputs to reduce shuffle fan-out."""
    oversized: dict[str, int] = zephyr_worker_ctx().get_shared(_SHARED_OVERSIZED_KEY)
    for shard in shards:
        yield from _join_shard(shard, oversized, params)


def _join_shard(
    shard: TextShard, oversized: Mapping[str, int], params: ClusterTextParams = ClusterTextParams()
) -> Iterator[dict[str, Any]]:
    """Select candidate text in Arrow before converting it to Python."""
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
    emitted_text_hashes: dict[str, bytes] = {}
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
                    text_hash = hashlib.sha256(raw_text.encode("utf-8", "surrogatepass")).digest()
                    cluster_id = attributes.pop(record_id, None)
                    if cluster_id is None:
                        if text_hash != emitted_text_hashes[record_id]:
                            raise ValueError(f"Repeated normalized ID {record_id!r} has inconsistent text")
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/repeated_normalized_ids", 1)
                        continue
                    emitted_text_hashes[record_id] = text_hash
                    text = raw_text
                    if len(text) > params.maximum_document_chars:
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/oversized_documents", 1)
                        text = text[: params.maximum_document_chars]
                    splits = oversized.get(cluster_id, 1)
                    cluster_key = cluster_id
                    if splits > 1:
                        split_index = _split_hash(text, params.split_ngram_size) % splits
                        sub_index = dupekit.hash_xxh3_64(record_id.encode()) % params.split_subdivisions
                        cluster_key = f"{cluster_id}:{split_index:04d}:{sub_index:02d}"
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/split_members", 1)
                    yield {
                        "cluster_key": cluster_key,
                        "dup_cluster_id": cluster_id,
                        "id": record_id,
                        "text": text,
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
    """Hash the split key so one oversized component can use multiple tasks."""
    return dupekit.hash_xxh3_64(cluster_key.encode("utf-8")) % groups


def cluster_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """Order by cluster and ID; ID breaks equal-length processing ties in the solver."""
    return record["cluster_key"], record["id"]


def _write_group(group: int, records: Iterator[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    path = prefix_join(output_dir, f"part-{group:06d}.parquet")
    result = write_parquet_file(records, path, schema=_TEXT_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_written", result["count"])
    return {**result, "group": group}


def build_shards(
    prefix: str, candidates: str, output_path: str, normalized_sources: Sequence[CopartitionedSource]
) -> list[TextShard]:
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
    if {source.source_key for source in normalized_sources} != set(candidate_artifact.sources):
        raise ValueError("Normalized sources do not match the candidate source keys")
    entries, _ = build_copartitioned_shards(
        sources=sorted(normalized_sources, key=lambda source: source.source_key),
        output_path=output_path,
    )

    expected_by_source: dict[str, set[str]] = {}
    for entry in entries:
        expected_by_source.setdefault(entry.source_key, set()).add(entry.basename)
    for source_key, source in candidate_artifact.sources.items():
        candidate_dir = resolve_data_path(prefix, source.attr_dir)
        candidate_paths = StoragePath(prefix_join(candidate_dir, "*.parquet")).glob()
        candidate_basenames = {path.name for path in candidate_paths}
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


def load_oversized(plan: LargeClusterPlan, max_cluster_size: int, candidate_path: str) -> tuple[dict[str, int], int]:
    """Return split counts and estimated members from the candidate plan."""
    if plan.candidates != candidate_path:
        raise ValueError(f"Large-cluster plan candidates {plan.candidates!r} do not match {candidate_path!r}")
    if plan.params.minimum_size > max_cluster_size:
        raise ValueError("Large-cluster plan threshold is above the materializer cap")
    with StoragePath(plan.counts_path).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["dup_cluster_id", "size"])
    sizes = table.column("size").to_pylist()
    oversized = {
        str(cluster_id): -(-int(size) // max_cluster_size)
        for cluster_id, size in zip(table.column("dup_cluster_id").to_pylist(), sizes, strict=True)
        if size > max_cluster_size
    }
    members = sum(int(size) for size in sizes if size > max_cluster_size)
    return oversized, members


def materialize_cluster_text(
    *,
    prefix: str,
    candidates: str,
    normalized_sources: Sequence[CopartitionedSource],
    plan: LargeClusterPlan,
    output_path: str,
    params: ClusterTextParams = ClusterTextParams(),
    shards_per_task: int = 8,
    max_workers: int = 64,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    max_shard_failures: int = DEFAULT_MAX_SHARD_FAILURES,
) -> ClusterTextData:
    """Join normalized text to candidates and write groups with explicit lineage."""
    if shards_per_task < 1 or max_workers < 1:
        raise ValueError("shards_per_task and max_workers must be positive")
    shards = build_shards(prefix, candidates, output_path, normalized_sources)
    candidate_path = resolve_data_path(prefix, candidates)
    oversized, oversized_cluster_members = load_oversized(plan, params.max_cluster_size, candidate_path)
    logger.info(
        "Grouping %d shards; %d clusters exceed %d members and will be split",
        len(shards),
        len(oversized),
        params.max_cluster_size,
    )

    manifest = ClusterTextManifest(
        candidates=candidate_path,
        max_cluster_size=params.max_cluster_size,
        output_shards=params.output_shards,
        groups_per_shard=params.groups_per_shard,
        split_ngram_size=params.split_ngram_size,
        split_subdivisions=params.split_subdivisions,
        maximum_document_chars=params.maximum_document_chars,
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
    write_cluster_text_manifest(output_path, manifest)

    context = ZephyrContext(
        name="fuzzy-cluster-text",
        resources=worker_resources,
        max_workers=max_workers,
        max_shard_failures=max_shard_failures,
    )
    context.put(_SHARED_OVERSIZED_KEY, oversized)
    shard_groups = [shards[start : start + shards_per_task] for start in range(0, len(shards), shards_per_task)]
    logger.info("Map side: %d tasks of up to %d shards", len(shard_groups), shards_per_task)

    key_groups = params.output_shards * params.groups_per_shard
    logger.info("Reduce side: %d output files across %d reduce tasks", key_groups, params.output_shards)
    pipeline = (
        Dataset.from_list(shard_groups)
        .flat_map(lambda group: _join_shard_group(group, params))
        .group_by(
            key=lambda record: group_key_of(record["cluster_key"], key_groups),
            reducer=lambda group, records: _write_group(
                group, records, prefix_join(output_path, CLUSTER_TEXT_SUBDIRECTORY)
            ),
            sort_by=cluster_sort_key,
            num_output_shards=params.output_shards,
        )
    )
    outcome = context.execute(
        pipeline, verbose=True, map_task_resources=map_task_resources, reduce_task_resources=reduce_task_resources
    )

    payload = {
        "manifest": prefix_join(output_path, CLUSTER_TEXT_MANIFEST_FILENAME),
        "shards": len(shards),
        "oversized_clusters": len(oversized),
        "oversized_cluster_members": oversized_cluster_members,
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(output_path, "summary.json")).write_bytes(json.dumps(payload, indent=2).encode())
    write_cluster_text_success(output_path)
    logger.info("Wrote %s", prefix_join(output_path, "summary.json"))
    return ClusterTextData(path=output_path, params=params, counters=outcome.counters)


def cluster_text_step(
    *,
    name: str,
    normalized_steps: Sequence[StepSpec],
    candidates: StepSpec,
    plan: StepSpec,
    params: ClusterTextParams = ClusterTextParams(),
    shards_per_task: int = 8,
    max_workers: int = 64,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    max_shard_failures: int = DEFAULT_MAX_SHARD_FAILURES,
) -> StepSpec:
    """Create a text shuffle with normalized, candidate, and planner dependencies."""

    def build(output_path: str) -> ClusterTextData:
        normalized = [read_artifact(step.output_path, NormalizedData) for step in normalized_steps]
        return materialize_cluster_text(
            prefix=marin_prefix(),
            candidates=candidates.output_path,
            normalized_sources=[
                CopartitionedSource(
                    source_key=datakit_source_key(source.main_output_dir), input_dir=source.main_output_dir
                )
                for source in normalized
            ],
            plan=read_artifact(plan.output_path, LargeClusterPlan),
            output_path=output_path,
            params=params,
            shards_per_task=shards_per_task,
            max_workers=max_workers,
            worker_resources=worker_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
            max_shard_failures=max_shard_failures,
        )

    return StepSpec(
        name=name,
        deps=[*normalized_steps, candidates, plan],
        hash_attrs={"version": 1, "params": params.model_dump(mode="json")},
        fn=build,
    )
