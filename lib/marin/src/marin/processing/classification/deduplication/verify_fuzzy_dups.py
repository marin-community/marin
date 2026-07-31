# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify fuzzy-duplicate clusters against full normalized text."""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ActorConfig, ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_WORKERS_PER_JOB, ZephyrContext
from zephyr.memory_store import MemoryStore
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import (
    CopartitionedSource,
    build_copartitioned_shards,
    write_copartitioned_source_manifest,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    VerificationResult,
    prepare_verification_text,
    verify_prepared_candidate,
)

logger = logging.getLogger(__name__)

VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION = 1
PERCENT_HISTOGRAM_METRICS = frozenset({"member_containment", "jaccard", "char_jaccard"})
SCORE_HISTOGRAM_MAX_PERCENT = 100
UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN = 33
_COUNTER_PREFIX = "dedup/fuzzy/verification"
_SHARED_SHARDS_KEY = "verified_fuzzy_dups_shards"


class VerifiedFuzzyDupsPerSource(BaseModel):
    """Attribute output for one normalized source."""

    attr_dir: DatakitArtifactPath


class VerifiedFuzzyDupsAttrData(BaseModel):
    """Sparse, co-partitioned markers for verified fuzzy duplicates."""

    version: str = f"v{VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION}"
    verification: FuzzyVerificationParams
    sources: dict[str, VerifiedFuzzyDupsPerSource]
    counters: dict[str, int | float]

    def attr_dir_for_source(self, source_path: str) -> str:
        """Return the attribute directory for a materialized source path."""
        source_key = datakit_source_key(source_path)
        entry = self.sources.get(source_key)
        if entry is None:
            raise KeyError(f"Verified fuzzy duplicate attributes have no entry for source_key={source_key!r}")
        return entry.attr_dir


@dataclass(frozen=True)
class VerificationShard:
    """One normalized shard and its candidate and output shards."""

    file_idx: int
    normalized_path: str
    candidate_path: str
    output_path: str
    source_key: str
    source_tag: str


@dataclass(frozen=True)
class FuzzyVerificationStoreConfig:
    """Resources and lookup sizing for candidate-document memory stores."""

    max_actors: int
    actor_resources: ResourceConfig
    actor_config: ActorConfig
    max_actor_bytes: int
    recovery_timeout: float
    ready_timeout: float
    lookup_batch_size: int

    def __post_init__(self) -> None:
        if self.max_actors < 1:
            raise ValueError("max_actors must be at least 1")
        if self.lookup_batch_size < 1:
            raise ValueError("lookup_batch_size must be at least 1")


_VERIFIED_DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("dup_representative_id", pa.string(), nullable=False),
        pa.field("dup_representative_source_key", pa.string(), nullable=False),
        pa.field("dup_member_containment", pa.float64(), nullable=False),
        pa.field("dup_jaccard", pa.float64(), nullable=False),
        pa.field("dup_under_tokenized", pa.bool_(), nullable=False),
        pa.field("dup_char_jaccard", pa.float64()),
    ]
)


def _rows(path: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    """Read selected Parquet columns and validate sorted, unique IDs."""
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows == 0:
            return
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} does not contain columns {sorted(missing)}")

        previous_id = None
        for batch in parquet.iter_batches(columns=columns):
            for row in batch.to_pylist():
                row_id = row["id"]
                if previous_id is not None and row_id <= previous_id:
                    raise ValueError(f"{path} IDs are not sorted and unique at {row_id!r}")
                previous_id = row_id
                yield row


def _candidate_cluster_members(shards: list[VerificationShard]) -> Iterator[dict[str, Any]]:
    """Read candidate metadata without carrying document text into the shuffle."""
    for shard in shards:
        yield {"kind": "sentinel", "file_idx": shard.file_idx}
        if not StoragePath(shard.candidate_path).exists():
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_shards_missing", 1)
            continue

        candidates = iter(_rows(shard.candidate_path, ["id", "dup_cluster_id", "is_cluster_canonical"]))
        candidate = next(candidates, None)
        if candidate is None:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_shards_empty", 1)
            continue

        while candidate is not None:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_members", 1)
            yield {
                "kind": "candidate",
                "file_idx": shard.file_idx,
                "source_key": shard.source_key,
                "source_tag": shard.source_tag,
                "id": candidate["id"],
                "dup_cluster_id": str(candidate["dup_cluster_id"]),
                "is_cluster_canonical": candidate["is_cluster_canonical"],
            }
            candidate = next(candidates, None)


def _candidate_text_entries(shard: VerificationShard) -> Iterator[tuple[tuple[int, str], str]]:
    """Join one candidate shard to normalized text for its store partition."""
    if not StoragePath(shard.candidate_path).exists():
        return

    candidates = iter(_rows(shard.candidate_path, ["id"]))
    candidate = next(candidates, None)
    if candidate is None:
        return

    normalized = iter(_rows(shard.normalized_path, ["id", "text"]))
    source = next(normalized, None)
    while candidate is not None:
        while source is not None and source["id"] < candidate["id"]:
            source = next(normalized, None)
        if source is None or source["id"] != candidate["id"]:
            raise ValueError(
                f"{shard.candidate_path} contains ID {candidate['id']!r} " f"that is absent from {shard.normalized_path}"
            )
        yield (shard.file_idx, candidate["id"]), source["text"] or ""
        candidate = next(candidates, None)
        source = next(normalized, None)


def _document_partition(key: tuple[int, str]) -> int:
    """Route a candidate document to its existing co-partitioned file index."""
    return key[0]


def _cluster_key(record: dict[str, Any]) -> tuple[str, str]:
    if record["kind"] == "sentinel":
        return "sentinel", str(record["file_idx"])
    return "cluster", record["dup_cluster_id"]


def _score_bin(score: float) -> str:
    return f"{min(int(score * 100), SCORE_HISTOGRAM_MAX_PERCENT):03d}"


def _size_bin(size: int) -> str:
    return f"{1 << (size.bit_length() - 1)}-{(1 << size.bit_length()) - 1}"


def _result_fields(result: VerificationResult) -> dict[str, Any]:
    return {
        "dup_member_containment": result.member_containment,
        "dup_jaccard": result.jaccard,
        "dup_under_tokenized": result.under_tokenized,
        "dup_char_jaccard": result.char_jaccard,
    }


def _make_cluster_verifier(
    params: FuzzyVerificationParams,
    document_store: MemoryStore[tuple[int, str], str],
    lookup_batch_size: int,
):
    """Build a reducer that verifies each member against one representative."""

    def verify(group_key: tuple[str, str], records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        first = next(records)
        if group_key[0] == "sentinel":
            if next(records, None) is not None:
                raise AssertionError(f"Sentinel group {group_key} has more than one record")
            yield {"kind": "sentinel", "file_idx": first["file_idx"], "id": ""}
            return

        representative = first
        if not representative["is_cluster_canonical"]:
            raise ValueError(f"Cluster {group_key[1]!r} has no canonical member")
        representative_raw_text = document_store.get((representative["file_idx"], representative["id"]))
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_text_chars", len(representative_raw_text))
        representative_text = prepare_verification_text(representative_raw_text, params)
        cluster_size = 1
        accepted = 0
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/clusters", 1)

        for members in batched(records, lookup_batch_size):
            member_texts = document_store.get_many([(member["file_idx"], member["id"]) for member in members])
            for member, member_raw_text in zip(members, member_texts, strict=True):
                cluster_size += 1
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_text_chars", len(member_raw_text))
                if member["is_cluster_canonical"]:
                    raise ValueError(f"Cluster {group_key[1]!r} has more than one canonical member")
                if member["id"] == representative["id"]:
                    if member_raw_text != representative_raw_text:
                        raise ValueError(f"Cluster {group_key[1]!r} has different text for content ID {member['id']!r}")
                    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/decision/delegated_global_exact", 1)
                    counters.pipeline.update_counter(
                        f"{_COUNTER_PREFIX}/source/{member['source_tag']}/decision/delegated_global_exact",
                        1,
                    )
                    continue
                result = verify_prepared_candidate(
                    prepare_verification_text(member_raw_text, params),
                    representative_text,
                    params,
                )
                decision = result.rejection.value if result.rejection is not None else "accepted"
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/decision/{decision}", 1)
                counters.pipeline.update_counter(
                    f"{_COUNTER_PREFIX}/source/{member['source_tag']}/decision/{decision}",
                    1,
                )
                counters.pipeline.update_counter(
                    f"{_COUNTER_PREFIX}/histogram/member_containment/{_score_bin(result.member_containment)}",
                    1,
                )
                counters.pipeline.update_counter(
                    f"{_COUNTER_PREFIX}/histogram/jaccard/{_score_bin(result.jaccard)}",
                    1,
                )
                if result.char_jaccard is not None:
                    counters.pipeline.update_counter(
                        f"{_COUNTER_PREFIX}/histogram/char_jaccard/{_score_bin(result.char_jaccard)}",
                        1,
                    )
                counters.pipeline.update_counter(
                    f"{_COUNTER_PREFIX}/histogram/member_unique/"
                    f"{min(result.member_unique_ngrams, UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN)}",
                    1,
                )
                if not result.accepted:
                    continue

                accepted += 1
                yield {
                    "kind": "verified",
                    "file_idx": member["file_idx"],
                    "id": member["id"],
                    "dup_doc": True,
                    "dup_cluster_id": member["dup_cluster_id"],
                    "dup_representative_id": representative["id"],
                    "dup_representative_source_key": representative["source_key"],
                    **_result_fields(result),
                }

        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/cluster_size/{_size_bin(cluster_size)}", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/cluster_members", cluster_size)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/verified_duplicates", accepted)

    return verify


def _write_verified_shard(file_idx: int, records: Iterator[dict[str, Any]]) -> dict[str, Any]:
    """Write one sparse verified-duplicate attribute shard."""
    shard: VerificationShard = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)[file_idx]
    verified = 0

    def output_rows() -> Iterator[dict[str, Any]]:
        nonlocal verified
        for record in records:
            if record["kind"] == "sentinel":
                continue
            verified += 1
            yield {field.name: record[field.name] for field in _VERIFIED_DUPLICATE_SCHEMA}

    result = write_parquet_file(output_rows(), shard.output_path, schema=_VERIFIED_DUPLICATE_SCHEMA)
    return {**result, "file_idx": file_idx, "verified_duplicates": verified}


def _verification_shards(
    *,
    normalized_sources: dict[str, NormalizedData],
    candidates: FuzzyDupsAttrData,
    output_path: str,
) -> tuple[list[VerificationShard], dict[str, str]]:
    """Build and validate the co-partitioned verification layout."""
    normalized_by_key: dict[str, NormalizedData] = {}
    normalized_entries: list[tuple[str, str, NormalizedData]] = []
    for source_name, source in normalized_sources.items():
        source_key = datakit_source_key(source.main_output_dir)
        if source_key in normalized_by_key:
            raise ValueError(f"normalized_sources contains duplicate source_key={source_key!r}")
        normalized_by_key[source_key] = source
        normalized_entries.append((source_name, source_key, source))

    normalized_keys = set(normalized_by_key)
    candidate_keys = set(candidates.sources)
    if normalized_keys != candidate_keys:
        raise ValueError(
            "Normalized and candidate source sets differ: "
            f"normalized_only={sorted(normalized_keys - candidate_keys)!r}, "
            f"candidate_only={sorted(candidate_keys - normalized_keys)!r}"
        )

    entries, attr_dirs = build_copartitioned_shards(
        sources=[
            CopartitionedSource(source_key=source_key, input_dir=source.main_output_dir)
            for _source_name, source_key, source in sorted(normalized_entries)
        ],
        output_path=output_path,
    )

    expected_by_source: dict[str, set[str]] = {}
    for entry in entries:
        expected_by_source.setdefault(entry.source_key, set()).add(entry.basename)
    for source_key, candidate_source in candidates.sources.items():
        candidate_paths = StoragePath(prefix_join(candidate_source.attr_dir, "*.parquet")).glob()
        candidate_basenames = {os.path.basename(str(path)) for path in candidate_paths}
        extra = candidate_basenames - expected_by_source[source_key]
        if extra:
            raise ValueError(f"Candidate source {source_key!r} has unexpected shards: {sorted(extra)!r}")

    shards = [
        VerificationShard(
            file_idx=entry.file_idx,
            normalized_path=entry.input_path,
            candidate_path=prefix_join(candidates.sources[entry.source_key].attr_dir, entry.basename),
            output_path=entry.output_path,
            source_key=entry.source_key,
            source_tag=entry.source_tag,
        )
        for entry in entries
    ]
    return shards, attr_dirs


def verify_fuzzy_dups(
    *,
    normalized_sources: dict[str, NormalizedData],
    candidates: FuzzyDupsAttrData,
    output_path: str,
    verification_params: FuzzyVerificationParams,
    store_config: FuzzyVerificationStoreConfig,
    max_parallelism: int = MAX_WORKERS_PER_JOB,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
) -> VerifiedFuzzyDupsAttrData:
    """Verify existing candidate clusters and write sparse duplicate markers."""
    if not normalized_sources:
        raise ValueError("verify_fuzzy_dups requires at least one normalized source")
    if max_parallelism < 1:
        raise ValueError("max_parallelism must be at least 1")
    shards, attr_dirs = _verification_shards(
        normalized_sources=normalized_sources,
        candidates=candidates,
        output_path=output_path,
    )
    if not shards:
        raise ValueError("verify_fuzzy_dups found no normalized Parquet shards")

    ctx_kwargs: dict[str, Any] = {
        "name": "verify-fuzzy-dups",
        "max_workers": max_parallelism,
        "resources": worker_resources or ResourceConfig(cpu=2, ram="16g", disk="16g"),
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    if map_task_resources is not None:
        ctx_kwargs["map_task_resources"] = map_task_resources
    if reduce_task_resources is not None:
        ctx_kwargs["reduce_task_resources"] = reduce_task_resources
    shards.sort(key=lambda shard: shard.file_idx)
    if [shard.file_idx for shard in shards] != list(range(len(shards))):
        raise ValueError("verification shards do not have contiguous file indices")
    file_shards = min(max_parallelism, len(shards))
    shard_groups: list[list[VerificationShard]] = [[] for _ in range(file_shards)]
    for index, shard in enumerate(shards):
        shard_groups[index % file_shards].append(shard)

    with ZephyrContext(**ctx_kwargs) as ctx:
        ctx.put(_SHARED_SHARDS_KEY, {shard.file_idx: shard for shard in shards})
        document_store = ctx.load_memory_store(
            Dataset.from_list(shards).flat_map(_candidate_text_entries),
            name="fuzzy-verification-documents",
            hash_key=_document_partition,
            num_actors=min(store_config.max_actors, len(shards)),
            actor_resources=store_config.actor_resources,
            actor_config=store_config.actor_config,
            max_actor_bytes=store_config.max_actor_bytes,
            recovery_timeout=store_config.recovery_timeout,
            ready_timeout=store_config.ready_timeout,
        )
        store_stats = document_store.stats()
        pipeline = (
            Dataset.from_list(shard_groups)
            .flat_map(_candidate_cluster_members)
            .group_by(
                key=_cluster_key,
                sort_by=lambda record: (
                    0 if record["kind"] == "sentinel" or record["is_cluster_canonical"] else 1,
                    record["file_idx"],
                    record.get("id", ""),
                ),
                reducer=_make_cluster_verifier(
                    verification_params,
                    document_store,
                    store_config.lookup_batch_size,
                ),
                # Cluster IDs can use all workers, including when there are fewer input files.
                num_output_shards=max_parallelism,
            )
            .group_by(
                key=lambda record: record["file_idx"],
                sort_by=lambda record: record["id"],
                reducer=_write_verified_shard,
                num_output_shards=file_shards,
            )
        )
        outcome = ctx.execute(pipeline, verbose=True)
    write_copartitioned_source_manifest(output_path=output_path, attr_dirs=attr_dirs)

    verified = sum(result["verified_duplicates"] for result in outcome.results)
    output_counters = dict(outcome.counters)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/actors"] = len(store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/items"] = sum(stat.num_items for stat in store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/serialized_bytes"] = sum(
        stat.serialized_bytes for stat in store_stats
    )
    output_counters[f"{_COUNTER_PREFIX}/memory_store/max_actor_serialized_bytes"] = max(
        stat.serialized_bytes for stat in store_stats
    )
    logger.info(
        "Verified %d fuzzy duplicates from %d candidate members across %d shards",
        verified,
        int(output_counters.get(f"{_COUNTER_PREFIX}/candidate_members", 0)),
        len(shards),
    )
    return VerifiedFuzzyDupsAttrData(
        verification=verification_params,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dir) for source_key, attr_dir in attr_dirs.items()
        },
        counters=output_counters,
    )


def verify_fuzzy_dups_step(
    *,
    name: str,
    normalized_steps: dict[str, StepSpec],
    candidates_step: StepSpec,
    verification_params: FuzzyVerificationParams,
    store_config: FuzzyVerificationStoreConfig,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a step that verifies one existing fuzzy-candidate artifact."""
    ordered_normalized_steps = {name: normalized_steps[name] for name in sorted(normalized_steps)}
    return StepSpec(
        name=name,
        deps=[*ordered_normalized_steps.values(), candidates_step],
        fn=lambda output_path: verify_fuzzy_dups(
            normalized_sources={
                source_name: read_artifact(step.output_path, NormalizedData)
                for source_name, step in ordered_normalized_steps.items()
            },
            candidates=read_artifact(candidates_step.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            verification_params=verification_params,
            store_config=store_config,
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        ),
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
        },
        override_output_path=override_output_path,
    )
