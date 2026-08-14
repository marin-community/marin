# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify fuzzy-duplicate clusters against full normalized text."""

import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from itertools import batched, chain, groupby
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from fray.types import EnvironmentConfig, ResourceConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_IRIS_WORKER_REPLICAS, ZephyrContext
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
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    PreparedVerificationText,
    VerificationResult,
    line_count_ratio,
    prepare_verification_text,
    verify_prepared_candidate,
)

logger = logging.getLogger(__name__)

VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION = 3
PERCENT_HISTOGRAM_METRICS = frozenset({"member_containment", "jaccard", "char_jaccard", "local_line_count_ratio"})
SCORE_HISTOGRAM_MAX_PERCENT = 100
UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN = 33
_COUNTER_PREFIX = "dedup/fuzzy/verification"
_SHARED_SHARDS_KEY = "verified_fuzzy_dups_shards"
_LOCAL_TOKEN_SEQUENCE_REJECTION = "local_token_sequence_differs"
_LOCAL_LINE_COUNT_REJECTION = "local_line_count_ratio_below_threshold"
# Bounds on the head scan that picks each cluster anchor. Cluster size is
# heavily skewed - the p99 cluster holds 13 members - so a short head covers
# effectively every cluster while a pathological one stays bounded.
ANCHOR_SCAN_RECORDS = 64
ANCHOR_SCAN_CHARS = 2_000_000
# The reduce spills to /tmp through zephyr's external sort, and the run files live
# until the merge finishes. One shard was measured holding 8 GiB, and a worker runs
# several shards at once, so scratch is sized well above worker RAM. On Kubernetes
# this becomes the pod's ephemeral-storage limit, and exceeding it evicts the pod.
VERIFICATION_WORKER_SCRATCH = "256g"
DOCUMENT_TEXT_COMPRESSION_LEVEL = 1
PARQUET_READ_BATCH_SIZE = 4_096
_DOCUMENT_TEXT_CODECS = threading.local()


@dataclass
class _SharedArrowMemoryPool:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_users: int = 0
    memory_pool: pa.MemoryPool | None = None
    previous_pool: pa.MemoryPool | None = None


_SHARED_ARROW_MEMORY_POOL = _SharedArrowMemoryPool()


@dataclass
class _TextAttachmentControl:
    enabled: bool = True


class RepresentativeKind(StrEnum):
    """The retained document used for a direct verification."""

    CLUSTER_CANONICAL = "cluster_canonical"
    CLUSTER_LONGEST = "cluster_longest"
    LOCAL_REPRESENTATIVE = "local_representative"


class LocalRepresentativeParams(BaseModel):
    """Bounds and score gates for local representative verification."""

    model_config = ConfigDict(frozen=True)

    maximum_comparisons_per_document: int = Field(ge=1)
    maximum_representatives_per_cluster: int = Field(ge=1)
    maximum_local_representative_chars: int = Field(ge=1)
    maximum_local_representative_chars_per_cluster: int = Field(ge=1)
    minimum_local_line_count_ratio: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def comparisons_fit_representative_limit(self) -> "LocalRepresentativeParams":
        """Reject a comparison limit that cannot be reached."""
        if self.maximum_comparisons_per_document > self.maximum_representatives_per_cluster:
            raise ValueError("maximum_comparisons_per_document cannot exceed maximum_representatives_per_cluster")
        return self


REFERENCE_LOCAL_REPRESENTATIVE_PARAMS = LocalRepresentativeParams(
    maximum_comparisons_per_document=2,
    maximum_representatives_per_cluster=32,
    maximum_local_representative_chars=500_000,
    maximum_local_representative_chars_per_cluster=2_000_000,
    minimum_local_line_count_ratio=0.8,
)


class VerifiedFuzzyDupsPerSource(BaseModel):
    """Attribute output for one normalized source."""

    attr_dir: DatakitArtifactPath
    source_tag: str


class VerifiedFuzzyDupsAttrData(BaseModel):
    """Sparse, co-partitioned markers for verified fuzzy duplicates."""

    version: str = f"v{VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION}"
    verification: FuzzyVerificationParams
    local_representatives: LocalRepresentativeParams
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
    minhash_path: str
    output_path: str
    source_key: str
    source_tag: str


@dataclass(frozen=True)
class _VerificationLayout:
    shards: list[VerificationShard]
    attr_dirs: dict[str, str]
    source_tags: dict[str, str]


@dataclass(frozen=True)
class FuzzyVerificationStoreConfig:
    """Resources and lookup sizing for candidate-document memory stores."""

    recovery_timeout: float
    ready_timeout: float
    lookup_batch_size: int
    shards_per_worker: int
    load_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive")
        if self.lookup_batch_size < 1:
            raise ValueError("lookup_batch_size must be at least 1")
        if self.shards_per_worker < 1:
            raise ValueError("shards_per_worker must be at least 1")
        if self.load_concurrency < 1:
            raise ValueError("load_concurrency must be at least 1")


_VERIFIED_DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("dup_representative_id", pa.string(), nullable=False),
        pa.field("dup_representative_source_key", pa.string(), nullable=False),
        pa.field("dup_representative_kind", pa.string(), nullable=False),
        pa.field("dup_shared_lsh_buckets", pa.int32(), nullable=False),
        pa.field("dup_comparisons", pa.int32(), nullable=False),
        pa.field("dup_member_containment", pa.float64(), nullable=False),
        pa.field("dup_jaccard", pa.float64(), nullable=False),
        pa.field("dup_under_tokenized", pa.bool_(), nullable=False),
        pa.field("dup_char_jaccard", pa.float64()),
        pa.field("dup_local_line_count_ratio", pa.float64()),
    ]
)


def _parquet_rows(path: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    """Read selected Parquet columns without a row-order requirement."""
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows == 0:
            return
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} does not contain columns {sorted(missing)}")

        for batch in parquet.iter_batches(
            batch_size=PARQUET_READ_BATCH_SIZE,
            columns=columns,
            use_threads=False,
        ):
            arrays = [batch.column(index) for index in range(batch.num_columns)]
            for row_index in range(batch.num_rows):
                yield {name: array[row_index].as_py() for name, array in zip(columns, arrays, strict=True)}


def _rows(path: str, columns: list[str], *, repeated_ids: bool = False) -> Iterator[dict[str, Any]]:
    """Read selected Parquet columns and validate ascending IDs.

    ``repeated_ids`` accepts a repeated ID and yields only its first row. A
    source that normalizes with ``DedupMode.NONE`` keeps every record carrying
    an upstream content hash, so equal IDs there mean byte-identical text and
    either row answers the join. Descending IDs stay an error: the join walks
    both sides forward once and cannot recover from them.
    """
    previous_id = None
    for row in _parquet_rows(path, columns):
        row_id = row["id"]
        if previous_id is not None and row_id < previous_id:
            raise ValueError(f"{path} IDs are not sorted at {row_id!r}")
        if previous_id is not None and row_id == previous_id:
            if not repeated_ids:
                raise ValueError(f"{path} IDs are not unique at {row_id!r}")
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/repeated_source_ids", 1)
            continue
        previous_id = row_id
        yield row


@contextmanager
def _system_arrow_memory_pool() -> Iterator[pa.MemoryPool]:
    """Share an Arrow decoder pool safely across concurrent shard readers."""
    state = _SHARED_ARROW_MEMORY_POOL
    with state.lock:
        if state.active_users == 0:
            state.previous_pool = pa.default_memory_pool()
            state.memory_pool = pa.system_memory_pool()
            pa.set_memory_pool(state.memory_pool)
        state.active_users += 1
        memory_pool = state.memory_pool
    assert memory_pool is not None
    try:
        yield memory_pool
    finally:
        with state.lock:
            state.active_users -= 1
            if state.active_users == 0:
                memory_pool.release_unused()
                previous_pool = state.previous_pool
                assert previous_pool is not None
                pa.set_memory_pool(previous_pool)
                state.memory_pool = None
                state.previous_pool = None


def _copy_to_arrow_buffer(data: bytes, memory_pool: pa.MemoryPool) -> pa.Buffer:
    buffer = pa.allocate_buffer(len(data), memory_pool=memory_pool)
    memoryview(buffer).cast("B")[:] = data
    return buffer


def _candidate_documents(shards: list[VerificationShard]) -> Iterator[tuple[tuple[int, str], pa.Buffer]]:
    """Join candidate IDs to compressed text without a normalized row-order requirement."""
    value_memory_pool = pa.mimalloc_memory_pool()
    with _system_arrow_memory_pool() as memory_pool:
        for shard in shards:
            if not StoragePath(shard.candidate_path).exists():
                continue

            candidate_ids: dict[str, str | None] = {
                row["id"]: row["id"]
                for row in _rows(shard.candidate_path, ["id", "dup_cluster_id", "is_cluster_canonical"])
            }
            if not candidate_ids:
                continue

            compressor = zstd.ZstdCompressor(level=DOCUMENT_TEXT_COMPRESSION_LEVEL)
            loaded = 0
            text_bytes = 0
            stored_bytes = 0
            for source in _parquet_rows(shard.normalized_path, ["id", "text"]):
                source_id = source["id"]
                if source_id not in candidate_ids:
                    continue
                candidate_id = candidate_ids[source_id]
                if candidate_id is None:
                    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/repeated_source_ids", 1)
                    continue
                candidate_ids[source_id] = None
                text = (source["text"] or "").encode()
                compressed_bytes = compressor.compress(text)
                compressed_text = _copy_to_arrow_buffer(compressed_bytes, value_memory_pool)
                loaded += 1
                text_bytes += len(text)
                stored_bytes += len(compressed_bytes)
                del text, compressed_bytes
                yield (shard.file_idx, candidate_id), compressed_text

            missing_id = next(
                (candidate_id for candidate_id in candidate_ids.values() if candidate_id is not None),
                None,
            )
            if missing_id is not None:
                raise ValueError(
                    f"{shard.candidate_path} contains ID {missing_id!r} " f"that is absent from {shard.normalized_path}"
                )
            memory_pool.release_unused()
            logger.info(
                "Prepared memory-store shard %d with %d candidate texts: %d bytes compressed to %d bytes",
                shard.file_idx,
                loaded,
                text_bytes,
                stored_bytes,
            )


def _decompress_document_text(value: pa.Buffer) -> str:
    decompressor = getattr(_DOCUMENT_TEXT_CODECS, "decompressor", None)
    if decompressor is None:
        decompressor = zstd.ZstdDecompressor()
        _DOCUMENT_TEXT_CODECS.decompressor = decompressor
    return decompressor.decompress(value).decode()


def _joined_cluster_members(shards: list[VerificationShard]) -> Iterator[dict[str, Any]]:
    """Join candidate attributes to stored LSH buckets, leaving text in the store."""
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

        minhash_rows = iter(_rows(shard.minhash_path, ["id", "buckets"], repeated_ids=True))
        minhash = next(minhash_rows, None)
        while candidate is not None:
            while minhash is not None and minhash["id"] < candidate["id"]:
                minhash = next(minhash_rows, None)
            if minhash is None or minhash["id"] != candidate["id"]:
                raise ValueError(
                    f"{shard.candidate_path} contains ID {candidate['id']!r} "
                    f"that is absent from {shard.minhash_path}"
                )
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_members", 1)
            yield {
                "kind": "candidate",
                "file_idx": shard.file_idx,
                "source_key": shard.source_key,
                "source_tag": shard.source_tag,
                "id": candidate["id"],
                "dup_cluster_id": str(candidate["dup_cluster_id"]),
                "is_cluster_canonical": candidate["is_cluster_canonical"],
                "buckets": tuple(sorted(set(minhash["buckets"] or []))),
            }
            candidate = next(candidates, None)
            minhash = next(minhash_rows, None)


def _document_partition(positions: Mapping[int, int], key: tuple[int, str]) -> int:
    """Route a candidate document to its shard's position in this layout.

    The store requires ``hash_key(key) % partitions`` to name the source
    partition holding the key. File indices are global to the corpus, so a
    layout covering a subset of shards must route by position instead.
    """
    return positions[key[0]]


def _attach_document_text(
    records: Iterator[dict[str, Any]],
    document_store: MemoryStore[tuple[int, str], pa.Buffer],
    lookup_batch_size: int,
    control: _TextAttachmentControl | None = None,
) -> Iterator[dict[str, Any]]:
    """Fetch bounded text batches while preserving reducer record order."""
    for record_batch in batched(records, lookup_batch_size):
        if control is not None and not control.enabled:
            yield from record_batch
            continue
        compressed_texts = document_store.get_many([(record["file_idx"], record["id"]) for record in record_batch])
        for record, compressed_text in zip(record_batch, compressed_texts, strict=True):
            text = _decompress_document_text(compressed_text)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_text_chars", len(text))
            yield {**record, "text": text}


def _cluster_key(record: dict[str, Any]) -> tuple[str, str]:
    if record["kind"] == "sentinel":
        return "sentinel", str(record["file_idx"])
    return "cluster", record["dup_cluster_id"]


def _cluster_sort_key(record: dict[str, Any]) -> bytes:
    """Order a cluster by content ID so that equal IDs group together.

    The order must not depend on which record the reducer removes as its
    anchor, so it rests on the content ID alone. The reducer relies on that:
    it pulls one record out of the stream and still needs the remainder in
    ascending ID order to group equal IDs.
    """
    content_id = record.get("id", "").encode()
    return len(content_id).to_bytes(8, byteorder="big") + content_id + record["file_idx"].to_bytes(8, byteorder="big")


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


@dataclass(frozen=True)
class _RetainedRepresentative:
    """A retained document that can verify later cluster members."""

    id: str
    source_key: str
    prepared: PreparedVerificationText
    buckets: frozenset[str]
    kind: RepresentativeKind


def _record_document_decision(record: dict[str, Any], decision: str) -> None:
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/decision/{decision}", 1)
    counters.pipeline.update_counter(
        f"{_COUNTER_PREFIX}/source/{record['source_tag']}/decision/{decision}",
        1,
    )


def _record_comparison(
    result: VerificationResult,
    representative_kind: RepresentativeKind,
    decision: str,
    local_line_count_ratio: float | None = None,
) -> None:
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/direct_comparisons", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/comparison/{decision}", 1)
    counters.pipeline.update_counter(
        f"{_COUNTER_PREFIX}/comparison_representative/{representative_kind.value}",
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
    if local_line_count_ratio is not None:
        counters.pipeline.update_counter(
            f"{_COUNTER_PREFIX}/histogram/local_line_count_ratio/{_score_bin(local_line_count_ratio)}",
            1,
        )
    counters.pipeline.update_counter(
        f"{_COUNTER_PREFIX}/histogram/member_unique/"
        f"{min(result.member_unique_ngrams, UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN)}",
        1,
    )


@dataclass(frozen=True)
class _LocalVerificationDecision:
    accepted: bool
    reason: str
    line_count_ratio: float | None


def _local_verification_gate(
    member: PreparedVerificationText,
    representative: PreparedVerificationText,
    result: VerificationResult,
    params: LocalRepresentativeParams,
) -> _LocalVerificationDecision:
    """Apply the near-equality gates used only for local representatives."""
    if not result.accepted:
        assert result.rejection is not None
        return _LocalVerificationDecision(
            accepted=False,
            reason=result.rejection.value,
            line_count_ratio=None,
        )
    if member.text.casefold().split() != representative.text.casefold().split():
        return _LocalVerificationDecision(
            accepted=False,
            reason=_LOCAL_TOKEN_SEQUENCE_REJECTION,
            line_count_ratio=None,
        )

    local_line_count_ratio = line_count_ratio(member.text, representative.text)
    if local_line_count_ratio < params.minimum_local_line_count_ratio:
        return _LocalVerificationDecision(
            accepted=False,
            reason=_LOCAL_LINE_COUNT_REJECTION,
            line_count_ratio=local_line_count_ratio,
        )
    return _LocalVerificationDecision(
        accepted=True,
        reason="accepted",
        line_count_ratio=local_line_count_ratio,
    )


@dataclass(frozen=True)
class _StoredComparisonRequest:
    representative_texts: tuple[str, ...]
    verification_params: FuzzyVerificationParams
    local_params: LocalRepresentativeParams
    return_text: bool = False


@dataclass(frozen=True)
class _StoredComparisonResult:
    text: str | None
    comparisons: tuple[VerificationResult, ...]
    local_decisions: tuple[_LocalVerificationDecision, ...]


@dataclass
class _DeferredRecordGroup:
    member_id: str
    records: list[dict[str, Any]]
    nominee_indices: list[int]
    shared_counts: dict[int, int]
    results: list[_StoredComparisonResult | None]


def _compare_document_text(
    text: str,
    request: _StoredComparisonRequest,
    representative_cache: dict[tuple[str, ...], list[PreparedVerificationText]] | None = None,
) -> _StoredComparisonResult:
    if not request.representative_texts:
        return _StoredComparisonResult(text=text, comparisons=(), local_decisions=())

    prepared_member = prepare_verification_text(text, request.verification_params)
    prepared_representatives = (
        None if representative_cache is None else representative_cache.get(request.representative_texts)
    )
    if prepared_representatives is None:
        prepared_representatives = [
            prepare_verification_text(representative_text, request.verification_params)
            for representative_text in request.representative_texts
        ]
        if representative_cache is not None:
            representative_cache[request.representative_texts] = prepared_representatives
    primary_result = verify_prepared_candidate(
        prepared_member,
        prepared_representatives[0],
        request.verification_params,
    )
    comparisons = [primary_result]
    local_decisions: list[_LocalVerificationDecision] = []
    if not primary_result.accepted:
        for prepared_representative in prepared_representatives[1:]:
            result = verify_prepared_candidate(prepared_member, prepared_representative, request.verification_params)
            comparisons.append(result)
            decision = _local_verification_gate(prepared_member, prepared_representative, result, request.local_params)
            local_decisions.append(decision)
    return _StoredComparisonResult(
        text=text if request.return_text else None,
        comparisons=tuple(comparisons),
        local_decisions=tuple(local_decisions),
    )


def _compare_stored_documents(
    inputs: list[tuple[pa.Buffer, _StoredComparisonRequest]],
) -> list[_StoredComparisonResult]:
    """Decompress and compare a worker-local document batch."""
    representative_cache: dict[tuple[str, ...], list[PreparedVerificationText]] = {}
    return [
        _compare_document_text(_decompress_document_text(value), request, representative_cache)
        for value, request in inputs
    ]


def _choose_longest_representative(
    records_with_text: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """Return the longest document of a bounded head, and the rest in ID order.

    The accept rule is directional: a member is removed only when it is no
    longer than the representative. Anchoring on the connected-components
    canonical therefore throws away every removal in a cluster whose canonical
    is not its longest document, and that canonical is the component ID
    minimum, which carries no relation to length.

    Only the head is buffered, so memory stays bounded on a huge cluster. The
    bounds are deliberately independent of the local-representative budgets:
    how far to scan for the longest document is a question about memory, not
    about how many representatives a cluster may retain.

    Removing one record keeps the remainder in ascending ID order, which the
    caller relies on to group equal IDs.
    """
    head: list[dict[str, Any]] = []
    buffered_chars = 0
    for record in records_with_text:
        head.append(record)
        buffered_chars += len(record["text"])
        if len(head) >= ANCHOR_SCAN_RECORDS or buffered_chars >= ANCHOR_SCAN_CHARS:
            break
    longest = max(range(len(head)), key=lambda index: (len(head[index]["text"]), head[index]["id"]))
    if longest > 0:
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_longer_than_first", 1)
    representative = head.pop(longest)
    return representative, chain(head, records_with_text)


@dataclass(frozen=True)
class _SplitRecordGroup:
    first: dict[str, Any]
    remaining: Iterator[dict[str, Any]]


def _split_record_group(records: Iterator[dict[str, Any]]) -> _SplitRecordGroup:
    return _SplitRecordGroup(first=next(records), remaining=records)


def _make_cluster_verifier(
    verification_params: FuzzyVerificationParams,
    local_params: LocalRepresentativeParams,
    document_store: MemoryStore[tuple[int, str], pa.Buffer],
    lookup_batch_size: int,
):
    """Build a reducer that uses bounded, retained local representatives."""

    def verify(group_key: tuple[str, str], records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        first = next(records)
        if group_key[0] == "sentinel":
            if next(records, None) is not None:
                raise AssertionError(f"Sentinel group {group_key} has more than one record")
            yield {"kind": "sentinel", "file_idx": first["file_idx"], "id": ""}
            return

        attachment_control = _TextAttachmentControl()
        records_with_text = _attach_document_text(
            chain((first,), records),
            document_store,
            lookup_batch_size,
            attachment_control,
        )
        representative, records_with_text = _choose_longest_representative(records_with_text)
        # The cluster canonical is now provenance only. It still names the
        # cluster, but it no longer decides what every member is compared
        # against, because the component ID minimum is unrelated to length.
        representative_kind = RepresentativeKind.CLUSTER_CANONICAL
        if not representative["is_cluster_canonical"]:
            representative_kind = RepresentativeKind.CLUSTER_LONGEST
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_not_canonical", 1)
        primary = _RetainedRepresentative(
            id=representative["id"],
            source_key=representative["source_key"],
            prepared=prepare_verification_text(representative["text"], verification_params),
            buckets=frozenset(representative["buckets"]),
            kind=representative_kind,
        )
        retained = [primary]
        bucket_representatives: dict[str, list[int]] = defaultdict(list)
        local_representative_chars = 0
        cluster_size = 1
        # The representative is no longer guaranteed to be the canonical, so the
        # one-canonical-per-cluster invariant is counted across the cluster.
        canonicals_seen = int(representative["is_cluster_canonical"])
        accepted = 0
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/clusters", 1)

        def add_local_representative(member: dict[str, Any], prepared: PreparedVerificationText) -> bool:
            nonlocal local_representative_chars
            if len(retained) >= local_params.maximum_representatives_per_cluster:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/cluster_limit", 1)
                return False
            if prepared.chars > local_params.maximum_local_representative_chars:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/document_chars", 1)
                return False
            if local_representative_chars + prepared.chars > local_params.maximum_local_representative_chars_per_cluster:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/cluster_chars", 1)
                return False

            representative_index = len(retained)
            local = _RetainedRepresentative(
                id=member["id"],
                source_key=member["source_key"],
                prepared=prepared,
                buckets=frozenset(member["buckets"]),
                kind=RepresentativeKind.LOCAL_REPRESENTATIVE,
            )
            retained.append(local)
            for bucket in local.buckets:
                bucket_representatives[bucket].append(representative_index)
            local_representative_chars += prepared.chars
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/local_representatives_added", 1)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/local_representative_chars", prepared.chars)
            return True

        record_groups = groupby(records_with_text, key=lambda record: record["id"])
        frozen_record_groups: Iterator[tuple[str, Iterator[dict[str, Any]]]] | None = None
        for member_id, same_id_records in record_groups:
            record_group = _split_record_group(same_id_records)
            member = record_group.first
            representative_id_group = member_id == primary.id
            if representative_id_group:
                exact_records = chain((member,), record_group.remaining)
                expected_text = representative["text"]
            else:
                exact_records = record_group.remaining
                expected_text = member["text"]

            for exact_record in exact_records:
                cluster_size += 1
                canonicals_seen += exact_record["is_cluster_canonical"]
                if exact_record["text"] != expected_text:
                    raise ValueError(f"Cluster {group_key[1]!r} has different text for content ID {member_id!r}")
                _record_document_decision(exact_record, "delegated_global_exact")
            if representative_id_group:
                continue

            cluster_size += 1
            canonicals_seen += member["is_cluster_canonical"]
            if canonicals_seen > 1:
                raise ValueError(f"Cluster {group_key[1]!r} has more than one canonical member")

            prepared_member = prepare_verification_text(member["text"], verification_params)
            member_buckets = frozenset(member["buckets"])
            comparison_count = 1
            matched_representative: _RetainedRepresentative | None = None
            matched_result: VerificationResult | None = None
            matched_local_line_count_ratio: float | None = None
            shared_buckets = len(member_buckets & primary.buckets)
            result = verify_prepared_candidate(prepared_member, primary.prepared, verification_params)
            comparison_decision = result.rejection.value if result.rejection is not None else "accepted"
            _record_comparison(result, primary.kind, comparison_decision)
            if result.accepted:
                matched_representative = primary
                matched_result = result
            else:
                shared_counts: dict[int, int] = defaultdict(int)
                for bucket in member_buckets:
                    for representative_index in bucket_representatives.get(bucket, ()):
                        shared_counts[representative_index] += 1
                nominees = sorted(shared_counts, key=lambda index: (-shared_counts[index], index))
                local_comparison_limit = local_params.maximum_comparisons_per_document - 1
                if len(nominees) > local_comparison_limit:
                    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/comparison_limit_reached", 1)
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/local_representative_nominees", len(nominees))

                for representative_index in nominees[:local_comparison_limit]:
                    local = retained[representative_index]
                    comparison_count += 1
                    local_result = verify_prepared_candidate(prepared_member, local.prepared, verification_params)
                    local_decision = _local_verification_gate(
                        prepared_member,
                        local.prepared,
                        local_result,
                        local_params,
                    )
                    _record_comparison(
                        local_result,
                        local.kind,
                        local_decision.reason,
                        local_decision.line_count_ratio,
                    )
                    if local_decision.accepted:
                        matched_representative = local
                        matched_result = local_result
                        matched_local_line_count_ratio = local_decision.line_count_ratio
                        shared_buckets = shared_counts[representative_index]
                        break

            counters.pipeline.update_counter(
                f"{_COUNTER_PREFIX}/comparisons_per_document/{comparison_count}",
                1,
            )
            if matched_representative is None:
                _record_document_decision(member, "retained_no_match")
                added = add_local_representative(member, prepared_member)
                if added and len(retained) == local_params.maximum_representatives_per_cluster:
                    attachment_control.enabled = False
                    frozen_record_groups = record_groups
                    break
                continue

            assert matched_result is not None
            _record_document_decision(member, "accepted")
            counters.pipeline.update_counter(
                f"{_COUNTER_PREFIX}/accepted_representative/{matched_representative.kind.value}",
                1,
            )
            accepted += 1
            yield {
                "kind": "verified",
                "file_idx": member["file_idx"],
                "id": member["id"],
                "dup_doc": True,
                "dup_cluster_id": member["dup_cluster_id"],
                "dup_representative_id": matched_representative.id,
                "dup_representative_source_key": matched_representative.source_key,
                "dup_representative_kind": matched_representative.kind.value,
                "dup_shared_lsh_buckets": shared_buckets,
                "dup_comparisons": comparison_count,
                **_result_fields(matched_result),
                "dup_local_line_count_ratio": matched_local_line_count_ratio,
            }

        if frozen_record_groups is not None:
            materialized_groups = ((member_id, list(group)) for member_id, group in frozen_record_groups)
            for group_batch in batched(materialized_groups, lookup_batch_size):
                frozen_groups: list[_DeferredRecordGroup] = []
                remote_requests: list[tuple[tuple[int, str], _StoredComparisonRequest]] = []
                remote_slots: list[tuple[_DeferredRecordGroup, int]] = []
                representative_cache: dict[tuple[str, ...], list[PreparedVerificationText]] = {}

                for member_id, group_records in group_batch:
                    member = group_records[0]
                    nominee_indices: list[int] = []
                    shared_counts: dict[int, int] = {}
                    if member_id != primary.id:
                        member_buckets = frozenset(member["buckets"])
                        mutable_shared_counts: dict[int, int] = defaultdict(int)
                        for bucket in member_buckets:
                            for representative_index in bucket_representatives.get(bucket, ()):
                                mutable_shared_counts[representative_index] += 1
                        shared_counts = dict(mutable_shared_counts)
                        nominees = sorted(shared_counts, key=lambda index: (-shared_counts[index], index))
                        local_comparison_limit = local_params.maximum_comparisons_per_document - 1
                        nominee_indices = nominees[:local_comparison_limit]

                    frozen_group = _DeferredRecordGroup(
                        member_id=member_id,
                        records=group_records,
                        nominee_indices=nominee_indices,
                        shared_counts=shared_counts,
                        results=[None] * len(group_records),
                    )
                    frozen_groups.append(frozen_group)

                    for record_index, record in enumerate(group_records):
                        is_comparison = record_index == 0 and member_id != primary.id
                        representative_texts = ()
                        if is_comparison:
                            representative_texts = (
                                primary.prepared.text,
                                *(retained[index].prepared.text for index in nominee_indices),
                            )
                        request = _StoredComparisonRequest(
                            representative_texts=representative_texts,
                            verification_params=verification_params,
                            local_params=local_params,
                            return_text=not is_comparison or len(group_records) > 1,
                        )
                        if "text" in record:
                            frozen_group.results[record_index] = _compare_document_text(
                                record["text"],
                                request,
                                representative_cache,
                            )
                            continue
                        remote_requests.append(((record["file_idx"], record["id"]), request))
                        remote_slots.append((frozen_group, record_index))

                remote_results = document_store.compute_many(remote_requests, _compare_stored_documents)
                for (frozen_group, record_index), result in zip(remote_slots, remote_results, strict=True):
                    frozen_group.results[record_index] = result
                    if result.comparisons:
                        text_chars = result.comparisons[0].member_chars
                    else:
                        assert result.text is not None
                        text_chars = len(result.text)
                    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_text_chars", text_chars)

                for frozen_group in frozen_groups:
                    member = frozen_group.records[0]
                    resolved_results = frozen_group.results
                    assert all(result is not None for result in resolved_results)
                    results = [result for result in resolved_results if result is not None]
                    cluster_size += len(frozen_group.records)
                    canonicals_seen += sum(record["is_cluster_canonical"] for record in frozen_group.records)
                    if canonicals_seen > 1:
                        raise ValueError(f"Cluster {group_key[1]!r} has more than one canonical member")

                    if frozen_group.member_id == primary.id:
                        for record, result in zip(frozen_group.records, results, strict=True):
                            if result.text != primary.prepared.text:
                                raise ValueError(
                                    f"Cluster {group_key[1]!r} has different text "
                                    f"for content ID {frozen_group.member_id!r}"
                                )
                            _record_document_decision(record, "delegated_global_exact")
                        continue

                    member_result = results[0]
                    if len(frozen_group.records) > 1:
                        expected_text = member_result.text
                        assert expected_text is not None
                        for exact_record, exact_result in zip(frozen_group.records[1:], results[1:], strict=True):
                            if exact_result.text != expected_text:
                                raise ValueError(
                                    f"Cluster {group_key[1]!r} has different text "
                                    f"for content ID {frozen_group.member_id!r}"
                                )
                            _record_document_decision(exact_record, "delegated_global_exact")

                    comparison_results = member_result.comparisons
                    assert comparison_results
                    primary_result = comparison_results[0]
                    comparison_count = 1
                    matched_representative: _RetainedRepresentative | None = None
                    matched_result: VerificationResult | None = None
                    matched_local_line_count_ratio: float | None = None
                    member_buckets = frozenset(member["buckets"])
                    shared_buckets = len(member_buckets & primary.buckets)
                    primary_decision = (
                        primary_result.rejection.value if primary_result.rejection is not None else "accepted"
                    )
                    _record_comparison(primary_result, primary.kind, primary_decision)
                    if primary_result.accepted:
                        matched_representative = primary
                        matched_result = primary_result
                    else:
                        nominees = sorted(
                            frozen_group.shared_counts,
                            key=lambda index: (-frozen_group.shared_counts[index], index),
                        )
                        local_comparison_limit = local_params.maximum_comparisons_per_document - 1
                        if len(nominees) > local_comparison_limit:
                            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/comparison_limit_reached", 1)
                        counters.pipeline.update_counter(
                            f"{_COUNTER_PREFIX}/local_representative_nominees",
                            len(nominees),
                        )
                        for representative_index, local_result, local_decision in zip(
                            frozen_group.nominee_indices,
                            comparison_results[1:],
                            member_result.local_decisions,
                            strict=True,
                        ):
                            local = retained[representative_index]
                            comparison_count += 1
                            _record_comparison(
                                local_result,
                                local.kind,
                                local_decision.reason,
                                local_decision.line_count_ratio,
                            )
                            if local_decision.accepted:
                                matched_representative = local
                                matched_result = local_result
                                matched_local_line_count_ratio = local_decision.line_count_ratio
                                shared_buckets = frozen_group.shared_counts[representative_index]
                                break

                    counters.pipeline.update_counter(
                        f"{_COUNTER_PREFIX}/comparisons_per_document/{comparison_count}",
                        1,
                    )
                    if matched_representative is None:
                        _record_document_decision(member, "retained_no_match")
                        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/cluster_limit", 1)
                        continue

                    assert matched_result is not None
                    _record_document_decision(member, "accepted")
                    counters.pipeline.update_counter(
                        f"{_COUNTER_PREFIX}/accepted_representative/{matched_representative.kind.value}",
                        1,
                    )
                    accepted += 1
                    yield {
                        "kind": "verified",
                        "file_idx": member["file_idx"],
                        "id": member["id"],
                        "dup_doc": True,
                        "dup_cluster_id": member["dup_cluster_id"],
                        "dup_representative_id": matched_representative.id,
                        "dup_representative_source_key": matched_representative.source_key,
                        "dup_representative_kind": matched_representative.kind.value,
                        "dup_shared_lsh_buckets": shared_buckets,
                        "dup_comparisons": comparison_count,
                        **_result_fields(matched_result),
                        "dup_local_line_count_ratio": matched_local_line_count_ratio,
                    }

        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/cluster_size/{_size_bin(cluster_size)}", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representatives_per_cluster/{_size_bin(len(retained))}", 1)
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
    minhash_sources: dict[str, MinHashAttrData],
    candidates: FuzzyDupsAttrData,
    output_path: str,
    shard_basenames: frozenset[str] | None = None,
) -> _VerificationLayout:
    """Build and validate the co-partitioned verification layout.

    ``shard_basenames`` keeps only the named shard triples. The trees are
    co-partitioned, so a triple is self-contained and dropping one loses no
    row. Cluster membership is not co-partitioned, though, thus the caller
    must supply a set that holds every member of every cluster it verifies.
    """
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
    minhash_by_key: dict[str, MinHashAttrData] = {}
    for source_name, minhash in minhash_sources.items():
        if minhash.source_key in minhash_by_key:
            raise ValueError(f"minhash_sources contains duplicate source_key={minhash.source_key!r}")
        minhash_by_key[minhash.source_key] = minhash
        if minhash.params != candidates.params:
            raise ValueError(
                f"MinHash source {source_name!r} params {minhash.params} do not match candidate params "
                f"{candidates.params}"
            )
    minhash_keys = set(minhash_by_key)
    if normalized_keys != candidate_keys or normalized_keys != minhash_keys:
        raise ValueError(
            "Normalized, candidate, and MinHash source sets differ: "
            f"normalized_only={sorted(normalized_keys - candidate_keys)!r}, "
            f"candidate_only={sorted(candidate_keys - normalized_keys)!r}, "
            f"normalized_without_minhash={sorted(normalized_keys - minhash_keys)!r}, "
            f"minhash_only={sorted(minhash_keys - normalized_keys)!r}"
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
    for source_key, minhash_source in minhash_by_key.items():
        minhash_paths = StoragePath(prefix_join(minhash_source.attr_dir, "*.parquet")).glob()
        minhash_basenames = {os.path.basename(str(path)) for path in minhash_paths}
        if minhash_basenames != expected_by_source[source_key]:
            raise ValueError(
                f"MinHash source {source_key!r} shard set differs from normalized shards: "
                f"missing={sorted(expected_by_source[source_key] - minhash_basenames)!r}, "
                f"extra={sorted(minhash_basenames - expected_by_source[source_key])!r}"
            )

    shards = [
        VerificationShard(
            file_idx=entry.file_idx,
            normalized_path=entry.input_path,
            candidate_path=prefix_join(candidates.sources[entry.source_key].attr_dir, entry.basename),
            minhash_path=prefix_join(minhash_by_key[entry.source_key].attr_dir, entry.basename),
            output_path=entry.output_path,
            source_key=entry.source_key,
            source_tag=entry.source_tag,
        )
        for entry in entries
    ]
    if shard_basenames is not None:
        shards = [shard for shard in shards if shard.normalized_path.rsplit("/", 1)[-1] in shard_basenames]
        if not shards:
            raise ValueError("shard_basenames selected no shard of the verification layout")
    source_tags = {entry.source_key: entry.source_tag for entry in entries}
    return _VerificationLayout(shards=shards, attr_dirs=attr_dirs, source_tags=source_tags)


def verify_fuzzy_dups(
    *,
    normalized_sources: dict[str, NormalizedData],
    minhash_sources: dict[str, MinHashAttrData],
    candidates: FuzzyDupsAttrData,
    output_path: str,
    verification_params: FuzzyVerificationParams,
    local_representative_params: LocalRepresentativeParams,
    store_config: FuzzyVerificationStoreConfig,
    max_output_shards: int,
    max_workers: int | None = None,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    actor_environment: EnvironmentConfig | None = None,
    shard_basenames: frozenset[str] | None = None,
) -> VerifiedFuzzyDupsAttrData:
    """Verify existing candidate clusters and write sparse duplicate markers.

    ``shard_basenames`` verifies a prepared subset of the co-partitioned
    shards. Every cluster it covers must be complete inside that subset, since
    a cluster missing members verifies against the wrong representative.
    """
    verification_started = time.monotonic()
    if not normalized_sources:
        raise ValueError("verify_fuzzy_dups requires at least one normalized source")
    layout = _verification_shards(
        normalized_sources=normalized_sources,
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=output_path,
        shard_basenames=shard_basenames,
    )
    shards = layout.shards
    if not shards:
        raise ValueError("verify_fuzzy_dups found no normalized Parquet shards")
    if max_output_shards < 1:
        raise ValueError("max_output_shards must be at least 1")
    num_output_shards = min(max_output_shards, len(shards))

    resources = worker_resources or ResourceConfig(cpu=2, ram="16g", disk=VERIFICATION_WORKER_SCRATCH)
    # The document store loads onto the worker pool, which must know its size,
    # so an unset worker count falls back to one worker per shard.
    if max_workers is None:
        max_workers = min(len(shards), MAX_IRIS_WORKER_REPLICAS)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    ctx_kwargs: dict[str, Any] = {
        "name": "verify-fuzzy-dups",
        "resources": resources,
        "max_workers": max_workers,
        "actor_environment": actor_environment,
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    # The document store lives in the workers' own memory, so the context must
    # own an entered pool before the table loads.
    pool_started = time.monotonic()
    ctx = ZephyrContext(**ctx_kwargs).start()
    pool_start_elapsed = time.monotonic() - pool_started
    shard_groups = [[shard] for shard in shards]
    document_partitions = {shard.file_idx: position for position, shard in enumerate(shards)}

    try:
        ctx.put(_SHARED_SHARDS_KEY, {shard.file_idx: shard for shard in shards})
        document_store = ctx.load_memory_store(
            Dataset.from_list(shard_groups).flat_map(_candidate_documents),
            name="fuzzy-verification-documents",
            hash_key=partial(_document_partition, document_partitions),
            recovery_timeout=store_config.recovery_timeout,
            shards_per_worker=store_config.shards_per_worker,
            ready_timeout=store_config.ready_timeout,
            load_concurrency=store_config.load_concurrency,
        )
        pipeline = (
            Dataset.from_list(shard_groups)
            .flat_map(_joined_cluster_members)
            .group_by(
                key=_cluster_key,
                sort_by=_cluster_sort_key,
                reducer=_make_cluster_verifier(
                    verification_params,
                    local_representative_params,
                    document_store,
                    store_config.lookup_batch_size,
                ),
                num_output_shards=num_output_shards,
            )
            .group_by(
                key=lambda record: record["file_idx"],
                sort_by=lambda record: record["id"],
                reducer=_write_verified_shard,
                num_output_shards=num_output_shards,
            )
        )
        pipeline_started = time.monotonic()
        outcome = ctx.execute(
            pipeline,
            verbose=True,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        )
        pipeline_elapsed = time.monotonic() - pipeline_started
        store_stats = document_store.stats()
    finally:
        shutdown_started = time.monotonic()
        ctx.shutdown()
        shutdown_elapsed = time.monotonic() - shutdown_started

    write_copartitioned_source_manifest(output_path=output_path, attr_dirs=layout.attr_dirs)
    total_elapsed = time.monotonic() - verification_started

    verified = sum(result["verified_duplicates"] for result in outcome.results)
    output_counters = dict(outcome.counters)
    store_workers = len({stat.actor_index for stat in store_stats})
    store_load_cpu_time = sum(stat.load_cpu_time for stat in store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/workers"] = store_workers
    output_counters[f"{_COUNTER_PREFIX}/memory_store/shards"] = len(store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/items"] = sum(stat.num_items for stat in store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/load_cpu_time"] = store_load_cpu_time
    output_counters[f"{_COUNTER_PREFIX}/memory_store/load_elapsed"] = document_store.load_elapsed
    output_counters[f"{_COUNTER_PREFIX}/memory_store/load_average_cpu_cores"] = (
        store_load_cpu_time / document_store.load_elapsed
    )
    output_counters[f"{_COUNTER_PREFIX}/memory_store/resident_bytes"] = sum(stat.resident_bytes for stat in store_stats)
    output_counters[f"{_COUNTER_PREFIX}/memory_store/peak_resident_bytes_upper_bound"] = sum(
        stat.peak_resident_bytes for stat in store_stats
    )
    output_counters[f"{_COUNTER_PREFIX}/memory_store/max_shard_load_elapsed"] = max(
        stat.load_elapsed for stat in store_stats
    )
    output_counters[f"{_COUNTER_PREFIX}/timing/pool_start_elapsed"] = pool_start_elapsed
    output_counters[f"{_COUNTER_PREFIX}/timing/pipeline_elapsed"] = pipeline_elapsed
    output_counters[f"{_COUNTER_PREFIX}/timing/pool_shutdown_elapsed"] = shutdown_elapsed
    output_counters[f"{_COUNTER_PREFIX}/timing/total_elapsed"] = total_elapsed
    logger.info(
        "Fuzzy verification wall times: pool startup %.2f seconds, memory-store load %.2f seconds, "
        "pipeline %.2f seconds, pool shutdown %.2f seconds, total %.2f seconds",
        pool_start_elapsed,
        document_store.load_elapsed,
        pipeline_elapsed,
        shutdown_elapsed,
        total_elapsed,
    )
    logger.info(
        "Verified %d fuzzy duplicates from %d candidate members across %d shards",
        verified,
        int(output_counters.get(f"{_COUNTER_PREFIX}/candidate_members", 0)),
        len(shards),
    )
    return VerifiedFuzzyDupsAttrData(
        verification=verification_params,
        local_representatives=local_representative_params,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(
                attr_dir=attr_dir,
                source_tag=layout.source_tags[source_key],
            )
            for source_key, attr_dir in layout.attr_dirs.items()
        },
        counters=output_counters,
    )


def verify_fuzzy_dups_step(
    *,
    name: str,
    normalized_steps: dict[str, StepSpec],
    minhash_steps: dict[str, StepSpec],
    candidates_step: StepSpec,
    verification_params: FuzzyVerificationParams,
    local_representative_params: LocalRepresentativeParams,
    store_config: FuzzyVerificationStoreConfig,
    max_output_shards: int,
    max_workers: int | None = None,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    actor_environment: EnvironmentConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a step that verifies one existing fuzzy-candidate artifact."""
    ordered_normalized_steps = {name: normalized_steps[name] for name in sorted(normalized_steps)}
    ordered_minhash_steps = {name: minhash_steps[name] for name in sorted(minhash_steps)}
    if ordered_normalized_steps.keys() != ordered_minhash_steps.keys():
        raise ValueError("normalized_steps and minhash_steps must have the same source names")
    return StepSpec(
        name=name,
        deps=[*ordered_normalized_steps.values(), *ordered_minhash_steps.values(), candidates_step],
        fn=lambda output_path: verify_fuzzy_dups(
            normalized_sources={
                source_name: read_artifact(step.output_path, NormalizedData)
                for source_name, step in ordered_normalized_steps.items()
            },
            minhash_sources={
                source_name: read_artifact(step.output_path, MinHashAttrData)
                for source_name, step in ordered_minhash_steps.items()
            },
            candidates=read_artifact(candidates_step.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            verification_params=verification_params,
            local_representative_params=local_representative_params,
            store_config=store_config,
            max_output_shards=max_output_shards,
            max_workers=max_workers,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
            actor_environment=actor_environment,
        ),
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
            "local_representatives": local_representative_params.model_dump(mode="json"),
            "max_output_shards": max_output_shards,
        },
        override_output_path=override_output_path,
    )
