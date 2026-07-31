# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify fuzzy-duplicate clusters against full normalized text."""

import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain, groupby
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_WORKERS_PER_JOB, ZephyrContext
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
    character_ngram_jaccard,
    prepare_verification_text,
    verify_prepared_candidate,
)

logger = logging.getLogger(__name__)

VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION = 2
PERCENT_HISTOGRAM_METRICS = frozenset({"member_containment", "jaccard", "char_jaccard", "local_char_jaccard"})
SCORE_HISTOGRAM_MAX_PERCENT = 100
UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN = 33
_COUNTER_PREFIX = "dedup/fuzzy/verification"
_SHARED_SHARDS_KEY = "verified_fuzzy_dups_shards"
_LOCAL_TOKEN_JACCARD_REJECTION = "local_token_jaccard_below_threshold"
_LOCAL_CHAR_JACCARD_REJECTION = "local_char_jaccard_below_threshold"


class RepresentativeKind(StrEnum):
    """The retained document used for a direct verification."""

    CLUSTER_CANONICAL = "cluster_canonical"
    LOCAL_REPRESENTATIVE = "local_representative"


class LocalRepresentativeParams(BaseModel):
    """Bounds and score gates for local representative verification."""

    model_config = ConfigDict(frozen=True)

    maximum_comparisons_per_document: int = Field(ge=1)
    maximum_representatives_per_cluster: int = Field(ge=1)
    maximum_local_representative_chars: int = Field(ge=1)
    maximum_local_representative_chars_per_cluster: int = Field(ge=1)
    minimum_local_token_ngram_jaccard: float = Field(ge=0, le=1)
    local_char_ngram_size: int = Field(ge=1)
    minimum_local_char_jaccard: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def comparisons_fit_representative_limit(self) -> "LocalRepresentativeParams":
        """Reject a comparison limit that cannot be reached."""
        if self.maximum_comparisons_per_document > self.maximum_representatives_per_cluster:
            raise ValueError("maximum_comparisons_per_document cannot exceed maximum_representatives_per_cluster")
        return self


REFERENCE_LOCAL_REPRESENTATIVE_PARAMS = LocalRepresentativeParams(
    maximum_comparisons_per_document=2,
    maximum_representatives_per_cluster=64,
    maximum_local_representative_chars=500_000,
    maximum_local_representative_chars_per_cluster=2_000_000,
    minimum_local_token_ngram_jaccard=0.98,
    local_char_ngram_size=13,
    minimum_local_char_jaccard=0.98,
)


class VerifiedFuzzyDupsPerSource(BaseModel):
    """Attribute output for one normalized source."""

    attr_dir: DatakitArtifactPath


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
        pa.field("dup_local_token_sequence_equal", pa.bool_()),
        pa.field("dup_local_char_jaccard", pa.float64()),
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


def _joined_cluster_members(shards: list[VerificationShard]) -> Iterator[dict[str, Any]]:
    """Join candidate attributes to normalized text and stored LSH buckets."""
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

        normalized = iter(_rows(shard.normalized_path, ["id", "text"]))
        source = next(normalized, None)
        minhash_rows = iter(_rows(shard.minhash_path, ["id", "buckets"]))
        minhash = next(minhash_rows, None)
        while candidate is not None:
            while source is not None and source["id"] < candidate["id"]:
                source = next(normalized, None)
            if source is None or source["id"] != candidate["id"]:
                raise ValueError(
                    f"{shard.candidate_path} contains ID {candidate['id']!r} "
                    f"that is absent from {shard.normalized_path}"
                )
            while minhash is not None and minhash["id"] < candidate["id"]:
                minhash = next(minhash_rows, None)
            if minhash is None or minhash["id"] != candidate["id"]:
                raise ValueError(
                    f"{shard.candidate_path} contains ID {candidate['id']!r} "
                    f"that is absent from {shard.minhash_path}"
                )
            text = source["text"] or ""
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_members", 1)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/candidate_text_chars", len(text))
            yield {
                "kind": "candidate",
                "file_idx": shard.file_idx,
                "source_key": shard.source_key,
                "source_tag": shard.source_tag,
                "id": candidate["id"],
                "dup_cluster_id": str(candidate["dup_cluster_id"]),
                "is_cluster_canonical": candidate["is_cluster_canonical"],
                "text": text,
                "buckets": tuple(sorted(set(minhash["buckets"] or []))),
            }
            candidate = next(candidates, None)
            source = next(normalized, None)
            minhash = next(minhash_rows, None)


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
    local_char_jaccard: float | None = None,
    local_token_sequence_equal: bool | None = None,
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
    if local_char_jaccard is not None:
        counters.pipeline.update_counter(
            f"{_COUNTER_PREFIX}/histogram/local_char_jaccard/{_score_bin(local_char_jaccard)}",
            1,
        )
    if local_token_sequence_equal is not None:
        counters.pipeline.update_counter(
            f"{_COUNTER_PREFIX}/local_token_sequence_equal/{str(local_token_sequence_equal).lower()}",
            1,
        )
    counters.pipeline.update_counter(
        f"{_COUNTER_PREFIX}/histogram/member_unique/"
        f"{min(result.member_unique_ngrams, UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN)}",
        1,
    )


def _local_verification_gate(
    member: PreparedVerificationText,
    representative: PreparedVerificationText,
    result: VerificationResult,
    params: LocalRepresentativeParams,
) -> tuple[bool, str, float | None, bool | None]:
    """Apply the near-equality gates used only for local representatives."""
    if not result.accepted:
        assert result.rejection is not None
        return False, result.rejection.value, None, None
    if result.jaccard < params.minimum_local_token_ngram_jaccard:
        return False, _LOCAL_TOKEN_JACCARD_REJECTION, None, None

    token_sequence_equal = member.text.casefold().split() == representative.text.casefold().split()
    if token_sequence_equal:
        return True, "accepted", None, True

    char_jaccard = character_ngram_jaccard(
        member.text,
        representative.text,
        params.local_char_ngram_size,
    )
    if char_jaccard < params.minimum_local_char_jaccard:
        return False, _LOCAL_CHAR_JACCARD_REJECTION, char_jaccard, False
    return True, "accepted", char_jaccard, False


def _peek_record_group(
    records: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None, Iterator[dict[str, Any]]]:
    """Return the first two records and an iterator that includes all records."""
    first = next(records)
    second = next(records, None)
    all_records = chain((first,), () if second is None else (second,), records)
    return first, second, all_records


def _make_cluster_verifier(
    verification_params: FuzzyVerificationParams,
    local_params: LocalRepresentativeParams,
):
    """Build a reducer that uses bounded, retained local representatives."""

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
        canonical = _RetainedRepresentative(
            id=representative["id"],
            source_key=representative["source_key"],
            prepared=prepare_verification_text(representative["text"], verification_params),
            buckets=frozenset(representative["buckets"]),
            kind=RepresentativeKind.CLUSTER_CANONICAL,
        )
        retained = [canonical]
        bucket_representatives: dict[str, list[int]] = defaultdict(list)
        local_representative_chars = 0
        cluster_size = 1
        accepted = 0
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/clusters", 1)

        def add_local_representative(member: dict[str, Any], prepared: PreparedVerificationText) -> None:
            nonlocal local_representative_chars
            if len(retained) >= local_params.maximum_representatives_per_cluster:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/cluster_limit", 1)
                return
            if prepared.chars > local_params.maximum_local_representative_chars:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/document_chars", 1)
                return
            if local_representative_chars + prepared.chars > local_params.maximum_local_representative_chars_per_cluster:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/representative_skipped/cluster_chars", 1)
                return

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

        for member_id, same_id_records in groupby(records, key=lambda record: record["id"]):
            member, repeated_member, exact_records = _peek_record_group(same_id_records)
            exact_group = member_id == canonical.id or repeated_member is not None
            if exact_group:
                expected_text = representative["text"] if member_id == canonical.id else member["text"]
                for exact_record in exact_records:
                    cluster_size += 1
                    if exact_record["is_cluster_canonical"]:
                        raise ValueError(f"Cluster {group_key[1]!r} has more than one canonical member")
                    if exact_record["text"] != expected_text:
                        raise ValueError(f"Cluster {group_key[1]!r} has different text for content ID {member_id!r}")
                    _record_document_decision(exact_record, "delegated_global_exact")
                if member_id != canonical.id:
                    add_local_representative(
                        member,
                        prepare_verification_text(member["text"], verification_params),
                    )
                continue

            cluster_size += 1
            if member["is_cluster_canonical"]:
                raise ValueError(f"Cluster {group_key[1]!r} has more than one canonical member")

            prepared_member = prepare_verification_text(member["text"], verification_params)
            member_buckets = frozenset(member["buckets"])
            comparison_count = 1
            matched_representative: _RetainedRepresentative | None = None
            matched_result: VerificationResult | None = None
            matched_local_token_sequence_equal: bool | None = None
            matched_local_char_jaccard: float | None = None
            shared_buckets = len(member_buckets & canonical.buckets)
            result = verify_prepared_candidate(prepared_member, canonical.prepared, verification_params)
            comparison_decision = result.rejection.value if result.rejection is not None else "accepted"
            _record_comparison(result, canonical.kind, comparison_decision)
            if result.accepted:
                matched_representative = canonical
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
                    (
                        local_accepted,
                        local_decision,
                        local_char_jaccard,
                        local_token_sequence_equal,
                    ) = _local_verification_gate(
                        prepared_member,
                        local.prepared,
                        local_result,
                        local_params,
                    )
                    _record_comparison(
                        local_result,
                        local.kind,
                        local_decision,
                        local_char_jaccard,
                        local_token_sequence_equal,
                    )
                    if local_accepted:
                        matched_representative = local
                        matched_result = local_result
                        matched_local_token_sequence_equal = local_token_sequence_equal
                        matched_local_char_jaccard = local_char_jaccard
                        shared_buckets = shared_counts[representative_index]
                        break

            counters.pipeline.update_counter(
                f"{_COUNTER_PREFIX}/comparisons_per_document/{comparison_count}",
                1,
            )
            if matched_representative is None:
                _record_document_decision(member, "retained_no_match")
                add_local_representative(member, prepared_member)
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
                "dup_local_token_sequence_equal": matched_local_token_sequence_equal,
                "dup_local_char_jaccard": matched_local_char_jaccard,
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
    return shards, attr_dirs


def verify_fuzzy_dups(
    *,
    normalized_sources: dict[str, NormalizedData],
    minhash_sources: dict[str, MinHashAttrData],
    candidates: FuzzyDupsAttrData,
    output_path: str,
    verification_params: FuzzyVerificationParams,
    local_representative_params: LocalRepresentativeParams,
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
        minhash_sources=minhash_sources,
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
    ctx = ZephyrContext(**ctx_kwargs)
    ctx.put(_SHARED_SHARDS_KEY, {shard.file_idx: shard for shard in shards})

    file_shards = min(max_parallelism, len(shards))
    shard_groups: list[list[VerificationShard]] = [[] for _ in range(file_shards)]
    for index, shard in enumerate(shards):
        shard_groups[index % file_shards].append(shard)

    pipeline = (
        Dataset.from_list(shard_groups)
        .flat_map(_joined_cluster_members)
        .group_by(
            key=_cluster_key,
            sort_by=lambda record: (
                0 if record["kind"] == "sentinel" or record["is_cluster_canonical"] else 1,
                record.get("id", ""),
                record["file_idx"],
            ),
            reducer=_make_cluster_verifier(verification_params, local_representative_params),
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
    logger.info(
        "Verified %d fuzzy duplicates from %d candidate members across %d shards",
        verified,
        int(outcome.counters.get(f"{_COUNTER_PREFIX}/candidate_members", 0)),
        len(shards),
    )
    return VerifiedFuzzyDupsAttrData(
        verification=verification_params,
        local_representatives=local_representative_params,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dir) for source_key, attr_dir in attr_dirs.items()
        },
        counters=dict(outcome.counters),
    )


def verify_fuzzy_dups_step(
    *,
    name: str,
    normalized_steps: dict[str, StepSpec],
    minhash_steps: dict[str, StepSpec],
    candidates_step: StepSpec,
    verification_params: FuzzyVerificationParams,
    local_representative_params: LocalRepresentativeParams,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
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
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        ),
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
            "local_representatives": local_representative_params.model_dump(mode="json"),
        },
        override_output_path=override_output_path,
    )
