# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run and inspect fuzzy-duplicate verification on an existing Datakit sample.

The bounded inspection independently replays reducer decisions. It is a manual
data gate for this algorithm version, not a production validation API.
"""

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import groupby
from typing import Any, Protocol

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    MinHashParams,
    compute_minhash_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    VerificationResult,
    line_count_ratio,
    verify_candidate,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    DEFAULT_PIPELINE_SHARDS_PER_WORKER,
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VERIFICATION_WORKER_SCRATCH,
    FuzzyVerificationImplementation,
    FuzzyVerificationStoreConfig,
    LocalRepresentativeParams,
    VerifiedFuzzyDupsAttrData,
    verify_fuzzy_dups,
    verify_fuzzy_dups_step,
)
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import SAMPLE_PREFIX, sample_sources
from experiments.datakit.reports.dedup import dedup_report

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 64
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_INSPECTION_LIMIT = 10_000
TEXT_PREVIEW_CHARS = 500
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="64g", disk=VERIFICATION_WORKER_SCRATCH)
VERIFICATION_TASK_RESOURCES = ResourceConfig(cpu=2, ram="60g", disk=VERIFICATION_WORKER_SCRATCH)
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=False)
STORE_CONFIG = FuzzyVerificationStoreConfig(
    recovery_timeout=1_800,
    ready_timeout=1_800,
    lookup_batch_size=128,
    shards_per_worker=1,
)
VERIFIED_COLUMNS = (
    "id",
    "dup_doc",
    "dup_cluster_id",
    "dup_representative_id",
    "dup_representative_source_key",
    "dup_representative_kind",
    "dup_shared_lsh_buckets",
    "dup_comparisons",
    "dup_member_containment",
    "dup_jaccard",
    "dup_under_tokenized",
    "dup_char_jaccard",
    "dup_local_line_count_ratio",
)


class ReferenceComparisonMode(StrEnum):
    EXACT = "exact"
    REPORT = "report"


@dataclass(frozen=True)
class CandidateMember:
    """One candidate cluster member with its normalized text."""

    file_idx: int
    source_name: str
    source_key: str
    id: str
    cluster_id: str
    is_cluster_canonical: bool
    text: str
    buckets: tuple[str, ...]


class _LegacyMinHashEntry(BaseModel):
    """One entry in the combined MinHash artifact used by the 100B audit."""

    version: str
    params: MinHashParams
    source_main_dir: str
    attr_dir: str
    counters: dict[str, int | float]


class _LegacyMinHashCollection(BaseModel):
    """Combined MinHash payload used by the 100B audit."""

    inputs: list[_LegacyMinHashEntry]


class _VerifiedSourcePaths(BaseModel):
    attr_dir: DatakitArtifactPath


class _VerifiedArtifactPaths(BaseModel):
    sources: dict[str, _VerifiedSourcePaths]


class _VerifiedSource(Protocol):
    attr_dir: DatakitArtifactPath


def _rows(path: str, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(columns=columns):
            yield from batch.to_pylist()


def _join_candidate_text(
    *,
    normalized: dict[str, NormalizedData],
    minhash: dict[str, MinHashAttrData],
    candidates: FuzzyDupsAttrData,
    limit: int,
) -> list[CandidateMember]:
    """Read persisted inputs without calling the production join."""
    members: list[CandidateMember] = []
    file_idx = 0
    for source_name, source in sorted(normalized.items()):
        source_key = datakit_source_key(source.main_output_dir)
        minhash_source = minhash[source_name]
        if minhash_source.source_key != source_key:
            raise ValueError(f"MinHash source key differs for {source_name!r}")
        if minhash_source.params != candidates.params:
            raise ValueError(f"MinHash params differ for {source_name!r}")
        candidate_dir = candidates.sources[source_key].attr_dir
        normalized_paths = sorted(
            str(path) for path in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob()
        )
        for normalized_path in normalized_paths:
            candidate_path = prefix_join(candidate_dir, os.path.basename(normalized_path))
            minhash_path = prefix_join(minhash_source.attr_dir, os.path.basename(normalized_path))
            current_file_idx = file_idx
            file_idx += 1
            if not StoragePath(candidate_path).exists():
                continue

            candidate_rows = iter(_rows(candidate_path, ["id", "dup_cluster_id", "is_cluster_canonical"]))
            candidate = next(candidate_rows, None)
            if candidate is None:
                continue

            normalized_rows = iter(_rows(normalized_path, ["id", "text"]))
            normalized_row = next(normalized_rows, None)
            minhash_rows = iter(_rows(minhash_path, ["id", "buckets"]))
            minhash_row = next(minhash_rows, None)
            while candidate is not None:
                while normalized_row is not None and normalized_row["id"] < candidate["id"]:
                    normalized_row = next(normalized_rows, None)
                if normalized_row is None or normalized_row["id"] != candidate["id"]:
                    raise ValueError(f"Candidate {candidate['id']!r} is absent from {normalized_path}")
                while minhash_row is not None and minhash_row["id"] < candidate["id"]:
                    minhash_row = next(minhash_rows, None)
                if minhash_row is None or minhash_row["id"] != candidate["id"]:
                    raise ValueError(f"Candidate {candidate['id']!r} is absent from {minhash_path}")
                members.append(
                    CandidateMember(
                        file_idx=current_file_idx,
                        source_name=source_name,
                        source_key=source_key,
                        id=candidate["id"],
                        cluster_id=str(candidate["dup_cluster_id"]),
                        is_cluster_canonical=candidate["is_cluster_canonical"],
                        text=normalized_row["text"] or "",
                        buckets=tuple(sorted(set(minhash_row["buckets"] or []))),
                    )
                )
                if len(members) > limit:
                    raise ValueError(f"Inspection found more than {limit} candidate members")
                candidate = next(candidate_rows, None)
                normalized_row = next(normalized_rows, None)
                minhash_row = next(minhash_rows, None)
    return members


def _verified_rows_from_sources(
    sources: Mapping[str, _VerifiedSource],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source_key, source in sources.items():
        paths = sorted(str(path) for path in StoragePath(prefix_join(source.attr_dir, "*.parquet")).glob())
        for path in paths:
            for row in _rows(path, VERIFIED_COLUMNS):
                key = (source_key, row["id"])
                if key in rows:
                    raise ValueError(f"Verified output contains duplicate marker {key!r}")
                rows[key] = row
    return rows


def _verified_rows(artifact_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    verified = read_artifact(artifact_path, _VerifiedArtifactPaths)
    return _verified_rows_from_sources(verified.sources)


def _score_fields(result: VerificationResult) -> dict[str, Any]:
    return {
        "member_chars": result.member_chars,
        "representative_chars": result.representative_chars,
        "member_tokens": result.member_tokens,
        "representative_tokens": result.representative_tokens,
        "member_ngrams": result.member_ngrams,
        "representative_ngrams": result.representative_ngrams,
        "shared_ngrams": result.shared_ngrams,
        "member_unique_ngrams": result.member_unique_ngrams,
        "member_containment": result.member_containment,
        "jaccard": result.jaccard,
        "under_tokenized": result.under_tokenized,
        "char_jaccard": result.char_jaccard,
    }


def _add_local_representative(
    member: CandidateMember,
    retained: list[CandidateMember],
    local_representative_chars: int,
    params: LocalRepresentativeParams,
) -> tuple[int, str | None]:
    if len(retained) >= params.maximum_representatives_per_cluster:
        return local_representative_chars, "cluster_limit"
    if len(member.text) > params.maximum_local_representative_chars:
        return local_representative_chars, "document_chars"
    if len(member.text) + local_representative_chars > params.maximum_local_representative_chars_per_cluster:
        return local_representative_chars, "cluster_chars"
    retained.append(member)
    return local_representative_chars + len(member.text), None


def _assert_marker(
    *,
    marker: dict[str, Any],
    member: CandidateMember,
    representative: CandidateMember,
    result: VerificationResult,
    representative_kind: str,
    shared_lsh_buckets: int,
    comparisons: int,
    local_line_count_ratio: float | None,
) -> None:
    expected = {
        "dup_doc": True,
        "dup_cluster_id": member.cluster_id,
        "dup_representative_id": representative.id,
        "dup_representative_source_key": representative.source_key,
        "dup_representative_kind": representative_kind,
        "dup_shared_lsh_buckets": shared_lsh_buckets,
        "dup_comparisons": comparisons,
        "dup_member_containment": result.member_containment,
        "dup_jaccard": result.jaccard,
        "dup_under_tokenized": result.under_tokenized,
        "dup_char_jaccard": result.char_jaccard,
        "dup_local_line_count_ratio": local_line_count_ratio,
    }
    actual = {name: marker[name] for name in expected}
    if actual != expected:
        raise AssertionError(
            f"Verified marker differs for {(member.source_key, member.id)!r}: {actual!r} != {expected!r}"
        )


def inspect_verification(
    *,
    normalized: dict[str, NormalizedData],
    minhash: dict[str, MinHashAttrData],
    candidates: FuzzyDupsAttrData,
    verified: VerifiedFuzzyDupsAttrData,
    output_path: str,
    limit: int,
) -> dict[str, Any]:
    """Write a bounded, direct review of each candidate decision."""
    members = _join_candidate_text(normalized=normalized, minhash=minhash, candidates=candidates, limit=limit)
    by_cluster: dict[str, list[CandidateMember]] = defaultdict(list)
    for member in members:
        by_cluster[member.cluster_id].append(member)

    actual_markers = _verified_rows_from_sources(verified.sources)
    expected_marker_keys: set[tuple[str, str]] = set()
    decisions: Counter[str] = Counter()
    comparison_decisions: Counter[str] = Counter()
    cluster_reviews = []
    for cluster_id, cluster_members in sorted(by_cluster.items(), key=lambda item: (-len(item[1]), item[0])):
        canonicals = [member for member in cluster_members if member.is_cluster_canonical]
        if len(canonicals) != 1:
            raise AssertionError(f"Cluster {cluster_id!r} has {len(canonicals)} canonical members")
        # Mirror the verifier: the anchor is the longest document, and the rest
        # arrive in the shuffle's content-ID order. Replaying against the
        # connected-components canonical instead would disagree with a correct
        # run on every cluster whose canonical is not its longest member.
        stream = sorted(cluster_members, key=lambda member: (len(member.id), member.id, member.file_idx))
        anchor = max(stream, key=lambda member: (len(member.text), member.id))
        ordered = [anchor, *(member for member in stream if member is not anchor)]
        # The verifier labels the anchor by whether it is also the canonical.
        anchor_kind = "cluster_canonical" if anchor.is_cluster_canonical else "cluster_longest"
        retained = [anchor]
        local_representative_chars = 0
        member_reviews = []

        for member_id, same_id_iterator in groupby(ordered[1:], key=lambda member: member.id):
            same_id_members = list(same_id_iterator)
            if member_id == anchor.id:
                for member in same_id_members:
                    if member.text != anchor.text:
                        raise AssertionError(f"Content ID {member.id!r} has different text")
                    marker_key = (member.source_key, member.id)
                    decisions["delegated_global_exact"] += 1
                    if marker_key in actual_markers:
                        raise AssertionError(f"Global exact member {marker_key!r} has a fuzzy output marker")
                    member_reviews.append(
                        {
                            "member": {
                                **asdict(member),
                                "text": member.text[:TEXT_PREVIEW_CHARS],
                                "text_chars": len(member.text),
                            },
                            "decision": "delegated_global_exact",
                            "attempts": [],
                            "output_marker_present": False,
                        }
                    )
                continue

            member = same_id_members[0]
            for exact_member in same_id_members[1:]:
                if exact_member.text != member.text:
                    raise AssertionError(f"Content ID {exact_member.id!r} has different text")
                exact_marker_key = (exact_member.source_key, exact_member.id)
                decisions["delegated_global_exact"] += 1
                if exact_marker_key in actual_markers:
                    raise AssertionError(f"Global exact member {exact_marker_key!r} has a fuzzy output marker")
                member_reviews.append(
                    {
                        "member": {
                            **asdict(exact_member),
                            "text": exact_member.text[:TEXT_PREVIEW_CHARS],
                            "text_chars": len(exact_member.text),
                        },
                        "decision": "delegated_global_exact",
                        "attempts": [],
                        "output_marker_present": False,
                    }
                )

            marker_key = (member.source_key, member.id)
            attempts = []
            comparison_count = 1
            shared_buckets = len(set(member.buckets) & set(anchor.buckets))
            result = verify_candidate(member.text, anchor.text, verified.verification)
            comparison_decision = result.rejection.value if result.rejection is not None else "accepted"
            comparison_decisions[comparison_decision] += 1
            attempts.append(
                {
                    "representative_id": anchor.id,
                    "representative_kind": anchor_kind,
                    "shared_lsh_buckets": shared_buckets,
                    "decision": comparison_decision,
                    "scores": _score_fields(result),
                }
            )
            matched = anchor if result.accepted else None
            matched_result = result if result.accepted else None
            matched_kind = anchor_kind
            matched_local_line_count_ratio = None

            if matched is None:
                nominees = []
                member_buckets = set(member.buckets)
                for retained_index, local in enumerate(retained[1:], start=1):
                    shared = len(member_buckets & set(local.buckets))
                    if shared:
                        nominees.append((shared, retained_index))
                nominees.sort(key=lambda item: (-item[0], item[1]))
                local_limit = verified.local_representatives.maximum_comparisons_per_document - 1
                for local_shared, retained_index in nominees[:local_limit]:
                    local = retained[retained_index]
                    comparison_count += 1
                    local_result = verify_candidate(member.text, local.text, verified.verification)
                    if not local_result.accepted:
                        assert local_result.rejection is not None
                        local_accepted = False
                        local_decision = local_result.rejection.value
                        local_line_count_ratio = None
                    elif member.text.casefold().split() != local.text.casefold().split():
                        local_accepted = False
                        local_decision = "local_token_sequence_differs"
                        local_line_count_ratio = None
                    else:
                        local_line_count_ratio = line_count_ratio(member.text, local.text)
                        local_accepted = (
                            local_line_count_ratio >= verified.local_representatives.minimum_local_line_count_ratio
                        )
                        local_decision = "accepted" if local_accepted else "local_line_count_ratio_below_threshold"
                    comparison_decisions[local_decision] += 1
                    attempts.append(
                        {
                            "representative_id": local.id,
                            "representative_kind": "local_representative",
                            "shared_lsh_buckets": local_shared,
                            "decision": local_decision,
                            "scores": {
                                **_score_fields(local_result),
                                "local_line_count_ratio": local_line_count_ratio,
                            },
                        }
                    )
                    if local_accepted:
                        matched = local
                        matched_result = local_result
                        matched_kind = "local_representative"
                        matched_local_line_count_ratio = local_line_count_ratio
                        shared_buckets = local_shared
                        break

            if matched is None:
                decisions["retained_no_match"] += 1
                local_representative_chars, representative_skip = _add_local_representative(
                    member,
                    retained,
                    local_representative_chars,
                    verified.local_representatives,
                )
                if marker_key in actual_markers:
                    raise AssertionError(f"Retained member {marker_key!r} has an output marker")
            else:
                decisions["accepted"] += 1
                expected_marker_keys.add(marker_key)
                marker = actual_markers.get(marker_key)
                if marker is None:
                    raise AssertionError(f"Accepted member {marker_key!r} has no output marker")
                assert matched_result is not None
                _assert_marker(
                    marker=marker,
                    member=member,
                    representative=matched,
                    result=matched_result,
                    representative_kind=matched_kind,
                    shared_lsh_buckets=shared_buckets,
                    comparisons=comparison_count,
                    local_line_count_ratio=matched_local_line_count_ratio,
                )
                representative_skip = None

            member_reviews.append(
                {
                    "member": {
                        **asdict(member),
                        "text": member.text[:TEXT_PREVIEW_CHARS],
                        "text_chars": len(member.text),
                    },
                    "decision": "accepted" if matched is not None else "retained_no_match",
                    "attempts": attempts,
                    "representative_skip": representative_skip,
                    "output_marker_present": marker_key in actual_markers,
                }
            )

        cluster_reviews.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(ordered),
                "representative": {
                    **asdict(anchor),
                    "text": anchor.text[:TEXT_PREVIEW_CHARS],
                    "text_chars": len(anchor.text),
                },
                "retained_representatives": len(retained),
                "members": member_reviews,
            }
        )

    if set(actual_markers) != expected_marker_keys:
        unexpected = sorted(set(actual_markers) - expected_marker_keys)
        missing = sorted(expected_marker_keys - set(actual_markers))
        raise AssertionError(f"Verified marker set differs: unexpected={unexpected!r}, missing={missing!r}")

    review = {
        "verification_params": verified.verification.model_dump(mode="json"),
        "local_representative_params": verified.local_representatives.model_dump(mode="json"),
        "counts": {
            "candidate_members": len(members),
            "clusters": len(by_cluster),
            "comparisons": sum(comparison_decisions.values()),
            "verified_duplicates": len(expected_marker_keys),
            "max_cluster_size": max((len(cluster) for cluster in by_cluster.values()), default=0),
        },
        "decisions": dict(sorted(decisions.items())),
        "comparison_decisions": dict(sorted(comparison_decisions.items())),
        "clusters": cluster_reviews,
    }
    StoragePath(output_path).write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    return review


def _parse_source_names(value: str) -> list[str] | None:
    if value == "all":
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--sources must be 'all' or a comma-separated list")
    return names


def _read_candidate_artifact(path: str) -> FuzzyDupsAttrData:
    candidates = read_artifact(path, FuzzyDupsAttrData)
    sources = {}
    for source_path, source in candidates.sources.items():
        source_key = datakit_source_key(source_path)
        if source_key in sources:
            raise ValueError(f"Candidate source paths normalize to the same key {source_key!r}")
        sources[source_key] = source
    return candidates.model_copy(update={"sources": sources})


def _read_minhash_collection(
    path: str,
    normalized: dict[str, NormalizedData],
) -> dict[str, MinHashAttrData]:
    collection = read_artifact(path, _LegacyMinHashCollection)
    by_source_key = {}
    for entry in collection.inputs:
        source_key = datakit_source_key(entry.source_main_dir)
        if source_key in by_source_key:
            raise ValueError(f"MinHash collection contains duplicate source key {source_key!r}")
        by_source_key[source_key] = MinHashAttrData(
            version=entry.version,
            params=entry.params,
            source_key=source_key,
            attr_dir=entry.attr_dir,
            counters=entry.counters,
        )

    expected = {datakit_source_key(source.main_output_dir) for source in normalized.values()}
    if by_source_key.keys() != expected:
        raise ValueError(
            "MinHash collection and normalized source sets differ: "
            f"missing={sorted(expected - by_source_key.keys())!r}, "
            f"extra={sorted(by_source_key.keys() - expected)!r}"
        )
    return {
        source_name: by_source_key[datakit_source_key(source.main_output_dir)]
        for source_name, source in normalized.items()
    }


def _verify_existing_artifacts(
    *,
    output_path: str,
    normalized_steps: dict[str, StepSpec],
    minhash_collection_path: str,
    candidate_path: str,
    verification_params: FuzzyVerificationParams,
    max_workers: int,
) -> VerifiedFuzzyDupsAttrData:
    normalized = {
        source_name: read_artifact(step.output_path, NormalizedData) for source_name, step in normalized_steps.items()
    }
    return verify_fuzzy_dups(
        normalized_sources=normalized,
        minhash_sources=_read_minhash_collection(minhash_collection_path, normalized),
        candidates=_read_candidate_artifact(candidate_path),
        output_path=output_path,
        verification_params=verification_params,
        local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
        store_config=STORE_CONFIG,
        implementation=FuzzyVerificationImplementation.EXACT,
        max_workers=max_workers,
        worker_resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        map_task_resources=VERIFICATION_TASK_RESOURCES,
        reduce_task_resources=VERIFICATION_TASK_RESOURCES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--candidate-artifact", help="Existing FuzzyDupsAttrData artifact to verify")
    parser.add_argument("--minhash-collection", help="Combined MinHash artifact paired with --candidate-artifact")
    parser.add_argument(
        "--reference-verified-prefix",
        help="Optional verified artifact whose complete persisted output is compared",
    )
    parser.add_argument(
        "--expected-reference-markers",
        type=int,
        help="Optional nonnegative marker count required from --reference-verified-prefix",
    )
    parser.add_argument(
        "--reference-comparison-mode",
        type=ReferenceComparisonMode,
        choices=ReferenceComparisonMode,
        default=ReferenceComparisonMode.EXACT,
        help="Fail on reference differences or report their counts for an intentional algorithm change",
    )
    parser.add_argument("--sources", default="all")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--inspection-limit",
        type=int,
        default=DEFAULT_INSPECTION_LIMIT,
        help="Maximum candidate members for direct text inspection; 0 skips inspection",
    )
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    if args.inspection_limit < 0:
        raise ValueError("--inspection-limit must be nonnegative")
    if args.expected_reference_markers is not None and args.expected_reference_markers < 0:
        raise ValueError("--expected-reference-markers must be nonnegative")
    if args.expected_reference_markers is not None and not args.reference_verified_prefix:
        raise ValueError("--expected-reference-markers requires --reference-verified-prefix")
    if args.reference_comparison_mode is ReferenceComparisonMode.REPORT and not args.reference_verified_prefix:
        raise ValueError("--reference-comparison-mode=report requires --reference-verified-prefix")
    if bool(args.candidate_artifact) != bool(args.minhash_collection):
        raise ValueError("--candidate-artifact and --minhash-collection must be provided together")

    configure_logging(logging.INFO)
    output_prefix = args.output_prefix
    normalized_steps = sample_sources(args.sample_prefix, _parse_source_names(args.sources))
    verification_params = FuzzyVerificationParams()
    if args.candidate_artifact:
        candidate_path = args.candidate_artifact
        minhash_collection_path = args.minhash_collection
        minhash_steps = None
        verified_step = StepSpec(
            name="datakit/fuzzy_verification_testbed/verified_existing",
            deps=list(normalized_steps.values()),
            hash_attrs={
                "candidates": candidate_path,
                "minhash_collection": minhash_collection_path,
                "verification": verification_params.model_dump(mode="json"),
                "local_representatives": REFERENCE_LOCAL_REPRESENTATIVE_PARAMS.model_dump(mode="json"),
                "pipeline_shards_per_worker": DEFAULT_PIPELINE_SHARDS_PER_WORKER,
            },
            fn=lambda verified_output_path: _verify_existing_artifacts(
                output_path=verified_output_path,
                normalized_steps=normalized_steps,
                minhash_collection_path=minhash_collection_path,
                candidate_path=candidate_path,
                verification_params=verification_params,
                max_workers=args.max_workers,
            ),
            override_output_path=prefix_join(output_prefix, "verified"),
        )
        report_deps = [verified_step]
    else:
        minhash_steps = {
            source_name: compute_minhash_attrs_step(
                name=f"datakit/fuzzy_verification_testbed/minhash/{source_name}",
                normalize=normalize_step,
                worker_resources=WORKER_RESOURCES,
                max_workers=args.max_workers,
                override_output_path=prefix_join(output_prefix, f"minhash/{source_name}"),
            )
            for source_name, normalize_step in normalized_steps.items()
        }
        candidates_step = compute_fuzzy_dups_attrs_step(
            name="datakit/fuzzy_verification_testbed/candidates",
            minhash_steps=list(minhash_steps.values()),
            max_parallelism=args.max_workers,
            worker_resources=WORKER_RESOURCES,
            override_output_path=prefix_join(output_prefix, "candidates"),
        )
        candidate_path = candidates_step.output_path
        verified_step = verify_fuzzy_dups_step(
            name="datakit/fuzzy_verification_testbed/verified",
            normalized_steps=normalized_steps,
            minhash_steps=minhash_steps,
            candidates_step=candidates_step,
            verification_params=verification_params,
            local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            store_config=STORE_CONFIG,
            implementation=FuzzyVerificationImplementation.EXACT,
            max_workers=args.max_workers,
            worker_resources=WORKER_RESOURCES,
            coordinator_resources=COORDINATOR_RESOURCES,
            map_task_resources=VERIFICATION_TASK_RESOURCES,
            reduce_task_resources=VERIFICATION_TASK_RESOURCES,
            override_output_path=prefix_join(output_prefix, "verified"),
        )
        report_deps = [candidates_step, verified_step]
    report_step = StepSpec(
        name="datakit/fuzzy_verification_testbed/report",
        deps=report_deps,
        hash_attrs={"v": 2},
        fn=lambda output_path: dedup_report(
            output_path,
            _read_candidate_artifact(candidate_path),
            read_artifact(verified_step.output_path, VerifiedFuzzyDupsAttrData),
        ),
        override_output_path=prefix_join(output_prefix, "report"),
    )

    StepRunner().run([report_step], max_concurrent=args.max_concurrent)

    normalized = {
        source_name: read_artifact(step.output_path, NormalizedData) for source_name, step in normalized_steps.items()
    }
    candidates = _read_candidate_artifact(candidate_path)
    verified = read_artifact(verified_step.output_path, VerifiedFuzzyDupsAttrData)
    if args.reference_verified_prefix:
        actual_markers = _verified_rows(verified_step.output_path)
        reference_markers = _verified_rows(args.reference_verified_prefix)
        if args.expected_reference_markers is not None and len(reference_markers) != args.expected_reference_markers:
            raise AssertionError(
                f"Reference contains {len(reference_markers)} markers; " f"expected {args.expected_reference_markers}"
            )
        if actual_markers != reference_markers:
            unexpected = sorted(actual_markers.keys() - reference_markers.keys())
            missing = sorted(reference_markers.keys() - actual_markers.keys())
            changed = sorted(
                key
                for key in actual_markers.keys() & reference_markers.keys()
                if actual_markers[key] != reference_markers[key]
            )
            comparison_summary = (
                "Verified marker set differs from reference: "
                f"unexpected={len(unexpected)} sample={unexpected[:20]!r}, "
                f"missing={len(missing)} sample={missing[:20]!r}, "
                f"changed={len(changed)} sample={changed[:20]!r}"
            )
            if args.reference_comparison_mode is ReferenceComparisonMode.EXACT:
                raise AssertionError(comparison_summary)
            logger.warning(comparison_summary)
        else:
            logger.info("Verified output matches %d reference markers", len(actual_markers))

    if args.inspection_limit:
        if minhash_steps is None:
            minhash = _read_minhash_collection(minhash_collection_path, normalized)
        else:
            minhash = {
                source_name: read_artifact(step.output_path, MinHashAttrData)
                for source_name, step in minhash_steps.items()
            }
        review = inspect_verification(
            normalized=normalized,
            minhash=minhash,
            candidates=candidates,
            verified=verified,
            output_path=prefix_join(output_prefix, "inspection.json"),
            limit=args.inspection_limit,
        )
        logger.info("Verification testbed passed: %s", json.dumps(review["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
