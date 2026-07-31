# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run and inspect fuzzy-duplicate verification on an existing Datakit sample."""

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs_step
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    VerificationResult,
    verify_candidate,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    verify_fuzzy_dups_step,
)
from rigging.filesystem import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.execution import PoolMode, ZephyrContext

from experiments.datakit.reference_pipeline import SAMPLE_PREFIX, sample_sources
from experiments.datakit.reports.dedup import dedup_report

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 64
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_INSPECTION_LIMIT = 10_000
TEXT_PREVIEW_CHARS = 500
SHARED_POOL_NAME = "fuzzy-verification-testbed"
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=False)
VERIFIED_COLUMNS = [
    "id",
    "dup_doc",
    "dup_cluster_id",
    "dup_representative_id",
    "dup_representative_source_key",
    "dup_member_containment",
    "dup_jaccard",
    "dup_under_tokenized",
    "dup_char_jaccard",
]


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


def _rows(path: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(columns=columns):
            yield from batch.to_pylist()


def _join_candidate_text(
    *,
    normalized: dict[str, NormalizedData],
    candidates: FuzzyDupsAttrData,
    limit: int,
) -> list[CandidateMember]:
    """Read persisted inputs without calling the production join."""
    members: list[CandidateMember] = []
    file_idx = 0
    for source_name, source in sorted(normalized.items()):
        source_key = datakit_source_key(source.main_output_dir)
        candidate_dir = candidates.sources[source_key].attr_dir
        normalized_paths = sorted(
            str(path) for path in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob()
        )
        for normalized_path in normalized_paths:
            candidate_path = prefix_join(candidate_dir, os.path.basename(normalized_path))
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
            while candidate is not None:
                while normalized_row is not None and normalized_row["id"] < candidate["id"]:
                    normalized_row = next(normalized_rows, None)
                if normalized_row is None or normalized_row["id"] != candidate["id"]:
                    raise ValueError(f"Candidate {candidate['id']!r} is absent from {normalized_path}")
                members.append(
                    CandidateMember(
                        file_idx=current_file_idx,
                        source_name=source_name,
                        source_key=source_key,
                        id=candidate["id"],
                        cluster_id=str(candidate["dup_cluster_id"]),
                        is_cluster_canonical=candidate["is_cluster_canonical"],
                        text=normalized_row["text"] or "",
                    )
                )
                if len(members) > limit:
                    raise ValueError(f"Inspection found more than {limit} candidate members")
                candidate = next(candidate_rows, None)
                normalized_row = next(normalized_rows, None)
    return members


def _verified_rows(verified: VerifiedFuzzyDupsAttrData) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source_key, source in verified.sources.items():
        paths = sorted(str(path) for path in StoragePath(prefix_join(source.attr_dir, "*.parquet")).glob())
        for path in paths:
            for row in _rows(path, VERIFIED_COLUMNS):
                key = (source_key, row["id"])
                if key in rows:
                    raise ValueError(f"Verified output contains duplicate marker {key!r}")
                rows[key] = row
    return rows


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


def _assert_marker(
    *,
    marker: dict[str, Any],
    member: CandidateMember,
    representative: CandidateMember,
    result: VerificationResult,
) -> None:
    expected = {
        "dup_doc": True,
        "dup_cluster_id": member.cluster_id,
        "dup_representative_id": representative.id,
        "dup_representative_source_key": representative.source_key,
        "dup_member_containment": result.member_containment,
        "dup_jaccard": result.jaccard,
        "dup_under_tokenized": result.under_tokenized,
        "dup_char_jaccard": result.char_jaccard,
    }
    actual = {name: marker[name] for name in expected}
    if actual != expected:
        raise AssertionError(
            f"Verified marker differs for {(member.source_key, member.id)!r}: {actual!r} != {expected!r}"
        )


def inspect_verification(
    *,
    normalized: dict[str, NormalizedData],
    candidates: FuzzyDupsAttrData,
    verified: VerifiedFuzzyDupsAttrData,
    output_path: str,
    limit: int,
) -> dict[str, Any]:
    """Write a bounded, direct review of each candidate decision."""
    members = _join_candidate_text(normalized=normalized, candidates=candidates, limit=limit)
    by_cluster: dict[str, list[CandidateMember]] = defaultdict(list)
    for member in members:
        by_cluster[member.cluster_id].append(member)

    actual_markers = _verified_rows(verified)
    expected_marker_keys: set[tuple[str, str]] = set()
    decisions: Counter[str] = Counter()
    cluster_reviews = []
    for cluster_id, cluster_members in sorted(by_cluster.items(), key=lambda item: (-len(item[1]), item[0])):
        canonicals = [member for member in cluster_members if member.is_cluster_canonical]
        if len(canonicals) != 1:
            raise AssertionError(f"Cluster {cluster_id!r} has {len(canonicals)} canonical members")
        representative = canonicals[0]
        ordered = [
            representative,
            *sorted(
                (member for member in cluster_members if not member.is_cluster_canonical),
                key=lambda member: (member.file_idx, member.id),
            ),
        ]
        comparisons = []
        for member in ordered[1:]:
            marker_key = (member.source_key, member.id)
            if member.id == representative.id:
                if member.text != representative.text:
                    raise AssertionError(f"Content ID {member.id!r} has different text")
                decisions["delegated_global_exact"] += 1
                if marker_key in actual_markers:
                    raise AssertionError(f"Global exact member {marker_key!r} has a fuzzy output marker")
                comparisons.append(
                    {
                        "member": {
                            **asdict(member),
                            "text": member.text[:TEXT_PREVIEW_CHARS],
                            "text_chars": len(member.text),
                        },
                        "accepted": False,
                        "rejection": "delegated_global_exact",
                        "scores": None,
                        "output_marker_present": False,
                    }
                )
                continue

            result = verify_candidate(member.text, representative.text, verified.verification)
            decision = result.rejection.value if result.rejection is not None else "accepted"
            decisions[decision] += 1
            if result.accepted:
                expected_marker_keys.add(marker_key)
                marker = actual_markers.get(marker_key)
                if marker is None:
                    raise AssertionError(f"Accepted member {marker_key!r} has no output marker")
                _assert_marker(
                    marker=marker,
                    member=member,
                    representative=representative,
                    result=result,
                )
            elif marker_key in actual_markers:
                raise AssertionError(f"Rejected member {marker_key!r} has an output marker")

            comparisons.append(
                {
                    "member": {
                        **asdict(member),
                        "text": member.text[:TEXT_PREVIEW_CHARS],
                        "text_chars": len(member.text),
                    },
                    "accepted": result.accepted,
                    "rejection": result.rejection.value if result.rejection is not None else None,
                    "scores": _score_fields(result),
                    "output_marker_present": marker_key in actual_markers,
                }
            )

        cluster_reviews.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(ordered),
                "representative": {
                    **asdict(representative),
                    "text": representative.text[:TEXT_PREVIEW_CHARS],
                    "text_chars": len(representative.text),
                },
                "comparisons": comparisons,
            }
        )

    if set(actual_markers) != expected_marker_keys:
        unexpected = sorted(set(actual_markers) - expected_marker_keys)
        missing = sorted(expected_marker_keys - set(actual_markers))
        raise AssertionError(f"Verified marker set differs: unexpected={unexpected!r}, missing={missing!r}")

    review = {
        "params": verified.verification.model_dump(mode="json"),
        "counts": {
            "candidate_members": len(members),
            "clusters": len(by_cluster),
            "comparisons": sum(decisions.values()),
            "verified_duplicates": len(expected_marker_keys),
            "max_cluster_size": max((len(cluster) for cluster in by_cluster.values()), default=0),
        },
        "decisions": dict(sorted(decisions.items())),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--candidate-prefix",
        help="Optional shared prefix for MinHash and candidate artifacts; defaults to --output-prefix",
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
    parser.add_argument(
        "--reference-verified-prefix",
        help="Optional verified artifact whose complete output marker set must match",
    )
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    if args.inspection_limit < 0:
        raise ValueError("--inspection-limit must be nonnegative")

    configure_logging(logging.INFO)
    output_prefix = args.output_prefix.rstrip("/")
    candidate_prefix = (args.candidate_prefix or output_prefix).rstrip("/")
    normalized_steps = sample_sources(args.sample_prefix, _parse_source_names(args.sources))
    minhash_steps = [
        compute_minhash_attrs_step(
            name=f"datakit/fuzzy_verification_testbed/minhash/{source_name}",
            normalize=normalize_step,
            worker_resources=WORKER_RESOURCES,
            max_workers=args.max_workers,
            override_output_path=prefix_join(candidate_prefix, f"minhash/{source_name}"),
        )
        for source_name, normalize_step in normalized_steps.items()
    ]
    candidates_step = compute_fuzzy_dups_attrs_step(
        name="datakit/fuzzy_verification_testbed/candidates",
        minhash_steps=minhash_steps,
        max_parallelism=args.max_workers,
        worker_resources=WORKER_RESOURCES,
        override_output_path=prefix_join(candidate_prefix, "candidates"),
    )
    verification_params = FuzzyVerificationParams()
    verified_step = verify_fuzzy_dups_step(
        name="datakit/fuzzy_verification_testbed/verified",
        normalized_steps=normalized_steps,
        candidates_step=candidates_step,
        verification_params=verification_params,
        max_parallelism=args.max_workers,
        worker_resources=WORKER_RESOURCES,
        override_output_path=prefix_join(output_prefix, "verified"),
    )
    report_step = StepSpec(
        name="datakit/fuzzy_verification_testbed/report",
        deps=[candidates_step, verified_step],
        hash_attrs={"v": 1},
        fn=lambda output_path: dedup_report(
            output_path,
            read_artifact(candidates_step.output_path, FuzzyDupsAttrData),
            read_artifact(verified_step.output_path, VerifiedFuzzyDupsAttrData),
        ),
        override_output_path=prefix_join(output_prefix, "report"),
    )

    with ZephyrContext(
        mode=PoolMode.HOST,
        pool_name=SHARED_POOL_NAME,
        max_workers=args.max_workers,
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
    ):
        StepRunner().run([report_step], max_concurrent=args.max_concurrent)

    normalized = {
        source_name: read_artifact(step.output_path, NormalizedData) for source_name, step in normalized_steps.items()
    }
    verified = read_artifact(verified_step.output_path, VerifiedFuzzyDupsAttrData)
    if args.reference_verified_prefix:
        reference = read_artifact(args.reference_verified_prefix, VerifiedFuzzyDupsAttrData)
        actual_markers = _verified_rows(verified)
        reference_markers = _verified_rows(reference)
        if actual_markers != reference_markers:
            unexpected = sorted(actual_markers.keys() - reference_markers.keys())
            missing = sorted(reference_markers.keys() - actual_markers.keys())
            changed = sorted(
                key
                for key in actual_markers.keys() & reference_markers.keys()
                if actual_markers[key] != reference_markers[key]
            )
            raise AssertionError(
                "Verified marker set differs from reference: "
                f"unexpected={unexpected[:20]!r}, missing={missing[:20]!r}, changed={changed[:20]!r}"
            )
        logger.info("Verified output matches %d reference markers", len(actual_markers))

    if args.inspection_limit:
        review = inspect_verification(
            normalized=normalized,
            candidates=read_artifact(candidates_step.output_path, FuzzyDupsAttrData),
            verified=verified,
            output_path=prefix_join(output_prefix, "inspection.json"),
            limit=args.inspection_limit,
        )
        logger.info("Verification testbed passed: %s", json.dumps(review["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
