# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pre-store fuzzy verifier on fixed 100B artifacts for an A-B control."""

import argparse
import logging
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    verify_fuzzy_dups,
)
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.execution import PoolMode, ZephyrContext

from experiments.datakit.reference_pipeline import SAMPLE_PREFIX, sample_sources

logger = logging.getLogger(__name__)

SHARED_POOL_NAME = "fuzzy-verification-testbed"
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="64g", disk="8g")
VERIFICATION_TASK_RESOURCES = ResourceConfig(cpu=2, ram="60g", disk="8g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=False)
VERIFIED_COLUMNS = [
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
    "dup_local_token_sequence_equal",
    "dup_local_char_jaccard",
]
EXPECTED_REFERENCE_MARKERS = 27_203


class _LegacyMinHashEntry(BaseModel):
    version: str
    params: MinHashParams
    source_main_dir: str
    attr_dir: str
    counters: dict[str, int | float]


class _LegacyMinHashCollection(BaseModel):
    inputs: list[_LegacyMinHashEntry]


class _VerifiedSourcePaths(BaseModel):
    attr_dir: DatakitArtifactPath


class _VerifiedArtifactPaths(BaseModel):
    sources: dict[str, _VerifiedSourcePaths]


def _rows(path: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(columns=columns):
            yield from batch.to_pylist()


def _candidate_artifact(path: str) -> FuzzyDupsAttrData:
    candidates = read_artifact(path, FuzzyDupsAttrData)
    sources = {}
    for source_path, source in candidates.sources.items():
        source_key = datakit_source_key(source_path)
        if source_key in sources:
            raise ValueError(f"Candidate source paths normalize to the same key {source_key!r}")
        sources[source_key] = source
    return candidates.model_copy(update={"sources": sources})


def _minhash_sources(path: str, normalized: dict[str, NormalizedData]) -> dict[str, MinHashAttrData]:
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


def _verified_rows(artifact_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    verified = read_artifact(artifact_path, _VerifiedArtifactPaths)
    rows = {}
    for source_key, source in verified.sources.items():
        paths = sorted(str(path) for path in StoragePath(prefix_join(source.attr_dir, "*.parquet")).glob())
        for path in paths:
            for row in _rows(path, VERIFIED_COLUMNS):
                key = (source_key, row["id"])
                if key in rows:
                    raise ValueError(f"Verified output contains duplicate marker {key!r}")
                rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--minhash-collection", required=True)
    parser.add_argument("--reference-verified-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    configure_logging(logging.INFO)
    output_path = prefix_join(args.output_prefix.rstrip("/"), "verified")
    if not args.compare_only:
        normalized_steps = sample_sources(args.sample_prefix, None)
        normalized = {
            source_name: read_artifact(step.output_path, NormalizedData)
            for source_name, step in normalized_steps.items()
        }
        with ZephyrContext(
            mode=PoolMode.HOST,
            pool_name=SHARED_POOL_NAME,
            max_workers=args.max_workers,
            resources=WORKER_RESOURCES,
            coordinator_resources=COORDINATOR_RESOURCES,
        ):
            verified = verify_fuzzy_dups(
                normalized_sources=normalized,
                minhash_sources=_minhash_sources(args.minhash_collection, normalized),
                candidates=_candidate_artifact(args.candidate_artifact),
                output_path=output_path,
                verification_params=FuzzyVerificationParams(),
                local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
                max_parallelism=args.max_workers,
                worker_resources=WORKER_RESOURCES,
                map_task_resources=VERIFICATION_TASK_RESOURCES,
                reduce_task_resources=VERIFICATION_TASK_RESOURCES,
            )
        write_artifact(verified, output_path)

    actual_rows = _verified_rows(output_path)
    reference_rows = _verified_rows(args.reference_verified_prefix)
    if len(reference_rows) != EXPECTED_REFERENCE_MARKERS:
        raise AssertionError(f"Reference contains {len(reference_rows)} markers; expected {EXPECTED_REFERENCE_MARKERS}")
    if actual_rows != reference_rows:
        unexpected = sorted(actual_rows.keys() - reference_rows.keys())
        missing = sorted(reference_rows.keys() - actual_rows.keys())
        changed = sorted(
            key for key in actual_rows.keys() & reference_rows.keys() if actual_rows[key] != reference_rows[key]
        )
        raise AssertionError(
            "Verified marker set differs from reference: "
            f"unexpected={unexpected[:20]!r}, missing={missing[:20]!r}, changed={changed[:20]!r}"
        )
    logger.info("Verified output matches %d reference markers", len(actual_rows))


if __name__ == "__main__":
    main()
