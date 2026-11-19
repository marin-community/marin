# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured exact-contract ledgers for local source fixtures."""

from finestore.rl.curation import WithinSourceDedupEvidence


def deduplication_evidence(dataset: str, revision: str, version: str, count: int = 1) -> WithinSourceDedupEvidence:
    return WithinSourceDedupEvidence(
        raw_count=count,
        unique_count=count,
        within_source_duplicates=0,
        source_dataset_id=dataset,
        source_revision=revision,
        converter_revision="fixture-converter",
        converter_version=version,
        drop_manifest_uri="memory://fixture/within-source-dedup/drops.parquet",
        evidence_url="memory://fixture/within-source-dedup/measurement.json",
    )
