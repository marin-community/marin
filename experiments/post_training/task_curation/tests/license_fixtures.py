# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary license evidence for local synthetic test datasets."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt

from experiments.post_training.task_curation.license_targets import license_evidence
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource


def openmdw_license_text() -> bytes:
    """The official OpenMDW 1.1 text, exactly as published at the pinned OpenMDW commit."""
    return (
        Path(__file__).with_name("LICENSE.OpenMDW-1.1").read_bytes()
        if (Path(__file__).with_name("LICENSE.OpenMDW-1.1")).exists()
        else (
            b"OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)\n\n"
            b"By exercising rights granted to you under this agreement, you accept and agree\n"
            b"to be bound by its terms.\n"
        )
    )


def fixture_license(source_id: str, dataset: str, revision: str) -> LicenseReceipt:
    return LicenseReceipt(
        resolver_version=LICENSE_RESOLVER_VERSION,
        source_id=source_id,
        dataset_id=dataset,
        revision=revision,
        resolved_at=datetime(2026, 11, 15, tzinfo=UTC).isoformat(),
        evidence=[license_evidence("https://example.test/LICENSE", b"SPDX-License-Identifier: MIT\n", "license_file")],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


def with_license[ConfigT: CatalogConfig](source: RlDataSource[ConfigT]) -> RlDataSource[ConfigT]:
    dataset = source.dataset
    assert dataset is not None and dataset.revision is not None
    return replace(
        source,
        info=replace(source.info, license_receipt=fixture_license(source.info.id, dataset.name, dataset.revision)),
    )
