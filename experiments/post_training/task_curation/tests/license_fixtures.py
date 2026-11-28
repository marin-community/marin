# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary license evidence for local synthetic test datasets."""

from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt

from experiments.post_training.task_curation.license_targets import license_evidence


def openmdw_license_text() -> bytes:
    """The official OpenMDW 1.1 text, exactly as published at the pinned OpenMDW commit."""
    return Path(__file__).with_name("LICENSE.OpenMDW-1.1").read_bytes()


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
