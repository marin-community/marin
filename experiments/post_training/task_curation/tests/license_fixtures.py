# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary license evidence for local synthetic test datasets."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt

from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource
from infra.marina.apps.rl_data_catalog.server import licensing


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


@pytest.fixture(autouse=True)
def isolated_license_bans(monkeypatch, request):
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    monkeypatch.setattr(licensing, "LICENSE_BANS", '{"version":"fixture","bans":[]}')
    licensing.license_bans.cache_clear()
    yield
    licensing.license_bans.cache_clear()
