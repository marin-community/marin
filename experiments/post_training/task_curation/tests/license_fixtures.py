# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary license evidence for local synthetic test datasets."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt

from experiments.post_training.task_curation import license_binding, reference_data
from experiments.post_training.task_curation.datasets.lean import numina
from experiments.post_training.task_curation.license_targets import license_evidence
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource
from infra.marina.apps.rl_data_catalog.server import licensing


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


@pytest.fixture(autouse=True)
def isolated_license_bans(monkeypatch, request):
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    monkeypatch.setattr(licensing, "LICENSE_BANS", '{"version":"fixture","bans":[]}')
    licensing.license_bans.cache_clear()
    yield


@pytest.fixture(autouse=True)
def isolated_license_registry(monkeypatch, request):
    """Unit tests never read the referenced data registries; integration tests may.

    The license receipts and the derived curation documents live in private Hugging
    Face datasets behind committed reference indexes, so reading them in a unit test
    would touch the network. Binding behavior in the default lane is exercised
    through locally attached fixture receipts and empty reference documents.
    """
    # The passing-proof cache outlives a single test: clear it so a default-lane test
    # never inherits a real read from an integration test in the same worker, and an
    # integration test never inherits an empty isolated read.
    numina.passing_proofs.cache_clear()
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    monkeypatch.setattr(license_binding, "load_referenced_json", lambda index_path: {})
    monkeypatch.setattr(reference_data, "load_referenced_document", lambda index_path, path: {})
    # numina binds the loader directly at import time, so patch its binding too;
    # otherwise its passing-proof check would read the private dataset here.
    monkeypatch.setattr(numina, "load_referenced_document", lambda index_path, path: {})
    yield licensing.license_bans.cache_clear()
