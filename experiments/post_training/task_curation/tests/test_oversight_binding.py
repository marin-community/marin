# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The oversight level is derived from current-pin evidence, never hand-declared."""

from dataclasses import replace

import pytest
from finestore.rl.curation import OversightEvidence, OversightLevel, QualityResultFinalization

from experiments.post_training.task_curation.full_content_binding import bind_full_content_result
from experiments.post_training.task_curation.oversight_binding import bind_derived_oversight
from experiments.post_training.task_curation.sample_qc_binding import bind_sample_result
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.test_full_content_binding import (
    sample_receipt,
    whole_source_binding,
)
from infra.marina.apps.rl_data_catalog.server.quality import quality
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


@pytest.fixture
def content_source(tmp_path):
    source = number_source(tmp_path / "input")
    actor = actor_census_input(source.dataset.name, source.dataset.revision, source.version, 201)
    return replace(
        source,
        info=replace(
            source.info,
            actor_census_input=actor,
            within_source_dedup=deduplication_evidence(actor.dataset_id, actor.dataset_revision, source.version, 201),
        ),
    )


@pytest.fixture
def full_binding(content_source):
    return whole_source_binding(content_source, QualityResultFinalization.COMPLETE)


def test_full_binding_derives_full_with_receipt(content_source, full_binding):
    bound = bind_derived_oversight(content_source, None, full_binding)
    assert bound.info.oversight.level is OversightLevel.FULL
    assert bound.info.oversight.receipt_uri == full_binding.review_receipt.uri
    assert bound.info.oversight.reviewed == full_binding.result.reviewed


def test_sample_receipt_derives_sample_with_receipt(content_source):
    receipt = sample_receipt(content_source)
    bound = bind_derived_oversight(content_source, receipt, None)
    assert bound.info.oversight.level is OversightLevel.SAMPLE
    assert bound.info.oversight.receipt_uri == receipt.review.path


def test_no_evidence_derives_quick(content_source):
    bound = bind_derived_oversight(content_source, None, None)
    assert bound.info.oversight.level is OversightLevel.QUICK
    assert bound.info.oversight.receipt_uri is None


def test_full_binding_for_another_pin_falls_back_to_sample(content_source, full_binding):
    receipt = sample_receipt(content_source)
    stale = full_binding.model_copy(update={"source_id": "Marin:another-source"})
    bound = bind_derived_oversight(content_source, receipt, stale)
    assert bound.info.oversight.level is OversightLevel.SAMPLE
    assert bound.info.oversight.receipt_uri == receipt.review.path


@pytest.mark.parametrize(
    ("declared_level", "sample", "full_content"),
    [
        (OversightLevel.FULL, "receipt", None),
        (OversightLevel.SAMPLE, None, None),
    ],
)
def test_contradicting_declared_level_fails_ingestion(content_source, declared_level, sample, full_content):
    """A declared level that disagrees with the current-pin evidence fails ingestion."""
    receipt = sample_receipt(content_source) if sample else None
    declared = replace(
        content_source,
        info=replace(content_source.info, oversight=OversightEvidence(level=declared_level, verification="old")),
    )
    with pytest.raises(ValueError, match="contradicts the evidence-derived level"):
        bind_derived_oversight(declared, receipt, full_content)


def test_matching_declared_level_is_tolerated(content_source):
    receipt = sample_receipt(content_source)
    declared = replace(
        content_source,
        info=replace(
            content_source.info, oversight=OversightEvidence(level=OversightLevel.SAMPLE, verification="agrees")
        ),
    )
    bound = bind_derived_oversight(declared, receipt, None)
    assert bound.info.oversight.level is OversightLevel.SAMPLE


def test_derived_level_matches_table_quality(content_source, full_binding):
    """The Evidence-page level equals the main-table Quality for every level."""
    # Production order: content binding attaches the Result, then the level is derived.
    quick = bind_derived_oversight(content_source, None, None)
    sample = bind_derived_oversight(
        bind_sample_result(content_source, sample_receipt(content_source)), sample_receipt(content_source), None
    )
    full = bind_derived_oversight(bind_full_content_result(content_source, full_binding), None, full_binding)
    for bound in (quick, sample, full):
        served = bound.declaration.model_dump(mode="json")
        assert served["oversight"].upper() == quality(served, None).value, bound.info.oversight.level
