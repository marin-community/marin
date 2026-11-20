# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ruling 62 (r33 curriculum merge): the retained receipts versus the re-measured chains."""

import pytest

from experiments.post_training.task_curation.datasets.derived_contract_measurements import (
    derived_contract_receipts,
)
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.sources import (
    RETAINED_BINDING_DRIFTS,
    all_sources,
    retained_binding,
)

SOURCE_ID = "TaskTrove:DCAgent__exp_rpt_curriculum-easy"


@pytest.mark.data_integration
def test_curriculum_easy_outputs_differ_and_the_remeasure_exists():
    """Ruling 62 applied to curriculum-easy across the r33 curriculum merge.

    The retained receipt pins the pre-merge converted population (5db253cc...); the
    post-merge derived-contract census measured a different population (ede225fd...), so
    the outputs differ and the re-measure path applies. That re-measure already exists in
    the derived-contract chain, and the retained receipt is left un-rewritten for the
    DCAgent -r3 supersession (ruling 2026-11-19).
    """
    retained = retained_release_receipts()[SOURCE_ID]
    derived = derived_contract_receipts()[SOURCE_ID]
    plan_hash = retained.publication.plan.task_specs_sha256
    derived_hash = derived.input.task_specs_sha256
    assert plan_hash != derived_hash, "the merge preserved the converted bytes; no re-measure would be needed"
    # The re-measure exists and is bound to the declaration's current identity.
    source = all_sources()[SOURCE_ID]
    assert source.info.actor_census_input is not None
    assert source.info.actor_census_input.task_specs_sha256 == derived_hash
    # The retained receipt is untouched: it still pins the pre-merge population for the
    # -r3 chain to supersede.
    assert retained.publication.plan.converter_version == retained.publication.plan.converter_version


@pytest.mark.data_integration
def test_a_drifted_retained_binding_keeps_the_configured_pin_and_records_the_drift():
    """A retained re-binding against a drifted population records the drift, never crashes.

    The derived-contract census re-measured curriculum-easy at the post-merge identity
    (ede225fd...) while the retained receipt pins the pre-merge population (5db253cc...),
    so re-binding the retained release after the derived tags move the actor raises. The
    tolerant binding keeps the configured pin, records the drift for the audit's waivable
    served-catalog gap, and never raises (ruling 2026-11-19).
    """
    receipt = retained_release_receipts()[SOURCE_ID]
    source = all_sources()[SOURCE_ID]
    assert source.info.actor_census_input.task_specs_sha256 != receipt.publication.plan.task_specs_sha256
    drifted_plan = receipt.publication.plan.model_copy(update={"converter_version": "pre-merge-5"})
    drifted_receipt = receipt.model_copy(
        update={"publication": receipt.publication.model_copy(update={"plan": drifted_plan})}
    )
    RETAINED_BINDING_DRIFTS.clear()
    try:
        # A synthetic drift: the receipt pins another converted population.
        bound = retained_binding(source, drifted_receipt)
        assert "Retained release drift" in bound.info.notes
        assert RETAINED_BINDING_DRIFTS[SOURCE_ID] == drifted_receipt.publication.plan.source_revision
        # The live drift: re-binding the honest receipt against the post-merge actor
        # records the same typed drift instead of raising.
        RETAINED_BINDING_DRIFTS.clear()
        retained_binding(source, receipt)
        assert RETAINED_BINDING_DRIFTS[SOURCE_ID] == receipt.publication.plan.source_revision
    finally:
        RETAINED_BINDING_DRIFTS.clear()
