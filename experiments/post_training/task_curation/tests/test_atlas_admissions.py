# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every Atlas-admitted record lands together with the import that runs it."""

import pytest
from finestore.rl.source_identity import canonical_source_id

from experiments.post_training.task_curation.datasets import atlas_admissions
from experiments.post_training.task_curation.retained_release import RetainedCohort, import_retained
from experiments.post_training.task_curation.sources import all_sources


@pytest.mark.data_integration
def test_every_admission_record_binds_to_its_sealed_pin_and_runs():
    """A record without a sealed pin, a pipeline, or license evidence fails here."""
    sources = atlas_admissions.sources()
    assert sources, "the retained-pin table must resolve for the records to bind"
    declared = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    for source in sources:
        assert isinstance(source.config, RetainedCohort), f"{source.info.id} lost its retained reader"
        assert source.pipeline is import_retained
        assert canonical_source_id(source.info.id) in declared
        source.require_license()
    ifbench = next(s for s in sources if s.info.id == "AllenAI:IF_multi_constraints_upto5")
    assert "Final-pin QA (2026-11-26" in ifbench.info.notes, "the IFBench grader-QA note was dropped"
