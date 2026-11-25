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


@pytest.mark.data_integration
def test_the_admitted_records_leave_no_pipeline_less_declared_source():
    """The #11903 resolution: a record lands with its import, never pipeline-less."""
    for source in all_sources().values():
        if "atlas-admitted" in source.info.tags:
            raise AssertionError(f"{source.info.id} declares a pipeline-less Atlas-admitted record")
        if source.pipeline is None and source.info.build_path == "Atlas-admitted record":
            raise AssertionError(f"{source.info.id} is a pipeline-less admission record")
