# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas-admitted records retain their pins and remain pending import."""

from dataclasses import replace

import pytest
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.datasets import atlas_admissions
from experiments.post_training.task_curation.source import (
    ATLAS_ADMITTED_STATUS,
    ATLAS_ADMITTED_TAG,
    RlDataSource,
    SourceInfo,
    SourceOrigin,
)


def test_every_admission_record_is_pending_import_and_passes_its_license_gate():
    sources = atlas_admissions.sources()
    for source in sources:
        declaration = source.declaration
        assert source.pipeline is None, f"{source.info.id} must stay a record until its import lands"
        assert declaration.status == ATLAS_ADMITTED_STATUS
        source.require_license()


def test_an_atlas_admitted_record_with_a_stale_pin_fails_closed():
    source = atlas_admissions.sources()[0]
    stale = replace(
        source,
        info=replace(
            source.info,
            current_release=source.info.current_release.model_copy(update={"revision": "0" * 40}),
        ),
    )
    with pytest.raises(ValueError, match="must match the configured dataset pin"):
        _ = stale.current_release


def test_an_untagged_pipeline_less_source_stays_inventory():
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        origin=SourceOrigin.MARIN,
        upstream_dataset="fixture",
        upstream_name="fixture",
        build_path="test",
    )
    assert RlDataSource(info=info).declaration.status == "Inventory"
    tagged = RlDataSource(info=replace(info, tags=(ATLAS_ADMITTED_TAG,)))
    assert tagged.declaration.status == ATLAS_ADMITTED_STATUS
