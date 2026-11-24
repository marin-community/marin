# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every Atlas-admitted source is declared on main, resolves, and passes its license gate."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.source_identity import canonical_source_id
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.datasets import atlas_admissions
from experiments.post_training.task_curation.source import (
    ATLAS_ADMITTED_TAG,
    RlDataSource,
    SourceInfo,
    SourceOrigin,
)
from experiments.post_training.task_curation.sources import all_sources

ATLAS_ADMITTED_IDS = Path(__file__).with_name("atlas_available_ids.json")
"""The Atlas Available id set the embedded receipts resolved against; refresh both together."""


def test_every_atlas_available_source_is_declared():
    declared = {canonical_source_id(source.info.id) for source in all_sources().values()}
    missing = sorted(set(json.loads(ATLAS_ADMITTED_IDS.read_text())) - declared)
    assert missing == []


def test_every_admission_record_is_pending_import_and_passes_its_license_gate():
    sources = atlas_admissions.sources()
    for source in sources:
        declaration = source.declaration
        assert ATLAS_ADMITTED_TAG in source.info.tags
        assert source.pipeline is None, f"{source.info.id} must stay a record until its import lands"
        assert declaration.status == "Atlas-admitted"
        assert source.info.current_release is not None
        assert source.info.license_receipt is not None
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
    assert tagged.declaration.status == "Atlas-admitted"
