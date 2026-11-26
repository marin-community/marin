# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every unresolved row renders its owning lane (atlas-2026.11.27 directive)."""

import pytest
from finestore.rl.status_taxonomy import SourceStatus

from experiments.post_training.task_curation import source_status
from experiments.post_training.task_curation.source_owners import SOURCE_OWNERS, source_owner


def test_registry_entries_are_well_formed():
    assert SOURCE_OWNERS, "the owner registry is empty"
    for source_id, entry in SOURCE_OWNERS.items():
        assert entry.owner, source_id
        assert entry.note, source_id
        assert source_owner(source_id) is entry


@pytest.mark.data_integration
def test_every_unresolved_row_carries_an_owner():
    rows = [*source_status.pending_rows(), *source_status.remediation_rows(), *source_status.metadata_rows()]
    assert rows, "no unresolved rows built"
    for row in rows:
        assert row["owner"], f"{row['status']} row carries no owner: {row['id']}"
    served = [
        source.info.id
        for source in source_status.all_sources().values()
        if source.status in (SourceStatus.PENDING, SourceStatus.EXCLUDED)
    ]
    assert served, "no served unresolved sources"
    for source_id in served:
        entry = source_owner(source_id)
        assert entry is not None, f"served unresolved source carries no owner: {source_id}"


@pytest.mark.data_integration
def test_extras_owners_overlay_covers_every_unresolved_source():
    extras = source_status.status_extras()
    owners = extras["owners"]
    queued_ids = {
        row["id"] for row in [*extras["pending"], *extras["metadata_exclusions"], *extras["remediation_queue"]]
    }
    served_ids = {
        source.info.id
        for source in source_status.all_sources().values()
        if source.status in (SourceStatus.PENDING, SourceStatus.EXCLUDED)
    }
    assert set(owners) == queued_ids | served_ids
    assert all(owners.values())


@pytest.mark.data_integration
def test_require_typed_rows_rejects_an_ownerless_pending_row():
    row = source_status.pending_rows()[0]
    ownerless = {**row, "owner": None}
    with pytest.raises(ValueError, match="carries no owner"):
        source_status.require_typed_rows([ownerless])
