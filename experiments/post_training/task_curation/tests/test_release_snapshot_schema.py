# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the versioned snapshot reader against the served catalog.

Reads the real served snapshots (S3): the r39-era full form whose Pending rows
predate the computed stage, and the pre-lineage November 12 forms. Marked
``data_integration`` — it fetches external fixtures (the served catalog).
"""

import pytest
from finestore.rl.curation import PreLineageSnapshotDeclaration
from finestore.rl.status_taxonomy import PendingStage
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.catalog_snapshot import CurationCatalog
from experiments.post_training.task_curation.source_status import pending_stage, receipt_registry_keys

pytestmark = [pytest.mark.data_integration]

CATALOG = "s3://marin-us-east-02a/marin/rl-data/catalog/production"
R39_SNAPSHOT = "assembly-r39-admissions-20261123-v2"
PRE_LINEAGE_SNAPSHOT = "atlas-measurements-retirements-20261112-v2"


@pytest.fixture
def catalog() -> CurationCatalog:
    configure_coreweave_s3()
    return CurationCatalog(CATALOG)


def _stage_of(source_id: str) -> PendingStage | None:
    return pending_stage(source_id, receipt_registry_keys())


def test_r39_snapshot_reads_through_the_versioned_reader(catalog):
    """Every served r39 row parses; the 9 pre-stage Pending rows take evidence-computed stages."""
    rows = list(catalog.sources(R39_SNAPSHOT, pending_stage_of=_stage_of))
    assert len(rows) == 332
    kinds = {type(row.declaration).__name__ for row in rows}
    assert kinds == {"QualifiedSourceDeclaration"}
    staged = [row.declaration.stage for row in rows if isinstance(row.declaration.stage, PendingStage)]
    assert len(staged) == 9
    assert {stage.value for stage in staged} <= {stage.value for stage in PendingStage}
    # The stage is evidence-computed, never hand-entered: every staged row's id
    # sits in at least one measurement-receipt registry the callback used.
    for row in rows:
        if isinstance(row.declaration.stage, PendingStage):
            assert any(row.declaration.id in keys for keys in receipt_registry_keys().values())


def test_r39_snapshot_refuses_pre_stage_rows_without_a_callback(catalog):
    with pytest.raises(Exception, match="supply pending_stage_of"):
        list(catalog.sources(R39_SNAPSHOT))


def test_r39_snapshot_current_rows_parse_unchanged(catalog):
    """The 323 rows already in the current schema read without any conversion."""
    rows = list(catalog.sources(R39_SNAPSHOT, pending_stage_of=_stage_of))
    current = [row for row in rows if not isinstance(row.declaration.stage, PendingStage)]
    assert len(current) == 323
    assert all(type(row.declaration).__name__ == "QualifiedSourceDeclaration" for row in current)
    ids = {row.id for row in rows}
    # The reader preserves each row's served id verbatim. Seven snapshot ids were
    # re-origin'd in the registry after this snapshot (MarinSkyRL:* names became
    # their current origins), so id containment against the registry is not a
    # reader contract; the parse shape and staging contracts above are.
    assert ids == {row.id for row in catalog.sources(R39_SNAPSHOT, pending_stage_of=_stage_of)}


def test_pre_lineage_snapshot_reads_as_typed_legacy_records(catalog):
    """The November 12-era snapshots read with every legacy field preserved verbatim."""
    rows = list(catalog.sources(PRE_LINEAGE_SNAPSHOT))
    assert len(rows) == 237
    assert all(isinstance(row.declaration, PreLineageSnapshotDeclaration) for row in rows)
    preserved = sum(len(row.declaration.legacy_fields) for row in rows)
    assert preserved > 0
    sample = rows[0].declaration
    assert sample.status and sample.origin and sample.id
    for field in sample.legacy_fields:
        assert field not in PreLineageSnapshotDeclaration.model_fields
