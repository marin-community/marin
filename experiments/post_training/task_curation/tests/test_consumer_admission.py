# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consumer failures hold a source pending without discarding its population."""

from dataclasses import replace

import pytest
from finestore.rl.curation import ConsumerNeed, ConsumerReason, OversightEvidence, OversightLevel
from finestore.rl.status_taxonomy import SourceStatus

from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from infra.marina.apps.rl_data_catalog.server.admission import require_population_admission
from infra.marina.apps.rl_data_catalog.server.consumer import consumer_projection
from infra.marina.apps.rl_data_catalog.server.filtration import filtration


@pytest.fixture
def qualified_source(tmp_path):
    source = with_admission_census(number_source(tmp_path / "input"))
    return replace(
        source,
        info=replace(
            source.info,
            consumer_requirements=(ConsumerNeed.TOOL_PROVIDER,),
            oversight=OversightEvidence(
                level=OversightLevel.SAMPLE,
                verification="Executed controls",
            ),
        ),
    )


@pytest.mark.parametrize(
    "failure",
    [None, ConsumerReason.JUDGE_UNREGISTERED, ConsumerReason.TOOL_UNREGISTERED, ConsumerReason.SERVICE_UNREGISTERED],
)
def test_consumer_failure_blocks_new_and_retained_admission(qualified_source, failure):
    source = qualified_source
    receipt = source.info.consumer_receipt
    assert receipt is not None
    receipt = None if failure is None else receipt.model_copy(update={"failed_tasks": 1, "failures": {failure: 1}})
    held = replace(source, info=replace(source.info, consumer_receipt=receipt))
    reason = failure or ConsumerReason.RECEIPT_MISSING
    assert held.status is SourceStatus.PENDING
    assert not held.exclusion_claims
    with pytest.raises(ValueError, match=reason.value):
        require_population_admission(held.declaration)
    historical = {**held.declaration.model_dump(mode="json"), "status": "Available"}
    projected = consumer_projection(historical)
    assert projected["status"] == "Pending"
    assert projected["consumer_blockers"] == [reason.value]
    assert projected["task_count"] == source.declaration.task_count
    assert filtration(historical)["admitted_count"] is None


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"runtime_commit": "c" * 40}, ConsumerReason.RUNTIME_STALE),
        ({"task_specs_sha256": "c" * 64}, ConsumerReason.POPULATION_MISMATCH),
        ({"checked_tasks": 2}, ConsumerReason.POPULATION_MISMATCH),
        ({"dataset_revision": "c" * 40}, ConsumerReason.POPULATION_MISMATCH),
    ],
)
def test_consumer_receipt_must_match_current_runtime_and_full_population(qualified_source, update, reason):
    source = qualified_source
    receipt = source.info.consumer_receipt.model_copy(update=update)
    held = replace(source, info=replace(source.info, consumer_receipt=receipt))
    assert held.status is SourceStatus.PENDING
    with pytest.raises(ValueError, match=reason.value):
        require_population_admission(held.declaration)


def test_passing_consumer_receipt_permits_qualified_population(qualified_source):
    source = qualified_source
    assert source.status is SourceStatus.AVAILABLE
    require_population_admission(source.declaration)
    projected = consumer_projection(source.declaration.model_dump(mode="json"))
    assert projected["status"] == "Available"
    assert projected.get("consumer_blockers", []) == []


@pytest.mark.parametrize("status", ["Available", "Excluded"])
def test_standard_sources_keep_status_without_a_receipt(qualified_source, status):
    source = replace(
        qualified_source, info=replace(qualified_source.info, consumer_requirements=(), consumer_receipt=None)
    )
    assert source.status is SourceStatus.AVAILABLE
    declaration = {**source.declaration.model_dump(mode="json"), "status": status}
    assert consumer_projection(declaration) is declaration


def test_global_schema_drift_does_not_block_bespoke_admission(qualified_source):
    source = qualified_source
    receipt = source.info.consumer_receipt.model_copy(
        update={
            "failed_tasks": 1,
            "failures": {ConsumerReason.SCHEMA_REJECTED: 1},
        }
    )
    source = replace(source, info=replace(source.info, consumer_receipt=receipt))
    assert source.status is SourceStatus.AVAILABLE
    require_population_admission(source.declaration)
