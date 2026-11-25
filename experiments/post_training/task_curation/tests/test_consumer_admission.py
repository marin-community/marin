# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consumer findings are advisory until a source-specific user decision applies."""

import json
from dataclasses import replace

import pytest
from finestore.rl.curation import (
    ConsumerDecision,
    ConsumerNeed,
    ConsumerReason,
    OversightEvidence,
    OversightLevel,
    QualifiedSourceDeclaration,
)
from finestore.rl.status_taxonomy import SourceStatus
from pydantic import ValidationError

from experiments.post_training.task_curation.consumer_scope import CONSUMER_RUNTIME_COMMIT
from experiments.post_training.task_curation.source_status import consumer_inputs
from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from infra.marina.apps.rl_data_catalog.server import consumer
from infra.marina.apps.rl_data_catalog.server.admission import require_population_admission
from infra.marina.apps.rl_data_catalog.server.consumer import consumer_projection
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.restrictions import private_declaration_projection


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
def test_consumer_failure_cannot_change_admission_or_status_without_user_decision(qualified_source, failure):
    source = qualified_source
    receipt = source.info.consumer_receipt
    assert receipt is not None
    receipt = None if failure is None else receipt.model_copy(update={"failed_tasks": 1, "failures": {failure: 1}})
    held = replace(source, info=replace(source.info, consumer_receipt=receipt))
    reason = failure or ConsumerReason.RECEIPT_MISSING
    assert held.status is SourceStatus.AVAILABLE
    assert not held.exclusion_claims
    require_population_admission(held.declaration)
    historical = {**held.declaration.model_dump(mode="json"), "status": "Available"}
    projected = consumer_projection(historical)
    assert projected["status"] == "Available"
    assert projected["consumer_findings"] == [reason.value]
    assert projected["task_count"] == source.declaration.task_count
    baseline = {**historical, "consumer_requirements": [], "consumer_findings": []}
    assert filtration(historical) == filtration(baseline)


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
    assert held.status is SourceStatus.AVAILABLE
    assert held.consumer_findings == (reason,)
    require_population_admission(held.declaration)


def test_passing_consumer_receipt_permits_qualified_population(qualified_source):
    source = qualified_source
    assert source.status is SourceStatus.AVAILABLE
    require_population_admission(source.declaration)
    projected = consumer_projection(source.declaration.model_dump(mode="json"))
    assert projected["status"] == "Available"
    assert projected.get("consumer_findings", []) == []


@pytest.mark.parametrize("status", ["Available", "Excluded"])
def test_standard_sources_keep_status_without_a_receipt(qualified_source, status):
    source = replace(
        qualified_source, info=replace(qualified_source.info, consumer_requirements=(), consumer_receipt=None)
    )
    assert source.status is SourceStatus.AVAILABLE
    declaration = {**source.declaration.model_dump(mode="json"), "status": status}
    declaration["consumer_requirements"] = None
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


@pytest.mark.parametrize("status", ["Pending", "Excluded"])
def test_only_matching_source_specific_user_decision_can_apply_consumer_status(qualified_source, status):
    source = qualified_source
    receipt = source.info.consumer_receipt.model_copy(
        update={"failed_tasks": 1, "failures": {ConsumerReason.TOOL_UNREGISTERED: 1}}
    )
    decision = ConsumerDecision(
        source_id=source.info.id,
        runtime_commit=receipt.runtime_commit,
        task_specs_sha256=receipt.task_specs_sha256,
        reason=ConsumerReason.TOOL_UNREGISTERED,
        status=status,
        user_ruling="fixture user ruling: hold only this exact source population",
    )
    source = replace(source, info=replace(source.info, consumer_receipt=receipt, consumer_decision=decision))
    assert source.status.value == status
    # Recheck an existing Available declaration; new Excluded declarations also need typed exclusion claims.
    historical = {
        **qualified_source.declaration.model_dump(mode="json"),
        "consumer_receipt": receipt.model_dump(mode="json"),
        "consumer_decision": decision.model_dump(mode="json"),
    }
    with pytest.raises(ValueError, match="User-approved consumer decision"):
        require_population_admission(QualifiedSourceDeclaration.model_validate(historical))
    projected = consumer_projection(historical)
    assert projected["status"] == status
    assert projected["consumer_decision"]["user_ruling"] == decision.user_ruling
    for update in (
        {"source_id": "fixture:another-source"},
        {"runtime_commit": "c" * 40},
        {"task_specs_sha256": "c" * 64},
        {"reason": ConsumerReason.SERVICE_UNREGISTERED},
    ):
        stale = replace(source, info=replace(source.info, consumer_decision=decision.model_copy(update=update)))
        assert stale.status is SourceStatus.AVAILABLE
        require_population_admission(stale.declaration)
        assert (
            consumer_projection({**stale.declaration.model_dump(mode="json"), "status": "Available"})["status"]
            == "Available"
        )


def test_user_decision_requires_a_ruling_citation(qualified_source):
    receipt = qualified_source.info.consumer_receipt
    with pytest.raises(ValueError, match="user_ruling"):
        ConsumerDecision.model_validate(
            {
                "source_id": qualified_source.info.id,
                "runtime_commit": receipt.runtime_commit,
                "task_specs_sha256": receipt.task_specs_sha256,
                "reason": ConsumerReason.TOOL_UNREGISTERED,
                "status": "Pending",
            }
        )


@pytest.mark.parametrize("runtime_pin", [None, "c" * 40])
def test_source_runtime_selection_survives_declaration_and_serving_evidence(qualified_source, runtime_pin, monkeypatch):
    selected = runtime_pin or CONSUMER_RUNTIME_COMMIT
    receipt = qualified_source.info.consumer_receipt.model_copy(update={"runtime_commit": selected})
    source = replace(
        qualified_source,
        info=replace(qualified_source.info, consumer_runtime_commit=runtime_pin, consumer_receipt=receipt),
    )
    declaration = QualifiedSourceDeclaration.model_validate_json(source.declaration.model_dump_json())
    assert source.consumer_findings == ()
    assert declaration.consumer_findings == []
    require_population_admission(declaration)
    assert consumer_projection(declaration.model_dump(mode="json"))["consumer_findings"] == []
    private = private_declaration_projection(
        {**declaration.model_dump(mode="json"), "data_restriction": "private_always"}
    )
    assert consumer_projection(private)["consumer_findings"] == []

    snapshot = consumer_inputs([source])
    assert ("consumer_runtime_commit" in snapshot[source.info.id]) is (runtime_pin is not None)
    assert ("consumer_runtime_commit" in declaration.model_dump(mode="json")) is (runtime_pin is not None)
    # The packaged snapshot is the serving policy even when a stored declaration has an older pin.
    historical = qualified_source.declaration.model_dump(mode="json")
    historical["consumer_runtime_commit"] = "d" * 40
    monkeypatch.setattr(consumer, "CONSUMER_INPUTS", json.dumps(snapshot))
    consumer.packaged_consumer_inputs.cache_clear()
    try:
        projected = consumer_projection(historical)
        assert projected["consumer_findings"] == []
        assert projected.get("consumer_runtime_commit") == runtime_pin
        stale_receipt = receipt.model_copy(update={"runtime_commit": "e" * 40})
        stale = replace(source, info=replace(source.info, consumer_receipt=stale_receipt))
        assert stale.consumer_findings == (ConsumerReason.RUNTIME_STALE,)
    finally:
        consumer.packaged_consumer_inputs.cache_clear()


@pytest.mark.parametrize("runtime_pin", [None, "c" * 40])
def test_source_runtime_controls_user_decisions_at_every_admission_path(qualified_source, runtime_pin, monkeypatch):
    selected = runtime_pin or CONSUMER_RUNTIME_COMMIT
    receipt = qualified_source.info.consumer_receipt.model_copy(
        update={"runtime_commit": selected, "failed_tasks": 1, "failures": {ConsumerReason.TOOL_UNREGISTERED: 1}}
    )
    decision = ConsumerDecision(
        source_id=qualified_source.info.id,
        runtime_commit=selected,
        task_specs_sha256=receipt.task_specs_sha256,
        reason=ConsumerReason.TOOL_UNREGISTERED,
        status="Pending",
        user_ruling="Hold this source at its selected runtime until the provider is registered",
    )
    source = replace(
        qualified_source,
        info=replace(
            qualified_source.info,
            consumer_runtime_commit=runtime_pin,
            consumer_receipt=receipt,
            consumer_decision=decision,
        ),
    )
    assert source.approved_consumer_status is SourceStatus.PENDING
    declaration = QualifiedSourceDeclaration.model_validate_json(source.declaration.model_dump_json())
    with pytest.raises(ValueError, match="User-approved consumer decision"):
        require_population_admission(declaration)
    historical = {**declaration.model_dump(mode="json"), "status": "Available"}
    assert consumer_projection(historical)["status"] == "Pending"

    monkeypatch.setattr(consumer, "CONSUMER_INPUTS", json.dumps(consumer_inputs([source])))
    consumer.packaged_consumer_inputs.cache_clear()
    try:
        assert consumer_projection(historical)["status"] == "Pending"
        other = replace(source, info=replace(source.info, consumer_runtime_commit="e" * 40))
        assert other.approved_consumer_status is None
        require_population_admission(other.declaration)
    finally:
        consumer.packaged_consumer_inputs.cache_clear()


def test_source_runtime_requires_an_immutable_commit_in_declarations_and_serving_evidence(qualified_source, monkeypatch):
    declaration = qualified_source.declaration.model_dump(mode="json")
    declaration["consumer_runtime_commit"] = "main"
    with pytest.raises(ValidationError, match="consumer_runtime_commit"):
        QualifiedSourceDeclaration.model_validate(declaration)
    monkeypatch.setattr(consumer, "CONSUMER_INPUTS", "{}")
    consumer.packaged_consumer_inputs.cache_clear()
    try:
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            consumer_projection(declaration)
    finally:
        consumer.packaged_consumer_inputs.cache_clear()
