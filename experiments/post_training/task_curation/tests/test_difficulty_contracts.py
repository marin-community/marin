# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the vintage-pinned difficulty task contracts.

The 2026-11-17 TaskSpec serializer began omitting absent late-joining fields
that the 2026-11-13 difficulty freeze serialized as explicit nulls, which moved
the ``task_sha256`` digest space and killed hosted-v13 at prepare time. These
tests pin both receipt vintages, the resolution between them, and the precise
failure when a task matches neither.
"""

import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation.difficulty_contracts import (
    CONTRACT_VINTAGE_ABSENT_OMITTED,
    CONTRACT_VINTAGE_NULL_DEFAULTS,
    ContractVintageError,
    difficulty_task_contract,
    difficulty_task_payload,
    payload_differences,
    resolve_receipt_vintage,
)


def _task() -> TaskSpec:
    return TaskSpec(
        id="vintage-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Original question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="contract test fixture"),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def _task_with_execution_requirements() -> TaskSpec:
    return _task().model_copy(
        update={
            "execution_requirements": ExecutionRequirements(
                cpus=2, memory_mb=1024, storage_mb=2048, agent_timeout_sec=600.0, verifier_timeout_sec=600.0
            )
        }
    )


def test_null_defaults_vintage_serializes_absent_execution_requirements_as_null():
    task = _task()
    assert "execution_requirements" not in task.model_fields_set
    payload = difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS)
    assert payload["execution_requirements"] is None
    assert "verifier_execution_requirements" not in payload
    assert "completion_protocol" not in payload


def test_absent_omitted_vintage_omits_absent_execution_requirements():
    payload = difficulty_task_payload(_task(), CONTRACT_VINTAGE_ABSENT_OMITTED)
    assert "execution_requirements" not in payload


def test_the_two_vintages_disagree_exactly_on_the_absent_field():
    """The hosted-v13 break, pinned: same parsed task, different digest space."""
    task = _task()
    null_sha = canonical_sha256(difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS))
    omitted_sha = canonical_sha256(difficulty_task_payload(task, CONTRACT_VINTAGE_ABSENT_OMITTED))
    assert null_sha != omitted_sha
    differences = payload_differences(
        difficulty_task_payload(task, CONTRACT_VINTAGE_ABSENT_OMITTED),
        difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS),
    )
    assert differences == ["execution_requirements present only in the second serialization (None)"]


def test_vintages_agree_when_execution_requirements_is_present():
    task = _task_with_execution_requirements()
    for payload in (
        difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS),
        difficulty_task_payload(task, CONTRACT_VINTAGE_ABSENT_OMITTED),
    ):
        assert payload["execution_requirements"]["cpus"] == 2
    assert canonical_sha256(difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS)) == canonical_sha256(
        difficulty_task_payload(task, CONTRACT_VINTAGE_ABSENT_OMITTED)
    )


def test_null_defaults_vintage_rejects_late_joining_evidence():
    task = _task().model_copy(update={"completion_protocol": "tool_free_single"})
    with pytest.raises(ContractVintageError, match="completion_protocol"):
        difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS)


def test_contract_includes_runtime_digest():
    task = _task()
    contract = difficulty_task_contract(task, {"runtime": "packet"}, CONTRACT_VINTAGE_NULL_DEFAULTS)
    assert contract["task_id"] == task.id
    assert contract["task_sha256"] == canonical_sha256(difficulty_task_payload(task, CONTRACT_VINTAGE_NULL_DEFAULTS))
    assert contract["runtime_sha256"] == canonical_sha256({"runtime": "packet"})


def test_resolve_prefers_the_current_vintage_for_current_receipts():
    task = _task()
    current = difficulty_task_contract(task, {}, CONTRACT_VINTAGE_ABSENT_OMITTED)
    vintage = resolve_receipt_vintage({task.id: current["task_sha256"]}, [task], [{}])
    assert vintage == CONTRACT_VINTAGE_ABSENT_OMITTED


def test_resolve_recovers_the_frozen_null_defaults_receipts():
    """The 2026.11.13.8 queue's receipts resolve to the null-defaults vintage."""
    task = _task()
    frozen = difficulty_task_contract(task, {}, CONTRACT_VINTAGE_NULL_DEFAULTS)
    vintage = resolve_receipt_vintage({task.id: frozen["task_sha256"]}, [task], [{}])
    assert vintage == CONTRACT_VINTAGE_NULL_DEFAULTS


def test_resolve_names_the_task_when_no_vintage_matches():
    task = _task_with_execution_requirements()
    frozen = difficulty_task_contract(task, {}, CONTRACT_VINTAGE_ABSENT_OMITTED)
    with pytest.raises(ContractVintageError) as error:
        resolve_receipt_vintage({task.id: "f" * 64}, [task], [{}])
    message = str(error.value)
    assert task.id in message
    assert "ffff" in message
    assert frozen["task_sha256"][:16] in message
    assert "changed after the receipt was frozen" in message


def test_unknown_vintage_is_rejected():
    with pytest.raises(ContractVintageError, match="Unknown contract vintage"):
        difficulty_task_payload(_task(), "2049-01-01-hypothetical")
