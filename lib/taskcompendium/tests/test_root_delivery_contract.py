# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed root-delivery contract and its grading-environment assessment surface."""

import pytest
from pydantic import ValidationError

from taskcompendium.grading_environment import (
    GradingEnvironmentMode,
    assess_grading_environment,
    shared_integrity_issue,
)
from taskcompendium.integrity import (
    DeliveredVerifierFile,
    RootDeliveryContract,
    SharedDependencyState,
)
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)


def delivered_contract(**overrides) -> RootDeliveryContract:
    fields = {
        "dependency_state": SharedDependencyState.AGENT_MUTABLE,
        "verifier_files": (
            DeliveredVerifierFile(name="test.sh", mode=0o755, sha256="a" * 64),
            DeliveredVerifierFile(name="test_state.py", mode=0o644, sha256="b" * 64),
        ),
    }
    fields.update(overrides)
    return RootDeliveryContract(**fields)


def test_root_delivery_contract_pins_delivery_identity_and_deadline() -> None:
    contract = delivered_contract(agent_user="root", undeclared_deadline=True)
    assert (contract.id, contract.version) == ("root-delivery", 1)
    assert contract.dependency_state is SharedDependencyState.AGENT_MUTABLE
    assert contract.agent_user == "root"
    assert contract.undeclared_deadline is True
    assert contract.known_open_gaps == (
        "Agent-installed pytest plugins and imports in the shared image environment",
        "Candidate-produced test-output injection",
    )
    assert [f.mode for f in contract.verifier_files] == [0o755, 0o644]


def test_root_delivery_contract_requires_files_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RootDeliveryContract(dependency_state=SharedDependencyState.AGENT_MUTABLE, verifier_files=())
    with pytest.raises(ValidationError):
        delivered_contract(agent_uid=10001)


def test_root_delivery_contract_declares_the_shared_boundary() -> None:
    """The contract itself asserts SHARED: delivery grades the agent's environment."""
    contract = delivered_contract()
    task = TaskSpec(
        id="rst-row",
        source=Source(
            dataset="laion/rts-1k-rootless-apptainer", revision="pinned", row="tasks.parquet:0", importer_revision="1"
        ),
        context=ConversationInput(events=(TextMessage(role="user", content="Build the summary table."),)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The delivered verifier runs in the agent's environment"),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(),
        integrity_contract=contract,
    )
    assessment = assess_grading_environment(task)
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SHARED
    assert assessment.status.value == "matched"
    assert shared_integrity_issue(task) is None
