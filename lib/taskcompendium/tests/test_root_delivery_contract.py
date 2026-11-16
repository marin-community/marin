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
from taskcompendium.harbor.adapter import unsupported_harbor_contract, write_harbor_task
from taskcompendium.integrity import (
    DeliveredVerifierFile,
    RootDeliveryContract,
    SharedDependencyState,
    SharedIntegrityContract,
)
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
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
        # The agreed RST converter binds a ScriptGrader collecting the reward file.
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="example.test/rst@sha256:" + "a" * 64,
            ),
            answer_path=None,
            artifacts=(
                VerifierArtifact(source="/logs/verifier/reward.txt", target="/reward.txt", kind=ArtifactKind.FILE),
            ),
            timeout=None,
        ),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(),
        integrity_contract=contract,
    )
    assessment = assess_grading_environment(task)
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SHARED
    assert assessment.status.value == "matched"
    assert shared_integrity_issue(task) is None


def test_root_delivery_contract_keeps_no_grader_tasks_shared() -> None:
    contract = delivered_contract()
    task = TaskSpec(
        id="rst-native",
        source=Source(
            dataset="laion/rts-1k-rootless-apptainer", revision="pinned", row="tasks.parquet:1", importer_revision="1"
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
    assert shared_integrity_issue(task) is None


def test_script_grader_timeout_carries_undeclared_deadlines() -> None:
    """None means the source declares no verifier deadline; the value must survive."""
    env = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image="example.test/rst@sha256:" + "a" * 64,
    )
    assert ScriptGrader(argv=("python", "/tests/grade.py"), environment=env, timeout=900.0).timeout == 900.0
    assert ScriptGrader(argv=("python", "/tests/grade.py"), environment=env, timeout=None).timeout is None
    assert ScriptGrader(argv=("python", "/tests/grade.py"), environment=env).timeout == 600.0


def test_root_delivery_task_exports_through_the_harbor_adapter(tmp_path) -> None:
    """The supported shared-executor opt-in: the exporter accepts the delivery contract."""

    contract = delivered_contract()
    grader = ScriptGrader(
        argv=("python", "/tests/grade.py"),
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image="example.test/rst@sha256:" + "a" * 64,
        ),
        answer_path=None,
        artifacts=(VerifierArtifact(source="/logs/verifier/reward.txt", target="/reward.txt", kind=ArtifactKind.FILE),),
        timeout=None,
    )
    task = TaskSpec(
        id="rst-exported",
        source=Source(
            dataset="laion/rts-1k-rootless-apptainer", revision="pinned", row="tasks.parquet:2", importer_revision="1"
        ),
        context=ConversationInput(events=(TextMessage(role="user", content="Build the summary table."),)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(),
        integrity_contract=contract,
    )
    assert unsupported_harbor_contract(task) is None
    requirements = ExecutionRequirements(
        cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900.0, verifier_timeout_sec=900.0
    )
    write_harbor_task(task, tmp_path / "packet", requirements)
    exported = TaskSpec.model_validate_json((tmp_path / "packet" / "task-spec.json").read_text())
    assert exported.integrity_contract == contract

    # A shared task without the delivery contract stays rejected.
    unbacked = task.model_copy(update={"integrity_contract": None})
    assert unsupported_harbor_contract(unbacked) == "task_spec_shared_executor_required"

    # A UID-demotion shared contract stays rejected: it needs a different executor.
    demotion = task.model_copy(
        update={
            "integrity_contract": SharedIntegrityContract(
                dependency_state=SharedDependencyState.AGENT_MUTABLE,
                reference_sha256="c" * 64,
                harness_sha256="d" * 64,
            )
        }
    )
    assert unsupported_harbor_contract(demotion) == "task_spec_shared_executor_required"
