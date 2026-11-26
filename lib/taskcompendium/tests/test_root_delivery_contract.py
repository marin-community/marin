# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed root-delivery contract and its grading-environment assessment surface."""

from pathlib import Path

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
    MultiSweDeliveryContract,
    RootDeliveryContract,
    SharedDependencyState,
    SharedIntegrityContract,
)
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    DockerBuildExecution,
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
from taskcompendium.runtime import shared_source
from taskcompendium.runtime.resources import inline_resource


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


REQUIREMENTS = ExecutionRequirements(
    cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900.0, verifier_timeout_sec=900.0
)


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
    write_harbor_task(task, tmp_path / "packet", REQUIREMENTS)
    exported = TaskSpec.model_validate_json((tmp_path / "packet" / "task-spec.json").read_text())
    assert exported.integrity_contract == contract

    # A shared task without the delivery contract and without any declared bridge stays rejected.
    unbacked = task.model_copy(
        update={"integrity_contract": None, "grader": grader.model_copy(update={"artifacts": ()})}
    )
    assert unsupported_harbor_contract(unbacked) == "task_spec_shared_executor_required"

    # A shared task whose grader declares its collect commands or artifact copies exports without
    # a contract: those declarations are the file-transfer boundary the grading reads through.
    bridged = task.model_copy(update={"integrity_contract": None})
    assert unsupported_harbor_contract(bridged) is None

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


def test_declared_build_opt_in_exports_root_delivery_tasks_with_their_original_context(tmp_path) -> None:
    """The DECLARATIVE enum: root-delivery tasks export their original Docker context."""
    dockerfile = (
        "FROM python:3.11-slim\nRUN apt-get update && apt-get install -y sqlite3 curl\n"
        "RUN curl -fSL -o /app/db.sqlite https://example.test/db.sqlite\nWORKDIR /app\n"
    )
    context = DockerBuildContext(
        files=(inline_resource("Dockerfile", dockerfile.encode()), inline_resource("metadata.toml", b"[meta]\n"))
    )
    contract = delivered_contract()
    grader = ScriptGrader(
        argv=("python", "/tests/grade.py"),
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=context,
        ),
        answer_path=None,
        artifacts=(VerifierArtifact(source="/logs/verifier/reward.txt", target="/reward.txt", kind=ArtifactKind.FILE),),
        timeout=900.0,
    )
    task = TaskSpec(
        id="rst-declared-build",
        source=Source(
            dataset="laion/rts-1k-rootless-apptainer", revision="pinned", row="tasks.parquet:3", importer_revision="1"
        ),
        context=ConversationInput(events=(TextMessage(role="user", content="Create the summary database."),)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(),
        integrity_contract=contract,
    )
    # Default: the unresolved build stays rejected, even with the contract.
    assert unsupported_harbor_contract(task) == "task_spec_unresolved_build"
    with pytest.raises(ValueError, match="task_spec_unresolved_build"):
        write_harbor_task(task, tmp_path / "rejected", REQUIREMENTS)
    # The explicit DECLARATIVE enum exports the original context as-is.
    assert unsupported_harbor_contract(task, docker_build=DockerBuildExecution.DECLARATIVE) is None
    write_harbor_task(task, tmp_path / "packet", REQUIREMENTS, docker_build=DockerBuildExecution.DECLARATIVE)
    exported = TaskSpec.model_validate_json((tmp_path / "packet" / "task-spec.json").read_text())
    assert exported.integrity_contract == contract
    assert exported.grader.environment.docker_build == context

    # Generic unsupported builds stay rejected even with the DECLARATIVE opt-in:
    # permission comes from the explicit enum for the caller's own task, and the
    # shared-executor gate still requires the delivery contract when no bridge is
    # declared. The grader's artifact copy is a declared bridge, so it exports.
    unbacked = task.model_copy(update={"integrity_contract": None})
    assert unsupported_harbor_contract(unbacked, docker_build=DockerBuildExecution.DECLARATIVE) is None
    unbridged = task.model_copy(
        update={"integrity_contract": None, "grader": grader.model_copy(update={"artifacts": ()})}
    )
    assert (
        unsupported_harbor_contract(unbridged, docker_build=DockerBuildExecution.DECLARATIVE)
        == "task_spec_shared_executor_required"
    )


def test_multi_swe_delivery_contract_declares_the_shared_boundary(tmp_path) -> None:
    """The Multi-SWE contract binds ScriptGrader tasks to the delivery executor."""
    contract = MultiSweDeliveryContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
        archive_sha256="e" * 64,
    )
    grader = ScriptGrader(
        argv=("python", "-m", "multiswe.host_grade"),
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image="example.test/multiswe@sha256:" + "a" * 64,
        ),
        answer_path=None,
        artifacts=(
            VerifierArtifact(source="/source-test-output.txt", target="/source-test-output.txt", kind=ArtifactKind.FILE),
        ),
        timeout=900.0,
    )
    task = TaskSpec(
        id="multiswe-row",
        source=Source(
            dataset="PrimeIntellect/Multi-SWE-RL-Verified",
            revision="pinned",
            row="train.parquet:0",
            importer_revision="1",
        ),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the build."),)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(),
        integrity_contract=contract,
    )
    assessment = assess_grading_environment(task)
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SHARED
    assert shared_integrity_issue(task) is None
    assert unsupported_harbor_contract(task) is None
    write_harbor_task(task, tmp_path / "packet", REQUIREMENTS)
    exported = TaskSpec.model_validate_json((tmp_path / "packet" / "task-spec.json").read_text())
    assert exported.integrity_contract == contract


def test_capture_preserves_pre_agent_symlinks_whose_targets_leave_the_workspace(tmp_path):
    """Pre-agent image symlinks are preserved verbatim, including escaping targets."""
    workspace = tmp_path / "workspace"
    (workspace / "tests" / "packages" / "symlinks").mkdir(parents=True)
    link = workspace / "tests" / "packages" / "symlinks" / "baz.py"
    link.symlink_to("/__init__.py")
    entry = shared_source.entry(workspace, "tests/packages/symlinks/baz.py")
    assert entry == {"kind": "symlink", "target": "/__init__.py"}


def test_quarantine_remove_moves_refused_inodes_out_of_the_workspace(tmp_path, monkeypatch):
    """A sandbox fs that refuses unlink still loses the member via rename."""
    common = type("Common", (), {})()
    refused = {"count": 0}

    def remove_path(path: Path) -> None:
        refused["count"] += 1
        raise OSError(75, "Value too large for defined data type", str(path))

    common.remove_path = remove_path
    quarantine = tmp_path / "protected" / "quarantine"
    quarantine.mkdir(parents=True)
    victim = tmp_path / "workspace" / "tests" / ".eslintrc.json"
    victim.parent.mkdir(parents=True)
    victim.write_text("config\n")

    shared_source.quarantine_remove(common, victim, str(quarantine))
    assert not victim.exists()
    quarantined = list(quarantine.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith("quarantine-")

    # Without a quarantine root the refusal propagates (fail closed).
    with pytest.raises(OSError):
        shared_source.quarantine_remove(common, victim, None)


def test_shared_source_capture_stream_appends_after_the_source_log(tmp_path: Path) -> None:
    """The shared-source guest controller captures the original test.sh's stdout into the SAME
    path the source script's own `cat > test_output.log` writes. Two writers on one file at
    independent offsets cut the source's test-name lines mid-stream (the DELTA5/DELTA6
    zero-capture fixtures: the harness echo glued into `✓ <name> (NN`). The capture stream must
    therefore open in append mode: the source's cat still owns and truncates its log, and the
    harness's writes land after the source's content instead of stomping it."""
    source = Path(shared_source.__file__).read_text()
    assert 'output.open("ab")' in source, (
        "the shared-source capture stream must open in append mode; 'wb' re-introduces the "
        "two-writers collision that cut test-name lines"
    )

    # The mechanism the fix relies on: a handle opened for append keeps writing after
    # content another writer truncates and rewrites through its own description.
    log = tmp_path / "test_output.log"
    with log.open("ab") as stream:
        stream.write(b"ECHO-SHOULD-NOT-OVERWRITE\n")
        child = log.open("wb")  # the source's own cat: truncates and rewrites
        child.write(b"PASS suite\n\xe2\x9c\x93 test line\n")
        child.close()
        stream.write(b"test_cmd exit code: 0\n")
    text = log.read_text()
    assert text.startswith("PASS suite\n✓ test line\n")
    assert text.index("ECHO-SHOULD-NOT-OVERWRITE") > text.index("✓ test line")
    assert "test_cmd exit code: 0" in text
