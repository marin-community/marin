# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declared integrity contracts parse and the consumer honors or refuses them as declared."""

import hashlib

import pytest
import tomlkit

from taskcompendium.integrity import (
    DeliveredVerifierFile,
    MultiSweDeliveryContract,
    RootDeliveryContract,
    SharedDependencyState,
    SharedIntegrityContract,
    SharedSourceFamily,
    SourceSharedRunner,
)
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    NoGrader,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.integrity import require_integrity_contract
from taskcompendium.runtime.resources import inline_resource

IMAGE = "fixture@sha256:" + "0" * 64
TEST_SCRIPT = b"#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
GRADER_PY = b"print('grade')\n"


def private_file(name: str, content: bytes, mode: int) -> DeliveredVerifierFile:
    return DeliveredVerifierFile(name=name, mode=mode, sha256=hashlib.sha256(content).hexdigest())


def root_delivery(**overrides) -> RootDeliveryContract:
    fields = {
        "dependency_state": SharedDependencyState.AGENT_MUTABLE,
        "verifier_files": (private_file("test.sh", TEST_SCRIPT, 0o755), private_file("grader.py", GRADER_PY, 0o644)),
    }
    return RootDeliveryContract(**{**fields, **overrides})


def delivery_task(contract, *, verifier=None, worker=(), timeout: float | None = 120.0, collect=True) -> TaskSpec:
    """A FineEnvs/RST-shaped task: private verifier files archived beside a root collect delivery."""
    if verifier is None:
        verifier = (
            inline_resource("source_archive/tests/test.sh", TEST_SCRIPT).model_copy(update={"mode": "0755"}),
            inline_resource("source_archive/tests/grader.py", GRADER_PY).model_copy(update={"mode": "0644"}),
        )
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE)
    grader = ScriptGrader(
        argv=("cp", "/source-reward.txt", "/logs/verifier/reward.txt"),
        cwd="/",
        environment=environment,
        answer_path=None,
        collect=(
            (VerifierCommand(argv=("bash", "-c", "quiesce; deliver; bash /tests/test.sh"), cwd="/"),) if collect else ()
        ),
        artifacts=(
            VerifierArtifact(source="/logs/verifier/reward.txt", target="/source-reward.txt", kind=ArtifactKind.FILE),
        ),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
        timeout=timeout,
    )
    return TaskSpec(
        id="delivery-fixture",
        source=Source(dataset="fixture/delivery", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Answer in /workdir/answer.txt."),)),
        environment_requirements=environment,
        integrity_contract=contract,
        resources=ResourceGroups(worker=worker, verifier=verifier),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
    )


def test_root_delivery_contract_round_trips_and_is_honored():
    task = delivery_task(root_delivery())
    restored = TaskSpec.model_validate_json(task.model_dump_json())
    assert restored.integrity_contract == task.integrity_contract
    require_integrity_contract(restored)


def test_multiswe_archive_must_be_privately_staged():
    archive = b"captured home archive"
    staged = (inline_resource("home-grading.tar.gz", archive),)
    contract = MultiSweDeliveryContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE, archive_sha256=hashlib.sha256(archive).hexdigest()
    )
    require_integrity_contract(delivery_task(contract, verifier=staged))
    with pytest.raises(ValueError, match="do not carry"):
        require_integrity_contract(delivery_task(contract, verifier=(inline_resource("other.tar.gz", b"x"),)))


def test_delivered_material_visible_to_the_actor_fails_closed():
    leaked = (inline_resource("workspace/test.sh", TEST_SCRIPT),)
    with pytest.raises(ValueError, match="actor-visible"):
        require_integrity_contract(delivery_task(root_delivery(), worker=leaked))


def test_unstaged_or_remoded_material_fails_closed():
    with pytest.raises(ValueError, match="do not carry"):
        require_integrity_contract(delivery_task(root_delivery(), verifier=()))
    remoded = root_delivery(verifier_files=(private_file("test.sh", TEST_SCRIPT, 0o600),))
    with pytest.raises(ValueError, match="do not carry"):
        require_integrity_contract(delivery_task(remoded))


def test_delivery_requires_the_graders_own_collect_path_and_deadline():
    with pytest.raises(ValueError, match="trusted collect"):
        require_integrity_contract(delivery_task(root_delivery(), collect=False))
    with pytest.raises(ValueError, match="deadline"):
        require_integrity_contract(delivery_task(root_delivery(undeclared_deadline=True)))
    require_integrity_contract(delivery_task(root_delivery(undeclared_deadline=True), timeout=None))


def test_unported_protections_fail_closed():
    shared = SharedIntegrityContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE, reference_sha256="a" * 64, harness_sha256="b" * 64
    )
    with pytest.raises(NotImplementedError, match="shared grading protections"):
        require_integrity_contract(delivery_task(shared))
    with pytest.raises(NotImplementedError, match="image user"):
        require_integrity_contract(delivery_task(root_delivery(agent_user="agent")))


CONTROLLER = b"print('control')\n"
SOURCE_RUNNER = b"print('shared pytest')\n"


def shared_source(**overrides) -> SharedIntegrityContract:
    fields = {
        "id": "shared-source",
        "dependency_state": SharedDependencyState.AGENT_MUTABLE,
        "reference_sha256": "a" * 64,
        "harness_sha256": hashlib.sha256(CONTROLLER).hexdigest(),
        "native": SourceSharedRunner(
            family=SharedSourceFamily.SWE_REBENCH, shared_controller_sha256=hashlib.sha256(SOURCE_RUNNER).hexdigest()
        ),
    }
    return SharedIntegrityContract(**{**fields, **overrides})


def harbor_shared_task(contract, *, package_contract=None, controller=CONTROLLER) -> TaskSpec:
    """A SWE-rebench-shaped Harbor package whose own pinned controller grades the trial."""
    declared = (contract if package_contract is None else package_contract).model_dump(mode="json", exclude_none=True)
    toml = tomlkit.dumps({"metadata": {"integrity_contract": declared}, "verifier": {"environment_mode": "shared"}})
    test = f"#!/bin/bash\nexec python -I {contract.harness} grade --contract /opt/contract.json\n".encode()
    return TaskSpec(
        id="fixture:harbor-shared",
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the bug."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Harbor grades the completed trial"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("task.toml", toml.encode()),
                inline_resource("tests/test.sh", test),
                inline_resource("environment/taskcompendium-shared/control.py", controller),
                inline_resource("environment/taskcompendium-shared/shared_pytest.py", SOURCE_RUNNER),
            )
        ),
        source=Source(dataset="fixture/harbor-shared", revision="r", row="0", importer_revision="fixture"),
        tags=("harbor",),
        integrity_contract=contract,
    )


def test_harbor_shared_source_package_is_accepted_only_with_its_pinned_controller():
    require_integrity_contract(harbor_shared_task(shared_source()))
    with pytest.raises(ValueError, match="pinned controller"):
        require_integrity_contract(harbor_shared_task(shared_source(), controller=b"print('tampered')\n"))
    with pytest.raises(ValueError, match="does not declare"):
        require_integrity_contract(harbor_shared_task(shared_source(), package_contract=shared_source(agent_uid=0)))
