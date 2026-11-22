# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native declared setup runs once before a source-bound physical nop control."""

import hashlib
import json
import tarfile
from contextlib import contextmanager, nullcontext
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from finestore.rl.curation import ContentEvidenceReference
from harbor.environments.base import ExecResult
from harbor.host_capabilities import HostCapabilities
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.negative_control import run_negative_control
from pydantic import TypeAdapter
from taskcompendium.harbor.negative_control import (
    DeclaredSourceSetup,
    NativeInstructionSetup,
    NativeNoSetup,
    NegativeControlSetupKind,
    NegativeControlSource,
)
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.negative_controls import (
    negative_control_from_archive,
    require_negative_control,
)


@pytest.fixture
def native_control_environment():
    environment = AsyncMock()
    environment.verifier_model_access = lambda _: nullcontext()
    environment.capabilities.mounted = True
    environment.os.value = "linux"
    environment.default_user = None

    @contextmanager
    def default_user(user):
        previous = environment.default_user
        environment.default_user = user
        try:
            yield
        finally:
            environment.default_user = previous

    environment.with_default_user = default_user
    environment.with_agent_timeout = lambda _: nullcontext()
    return environment


@pytest.mark.parametrize("setup_exit", [0, 7])
@pytest.mark.asyncio
async def test_declared_setup_control_runs_once_and_roundtrips_physical_archive(
    tmp_path, monkeypatch, setup_exit, native_control_environment
):
    task_dir = tmp_path / "packet"
    task_dir.mkdir()
    config_bytes = (
        b"[agent]\ntimeout_sec=10\n[verifier]\ntimeout_sec=10\n"
        b"[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
        b'[environment.setup]\ncommand="initialize-original-fixture"\ntimeout_sec=5\n'
    )
    files = {
        "task.toml": config_bytes,
        "instruction.md": b"Fix the source problem. There is no public instruction setup block.\n",
        "environment/Dockerfile": b"FROM ubuntu:24.04\n",
        "tests/test.sh": b"echo 0 > /logs/verifier/reward.txt\n",
    }
    for name, data in files.items():
        path = task_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shape = ExecutionRequirements(cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=10, verifier_timeout_sec=10)
    specification = TaskSpec(
        id="native-source",
        source=Source(dataset="source", revision="pinned", row="one", importer_revision="one"),
        context=ConversationInput(events=(TextMessage(role="user", content=files["instruction.md"].decode()),)),
        environment_requirements=EnvironmentRequirements(setup_commands=("initialize-original-fixture",)),
        execution_requirements=shape,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Native verifier"),
        resources=ResourceGroups(verifier=tuple(inline_resource(name, data) for name, data in files.items())),
        tags=("harbor",),
    )
    checksum = Task(task_dir).checksum
    source = NegativeControlSource(
        task_id=specification.id,
        source_task_sha256=canonical_sha256(specification.model_dump(mode="json")),
        source_task_checksum=checksum,
        task_sha256=canonical_sha256(specification.model_dump(mode="json")),
        runtime_sha256=None,
        task_checksum=checksum,
        execution_requirements=shape,
        verifier_execution_requirements=None,
    )
    root = tmp_path / "control"
    environment = native_control_environment
    initialized = []

    async def execute(command, **kwargs):
        if command == "initialize-original-fixture":
            initialized.append(command)
            return ExecResult(stdout="", stderr="fixture failed" if setup_exit else "", return_code=setup_exit)
        if "/tests/test.sh" in command:
            verifier = root / "trials/declared-nop/verifier"
            verifier.mkdir(parents=True, exist_ok=True)
            (verifier / "reward.txt").write_text("0")
        return ExecResult(stdout="/", stderr="", return_code=0)

    environment.exec.side_effect = execute
    environment.exec_control.side_effect = execute
    monkeypatch.setattr("harbor.trial.trial.EnvironmentFactory.create_environment_from_config", lambda **_: environment)
    record = await run_negative_control(
        TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="declared-nop",
            trials_dir=root / "trials",
            agent=AgentConfig(name="nop"),
            environment=EnvironmentConfig(type="local"),
        ),
        source,
        root,
        DeclaredSourceSetup(task_spec=specification),
        HostCapabilities(),
    )
    assert initialized == ["initialize-original-fixture"]
    assert record["setup"]["setup_exit_code"] is None
    assert record["setup"]["preparation_completed"] is (setup_exit == 0)
    assert record["hook_calls"] == (0 if setup_exit else 1)
    assert Task(task_dir).checksum == checksum
    archive_path = tmp_path / "control.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root, arcname="control")
    archive = ContentEvidenceReference(
        uri=str(archive_path), sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    receipt = negative_control_from_archive(
        archive,
        "control/negative-control.json",
        source_task_sha256=source.source_task_sha256,
        source_task_checksum=checksum,
        execution_requirements=shape,
    )
    require_negative_control(receipt)
    assert receipt.setup_kind is NegativeControlSetupKind.DECLARED_SETUP
    assert receipt.preparation_artifact is not None
    assert receipt.completed_public_setup is (setup_exit == 0)
    assert not receipt.passed(None)
    if setup_exit:
        assert receipt.trial.result["exception_info"]["exception_type"] == "EnvironmentSetupError"
    else:
        assert receipt.conclusive
        assert receipt.trial.result["verifier_result"]["rewards"] == {"reward": 0}

    # A self-consistently rewritten archive must still preserve the original source grant of setup.
    descriptor_path = root / "declared-setup-preparation.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["declared_setup"]["command"] = "different-fixture"
    descriptor_path.write_text(json.dumps(descriptor))
    control_path = root / "negative-control.json"
    record["preparation"] = descriptor
    record["setup"]["setup_sha256"] = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    (root / "setup-receipt.json").write_text(json.dumps(record["setup"]))
    control_path.write_text(json.dumps(record))
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root, arcname="control")
    changed = ContentEvidenceReference(
        uri=str(archive_path), sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="changed the source-bound native preparation"):
        negative_control_from_archive(
            changed,
            "control/negative-control.json",
            source_task_sha256=source.source_task_sha256,
            source_task_checksum=checksum,
            execution_requirements=shape,
        )


@pytest.mark.parametrize("declared_setup", [False, True])
@pytest.mark.asyncio
async def test_native_no_setup_control_preserves_zero_setup_and_collects_physical_reward(
    tmp_path, monkeypatch, declared_setup, native_control_environment
):
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "environment/Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests/Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN mkdir -p /tests\n"
        'RUN echo "#!/bin/bash\\necho 1 > /logs/verifier/reward.txt" > /tests/test.sh\n'
        "RUN chmod +x /tests/test.sh\n"
    )
    configuration = (
        "[agent]\ntimeout_sec=10\n[verifier]\ntimeout_sec=10\n"
        "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
    )
    if declared_setup:
        configuration += '[environment.setup]\ncommand="touch /workspace/answer"\ntimeout_sec=5\n'
    (task_dir / "task.toml").write_text(configuration)
    # This is an instruction example, not a declared setup command.
    (task_dir / "instruction.md").write_text("Produce an answer. Example:\n```bash\ntouch /workspace/answer\n```\n")
    (task_dir / "tests/test.sh").write_text("test -e /workspace/answer\n")
    source_task = TaskSpec(
        id="native-task",
        source=Source(dataset="native-fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=(task_dir / "instruction.md").read_text()),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Original native verifier"),
        resources=ResourceGroups(
            verifier=tuple(
                inline_resource(str(path.relative_to(task_dir)), path.read_bytes())
                for path in sorted(task_dir.rglob("*"))
                if path.is_file()
            )
        ),
        tags=("harbor",),
    )
    frozen_specification = source_task.model_dump(mode="json")
    del frozen_specification["labels"]
    source_task = TaskSpec.model_validate_json(json.dumps(frozen_specification))
    checksum = Task(task_dir).checksum
    source = NegativeControlSource(
        task_id=source_task.id,
        source_task_sha256=task_spec_sha256(source_task),
        source_task_checksum=checksum,
        task_sha256=task_spec_sha256(source_task),
        runtime_sha256=None,
        task_checksum=checksum,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=10, verifier_timeout_sec=10
        ),
        verifier_execution_requirements=None,
    )
    selected = TypeAdapter(NativeInstructionSetup | DeclaredSourceSetup | NativeNoSetup).validate_json(
        json.dumps({"kind": "native_no_setup", "task_spec": frozen_specification})
    )
    root = tmp_path / "control"
    answer = tmp_path / "answer"
    environment = native_control_environment

    async def execute(command, **kwargs):
        if "touch /workspace/answer" in command:
            answer.touch()
        if "/tests/test.sh" in command:
            verifier = root / "trials/native-nop/verifier"
            verifier.mkdir(parents=True, exist_ok=True)
            (verifier / "reward.txt").write_text("1" if answer.exists() else "0")
        return ExecResult(stdout="/", stderr="", return_code=0)

    environment.exec.side_effect = execute
    environment.exec_control.side_effect = execute
    monkeypatch.setattr(
        "harbor.trial.trial.EnvironmentFactory.create_environment_from_config", lambda **kwargs: environment
    )
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name="native-nop",
        trials_dir=root / "trials",
        agent=AgentConfig(name="nop"),
        environment=EnvironmentConfig(type="local"),
    )
    if declared_setup:
        with pytest.raises(ValueError, match="original zero-setup source"):
            await run_negative_control(config, source, root, selected, HostCapabilities())
        assert not root.exists() and not answer.exists()
        return
    record = await run_negative_control(config, source, root, selected, HostCapabilities())
    assert record["trial_result"]["verifier_result"]["rewards"] == {"reward": 0}
    assert not answer.exists() and Task(task_dir).checksum == checksum
    assert record["setup"]["setup_exit_code"] is None
    assert record["setup"]["preparation_completed"]
    assert record["setup_kind"] == "native_no_setup"
    assert not (root / "original-public-setup.sh").exists()
    archive = tmp_path / "control.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname="control")
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
        "control/negative-control.json",
        source_task_sha256=source.source_task_sha256,
        source_task_checksum=checksum,
        execution_requirements=source.execution_requirements,
    )
    assert receipt.public_preparation_completed and receipt.conclusive and not receipt.passed(None)
    assert receipt.trial.native_grade.rewards == {"reward": 0}
    assert json.loads((root / "native-no-setup-preparation.json").read_text())["declared_setup"] is None
