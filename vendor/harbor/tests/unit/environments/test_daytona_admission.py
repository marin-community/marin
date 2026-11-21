# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard command controls admission without changing task resources."""

import asyncio
import json
import sys

import pytest
from harbor.environments.daytona_admission import admit_create
from harbor.environments.daytona import DaytonaEnvironment
from harbor.host_capabilities import HostCapabilities
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec, UnsupportedMachineSpec
from taskcompendium.harbor.adapter import DockerBuildExecution, write_harbor_task
from taskcompendium.models import (
    AnswerType, PlainText, Source, CommandSemantics, ConversationInput, EnvironmentRequirements, FileReward,
    RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage,
)
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS


@pytest.mark.parametrize("cpu", [2, 8, None])
def test_command_rechecks_capacity_and_rejects_an_oversized_machine(tmp_path, cpu):
    report = {
        "max_concurrency": {"limit": 0},
        "org_limits": {"per_sandbox": {"cpu": 4, "memory_gib": 8, "disk_gib": 10}},
    }
    if cpu is None:
        report["org_limits"]["per_sandbox"].update(cpu=6, memory_gib=12, disk_gib=20)
    response = tmp_path / "response.json"
    response.write_text(json.dumps(report))
    calls = tmp_path / "calls.json"
    command = tmp_path / "standard-usage-command.py"
    command.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "response,calls=map(Path,sys.argv[1:3])\n"
        "seen=json.loads(calls.read_text()) if calls.exists() else []\n"
        "report=json.loads(response.read_text())\n"
        "report['max_concurrency']['limit']=int(bool(seen))\n"
        "seen.append(sys.argv[3:]);calls.write_text(json.dumps(seen))\n"
        "print(json.dumps(report))\n"
    )
    memory, disk = (None, None) if cpu is None else (2049, 3073)
    spec = MachineSpec(RegistryImage("original"), cpus=cpu, memory_mb=memory, storage_mb=disk)
    pending = admit_create((sys.executable, str(command), str(response), str(calls)), spec, recheck_interval=0)
    if cpu == 8:
        with pytest.raises(UnsupportedMachineSpec):
            asyncio.run(pending)
        assert len(json.loads(calls.read_text())) == 1
    else:
        asyncio.run(pending)
        checked = json.loads(calls.read_text())
        assert len(checked) == 2
        profiles = [dict(zip(arguments[:-1:2], arguments[1:-1:2], strict=True)) for arguments in checked]
        assert all(arguments[-1] == "--json" for arguments in checked)
        expected = [{"--cpu": "2", "--memory-gib": "3", "--disk-gib": "4"}] * 2 if cpu is not None else [
            {"--cpu": "1", "--memory-gib": "1", "--disk-gib": "1"},
            {"--cpu": "6", "--memory-gib": "12", "--disk-gib": "20"},
        ]
        assert profiles == expected
    assert spec.cpus == cpu and spec.memory_mb == memory and spec.storage_mb == disk


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["native", "task-spec"])
async def test_harbor_admission_error_prevents_any_provider_create(tmp_path, monkeypatch, entrypoint):
    class Provider:
        creates = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def create(self, *args, **kwargs):
            self.creates += 1
            raise AssertionError("Provider create must remain behind admission")

    provider = Provider()
    called = tmp_path / "admission-called"
    command = [sys.executable, "-c", "from pathlib import Path;import sys;Path(sys.argv[1]).touch();sys.exit(17)", str(called)]
    directory = tmp_path / "environment"
    if entrypoint == "native":
        directory.mkdir()
        adapter = DaytonaEnvironment(
            environment_dir=directory, environment_name="source", session_id="trial",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=TaskEnvironmentConfig(docker_image="ubuntu:24.04", cpus=1, memory_mb=1024, storage_mb=1024),
            provisioning="declarative", create_admission_command=command, client_factory=lambda: provider,
        )
        with pytest.raises(ExceptionGroup) as error:
            async with asyncio.TaskGroup() as services:
                await adapter.start(False, service_tasks=services)
        assert error.value.subgroup(RuntimeError) is not None
    else:
        environment = EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="ubuntu@sha256:" + "a" * 64
        )
        task = TaskSpec(
            id="quota-boundary", context=ConversationInput(events=(TextMessage(role="user", content="Do the task"),)),
            source=Source(dataset="fixture", revision="pin", row="0", importer_revision="1"),
            answer_type=AnswerType.TEXT, answer_format=PlainText(),
            environment_requirements=environment,
            grader=ScriptGrader(argv=("true",), environment=environment,
                reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),))),
        )
        write_harbor_task(task, directory, INITIAL_REQUIREMENTS, docker_build=DockerBuildExecution.DECLARATIVE)
        monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: provider)
        trial = await Trial.create(TrialConfig(
            task=TaskConfig(path=directory), trial_name="candidate", trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="nop", model_name=None),
            environment=EnvironmentConfig(type="daytona", kwargs={
                "provisioning": "declarative", "create_admission_command": command,
            }),
        ), host_capabilities=HostCapabilities(daytona=DaytonaConnection("fixture", "https://offline.invalid", "us")))
        result = await trial.run()
        assert result.exception_info.exception_type == "RuntimeError"
    assert called.exists() and provider.creates == 0
