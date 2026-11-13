# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original converted TaskSpecs cross the production trial and grading boundaries."""

import json
import shutil
import os
import tarfile
from decimal import Decimal
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from shellbox.machine import Backend
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    FileReward,
    RewardFile,
    RewardFileFormat,
    TaskSpec,
    TextMessage,
    verifyit_spec,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.grading import SPEC_PATH
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import BASE_URL_ENV
from verifyit.spec import parse_spec

from harbor.agents import taskcompendium as agents
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial import taskcompendium as adapter
from harbor.trial.trial import Trial
from experiments.post_training.task_curation.tests.review_server import (
    JudgeReply,
    running_review_server,
)
from lib.taskcompendium.tests.test_runtime import LocalGradingMachine

FIXTURES = Path(__file__).parents[4] / "experiments/post_training/task_curation/tests/fixtures/harbor_task_specs"


@dataclass
class LockedRuntime:
    root: Path
    bin_dir: Path = field(default_factory=lambda: Path(sys.executable).parent)
    mounts: tuple = ()
    env: dict = field(default_factory=dict)

    def ensure_built(self):
        pass


class FileMachine(LocalGradingMachine):
    async def upload(self, source, target):
        await super().upload(source, target)
        shutil.copystat(source, self.local(target))

    async def run(self, command):
        result = await super().run(command)
        if command.argv[:2] == ("tar", "-xf"):
            # Emulate the Linux guest's exact PAX metadata on the fake host filesystem.
            with tarfile.open(self.local(command.argv[2])) as archive:
                for member in archive:
                    if "mtime" in member.pax_headers:
                        nanos = int(Decimal(member.pax_headers["mtime"]) * 1_000_000_000)
                        os.utime(self.local(member.name), ns=(nanos, nanos))
        return result


@dataclass
class Machines:
    """Machine boundary: use real verifier subprocesses, with guest paths rooted in tmp_path."""

    root: Path
    backend = Backend.LOCAL
    cgroup_parent: Path = Path("/fixture/cgroup")
    machines: list = field(default_factory=list)
    specifications: list = field(default_factory=list)
    requirements: list = field(default_factory=list)

    def check_environment(self, requirements, spec):
        self.requirements.append(requirements)

    async def create(self, spec):
        root = self.root / f"machine-{len(self.machines)}"
        (root / spec.workdir.lstrip("/")).mkdir(parents=True)
        machine = FileMachine(root=root, workdir=spec.workdir, env=dict(spec.env))
        self.machines.append(machine)
        self.specifications.append(spec)
        return machine


def original_task(name):
    return TaskSpec.model_validate_json((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "agent_name,reward",
    [("task-spec-chat", 0.0), ("task-spec-chat", 1.0), ("oracle", 0.0), ("oracle", 1.0), ("nop", 0.0)],
)
@pytest.mark.parametrize("name", ["tau", "tau-rlvr1-repaired", "tau-rlvr2-repaired", "language-mixing"])
@pytest.mark.asyncio
async def test_converted_judge_task_preserves_contract_through_production_trial(
    tmp_path, monkeypatch, name, reward, agent_name
):
    task = original_task(name)
    original = task.model_dump_json()
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    assert json.loads((directory / "instruction.md").read_text()) == conversation_messages(task.context.events)
    assert TaskSpec.model_validate_json((directory / "task-spec.json").read_text()) == task
    assert not list(directory.rglob("Dockerfile"))
    machines = Machines(tmp_path / "sandboxes")
    monkeypatch.setattr(adapter, "LocalMachineFactory", lambda **kwargs: machines)
    # Frozen locks are an external artifact boundary; these tests do not qualify their availability.
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: True)
    locks = []

    def locked_runtime(path):
        locks.append(path)
        return LockedRuntime(tmp_path)

    monkeypatch.setattr("taskcompendium.runtime.environment.local_runtime", locked_runtime)
    candidate_requests = []
    if task.final_tools:
        candidate = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "candidate-call",
                    "type": "function",
                    "function": {"name": task.final_tools[0].name, "arguments": "{}"},
                }
            ],
        }
        finish = "tool_calls"
    else:
        candidate = {"role": "assistant", "content": "A candidate response."}
        finish = "stop"

    def candidate_response(request):
        candidate_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "candidate-response",
                "object": "chat.completion",
                "created": 0,
                "model": "candidate",
                "choices": [{"index": 0, "message": candidate, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            },
        )

    if agent_name == "oracle":
        (directory / "solution").mkdir()
        (directory / "solution/response.json").write_text(json.dumps([candidate]))

    real_client = agents.AsyncOpenAI
    monkeypatch.setattr(
        agents,
        "AsyncOpenAI",
        lambda **kwargs: real_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(candidate_response)),
        ),
    )
    with running_review_server() as server:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        judge = verifyit_spec(task.grader)
        label = next(label for label, score in judge.label_scores.items() if score == reward)
        server.judge_replies.append(JudgeReply(label))

        async def relay_boundary(machine, stack, service_tasks, *, judge, candidate):
            assert judge.model == verifyit_spec(task.grader).model
            assert candidate is None
            # The fake guest's loopback port maps to the real HTTP judge fixture.
            machines.machines[-1].env[BASE_URL_ENV] = endpoint

        monkeypatch.setattr(adapter, "start_model_relays", relay_boundary)
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=directory),
                trial_name="task-spec",
                trials_dir=tmp_path / "trials",
                agent=AgentConfig(
                    name=agent_name,
                    model_name="openai/candidate" if agent_name == "task-spec-chat" else None,
                    kwargs=(
                        {
                            "api_base": "https://candidate.fixture/v1",
                            "max_turns": 1,
                            "max_tokens": 128,
                        }
                        if agent_name == "task-spec-chat"
                        else {}
                    ),
                ),
                environment=EnvironmentConfig(kwargs={"model_endpoint": endpoint, "model_name": judge.model}),
            )
        )
        result = await trial.run()
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
        if agent_name != "nop":
            assert len(server.requests) == 1
            assert server.requests[0]["model"] == judge.model
            assert judge.references[0] in server.requests[0]["messages"][-1]["content"]
        else:
            assert machines.machines[-1].local("/app/answer.txt").read_bytes() == b""
    if agent_name == "task-spec-chat":
        assert candidate_requests[0]["messages"] == conversation_messages(task.context.events)
    else:
        assert candidate_requests == []
    if agent_name == "task-spec-chat":
        assert candidate_requests[0].get("tools", []) == [
            {"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in task.final_tools
        ]
    assert locks == [task.grader.environment.packages_lock]
    assert len(machines.machines) == 1  # No synthetic actor environment for either source.
    selected = machines.specifications[0]
    assert (selected.cpus, selected.memory_mb, selected.storage_mb) == (1, 2048, 2048)
    machine = machines.machines[0]
    assert machine.closed
    assert parse_spec(machine.local(SPEC_PATH).read_text()) == verifyit_spec(task.grader)
    verdict = json.loads((trial.paths.verifier_dir / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("scored", reward)
    assert (trial.paths.verifier_dir / "verdict.json").read_bytes() == machine.local(
        "/logs/verifier/verdict.json"
    ).read_bytes()
    assert task.model_dump_json() == original
    assert all(
        resource.path not in {str(path.relative_to(machine.root)) for path in machine.root.rglob("*")}
        for resource in task.resources.oracle
    )


@pytest.mark.asyncio
async def test_tau_missing_original_lock_stops_before_candidate_or_machine(tmp_path, monkeypatch):
    task = original_task("tau")
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: False)
    allocations = []
    monkeypatch.setattr(adapter, "LocalMachineFactory", lambda **kwargs: allocations.append(kwargs))
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="missing-lock",
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="task-spec-empty"),
        )
    )
    result = await trial.run()
    assert result.exception_info.exception_type == "FileNotFoundError"
    assert task.grader.environment.packages_lock in result.exception_info.exception_message
    assert result.verifier_result is None
    assert allocations == []


@pytest.mark.asyncio
async def test_environmentless_grader_runs_inside_declared_limits_without_actor(tmp_path, monkeypatch):
    task = exact_answer_task(
        RawRow("text", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), {}),
        prompt="What is six plus six?",
        answers=("12",),
        ignore_case=False,
    )
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    response = tmp_path / "response.json"
    response.write_text(json.dumps([{"role": "assistant", "content": "12"}]))
    machines = Machines(tmp_path / "sandboxes")
    monkeypatch.setattr(adapter, "LocalMachineFactory", lambda **kwargs: machines)
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="text",
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="task-spec-response", kwargs={"response_path": str(response)}),
        )
    )
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(machines.machines) == 1
    selected = machines.specifications[0]
    assert (selected.cpus, selected.memory_mb, selected.storage_mb) == (1, 2048, 2048)
    assert machines.machines[0].closed
    assert TaskSpec.model_validate_json(machines.machines[0].local("/tests/task-spec.json").read_text()) == task


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.asyncio
async def test_actor_files_and_private_grader_remain_in_their_declared_roles(tmp_path, monkeypatch, exit_code):
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="fixture-lock", working_directory="/app"
    )
    script = (
        b"import json,pathlib,stat,sys\np=pathlib.Path(sys.argv[1])\n"
        b"assert p.read_bytes()==b'41\\n'\nassert stat.S_IMODE(p.stat().st_mode)==0o750\n"
        b"assert p.stat().st_mtime_ns==123000000456\n"
        b"assert pathlib.Path(sys.argv[2]).read_text()=='42'\n"
        b"pathlib.Path(sys.argv[3]).write_text(json.dumps({'status':'scored','reward':1,'detail':{}}))\n"
        + f"raise SystemExit({exit_code})\n".encode()
    )
    public = inline_resource("app/value.txt", b"41\n").model_copy(update={"mode": "0750", "mtime_ns": 123000000456})
    task = TaskSpec(
        id="files",
        source=Source(dataset="fixture", revision="pin", row="0", importer_revision="1"),
        context=ConversationInput(
            events=(TextMessage(role="user", content="Read /app/value.txt and return its successor."),)
        ),
        environment_requirements=environment,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        resources=ResourceGroups(worker=(public,), verifier=(inline_resource("check.py", script),)),
        grader=ScriptGrader(
            argv=("python3", "/tests/check.py", "/app/value.txt", "/app/answer.txt", "/logs/verifier/verdict.json"),
            cwd="/app",
            environment=environment,
            answer_path="/app/answer.txt",
            reward=FileReward(
                files=(RewardFile(path="/logs/verifier/verdict.json", format=RewardFileFormat.VERDICT_JSON),)
            ),
        ),
    )
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    response = tmp_path / "response.json"
    response.write_text(json.dumps([{"role": "assistant", "content": "42"}]))
    machines = Machines(tmp_path / "sandboxes")
    monkeypatch.setattr(adapter, "LocalMachineFactory", lambda **kwargs: machines)
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: True)
    monkeypatch.setattr("taskcompendium.runtime.environment.local_runtime", lambda path: LockedRuntime(tmp_path))
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="files",
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="task-spec-response", kwargs={"response_path": str(response)}),
        )
    )
    result = await trial.run()
    if exit_code:
        assert result.exception_info.exception_type == "VerifierRuntimeError"
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": 1.0}
    actor, verifier = machines.machines
    assert actor.local("/app/value.txt").read_bytes() == b"41\n"
    assert not actor.local("/tests/check.py").exists()
    assert verifier.local("/tests/check.py").read_bytes() == script
    assert actor.closed and verifier.closed
    assert all((spec.cpus, spec.memory_mb, spec.storage_mb) == (1, 2048, 2048) for spec in machines.specifications)
