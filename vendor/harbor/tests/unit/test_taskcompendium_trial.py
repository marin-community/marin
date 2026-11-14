# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original converted TaskSpecs cross the production trial and grading boundaries."""

import json
import inspect
import shutil
import os
import tarfile
from decimal import Decimal
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import tomlkit
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Backend, UnsupportedMachineSpec
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.importers.evalchemy.financebench import convert_finance
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    ConversationTrace,
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
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import BASE_URL_ENV

from harbor.agents import taskcompendium as agents
from harbor.environments import local as local_environments
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
from lib.taskcompendium.tests.grading_machines import LocalGradingMachine

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
    prepared_image = False

    async def upload(self, source, target):
        await super().upload(source, target)
        shutil.copystat(source, self.local(target))

    async def run(self, command):
        if self.prepared_image and command.user not in (None, "0", "0:0"):
            raise UnsupportedMachineSpec("Prepared local images currently support only root USER")
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
    disk_pool: object | None = None
    machines: list = field(default_factory=list)
    specifications: list = field(default_factory=list)
    requirements: list = field(default_factory=list)

    def check_environment(self, requirements, spec):
        self.requirements.append(requirements)

    async def create(self, spec):
        assert self.disk_pool is not None
        root = self.root / f"machine-{len(self.machines)}"
        (root / spec.workdir.lstrip("/")).mkdir(parents=True)
        machine = FileMachine(root=root, workdir=spec.workdir, env=dict(spec.env))
        machine.prepared_image = isinstance(spec.source, RegistryImage)
        self.machines.append(machine)
        self.specifications.append(spec)
        return machine


def bind_factory(monkeypatch, machines, disk_root):
    pool = object()
    qualified = []

    async def qualify(root):
        qualified.append(root)
        return pool

    def factory(**kwargs):
        # Bind the real provider signature: forwarding disk_root used to bypass this boundary.
        inspect.signature(LocalMachineFactory).bind(**kwargs)
        assert qualified == [disk_root]
        assert kwargs["disk_pool"] is pool
        machines.disk_pool = pool
        return machines

    monkeypatch.setattr(local_environments, "verified_local_disk_pool", qualify)
    monkeypatch.setattr(local_environments, "LocalMachineFactory", factory)


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
    bind_factory(monkeypatch, machines, tmp_path / "disks")
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
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return list(range(10))

    monkeypatch.setattr(agents.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
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
                            "context_window": 65536,
                            "max_input_tokens": 49152,
                            "request_parameters": {"temperature": 0.7, "top_p": 0.95, "extra_body": {"top_k": 20}},
                            "tokenizer_model": "fixture/candidate", "tokenizer_revision": "f" * 40,
                        }
                        if agent_name == "task-spec-chat"
                        else {}
                    ),
                ),
                environment=EnvironmentConfig(
                    kwargs={"model_endpoint": endpoint, "model_name": judge.model, "disk_root": str(tmp_path / "disks")}
                ),
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
            conversation = ConversationTrace.model_validate_json(
                machines.machines[-1].local("/tests/conversation.json").read_text()
            )
            assert conversation.events[-1].content == ""
    if agent_name == "task-spec-chat":
        assert candidate_requests[0]["messages"] == conversation_messages(task.context.events)
        assert {key: candidate_requests[0][key] for key in ("temperature", "top_p", "top_k", "max_tokens")} == {
            "temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 128,
        }
    else:
        assert candidate_requests == []
    if agent_name == "task-spec-chat":
        assert candidate_requests[0].get("tools", []) == [
            {"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in task.final_tools
        ]
    assert locks == []  # The production judge service does not install SDK packages in the task environment.
    assert len(machines.machines) == 1  # No synthetic actor environment for either source.
    selected = machines.specifications[0]
    assert (selected.cpus, selected.memory_mb, selected.storage_mb) == (1, 2048, 2048)
    machine = machines.machines[0]
    assert machine.closed
    assert TaskSpec.model_validate_json(machine.local("/tests/task-spec.json").read_text()) == task
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
@pytest.mark.parametrize("override", [
    {"tools": []}, {"tool_choice": "none"}, {"parallel_tool_calls": False},
    {"extra_body": {"tools": []}}, {"extra_body": {"chat_template_kwargs": {"tools": []}}},
])
async def test_candidate_sampling_cannot_replace_original_task_tools(tmp_path, monkeypatch, override):
    task = original_task("tau")
    original = task.model_dump_json()
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    calls = []
    monkeypatch.setattr(agents, "AsyncOpenAI", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="sampling and template|task interface"):
        await Trial.create(TrialConfig(
            task=TaskConfig(path=directory), trial_name="tool-replacement", trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="task-spec-chat", model_name="openai/candidate", kwargs={
                "api_base": "https://candidate.fixture/v1", "max_turns": 1, "max_tokens": 128,
                "context_window": 65536, "max_input_tokens": 49152, "request_parameters": override,
                "tokenizer_model": "fixture/candidate", "tokenizer_revision": "f" * 40,
            }),
        ))
    assert calls == []
    assert TaskSpec.model_validate_json((directory / "task-spec.json").read_text()).model_dump_json() == original


@pytest.mark.asyncio
async def test_tau_missing_original_lock_stops_before_candidate_or_machine(tmp_path, monkeypatch):
    task = original_task("tau")
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: False)
    allocations = []
    monkeypatch.setattr(local_environments, "LocalMachineFactory", lambda **kwargs: allocations.append(kwargs))
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="missing-lock",
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(name="nop"),
        )
    )
    result = await trial.run()
    assert result.exception_info.exception_type == "FileNotFoundError"
    assert task.grader.environment.packages_lock in result.exception_info.exception_message
    assert result.verifier_result is None
    assert allocations == []


@pytest.mark.parametrize("resources", [{"gpus": 1}, {"tpu": {"type": "v6e", "topology": "2x4"}}], ids=["gpu", "tpu"])
@pytest.mark.asyncio
async def test_task_spec_local_hardware_policy_rejects_before_allocation(tmp_path, monkeypatch, resources):
    directory = tmp_path / "task"
    write_harbor_task(original_task("language-mixing"), directory, INITIAL_REQUIREMENTS)
    path = directory / "task.toml"
    document = tomlkit.parse(path.read_text())
    document["environment"].update(resources)
    path.write_text(tomlkit.dumps(document))
    allocations = []
    monkeypatch.setattr(local_environments, "LocalMachineFactory", lambda **kwargs: allocations.append(kwargs))
    with pytest.raises(ValueError, match="GPU and TPU allocations are unsupported"):
        await Trial.create(
            TrialConfig(
                task=TaskConfig(path=directory),
                trial_name="unsupported-hardware",
                trials_dir=tmp_path / "trials",
                agent=AgentConfig(name="nop"),
            )
        )
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
    bind_factory(monkeypatch, machines, tmp_path / "disks")
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="text",
            environment=EnvironmentConfig(kwargs={"disk_root": str(tmp_path / "disks")}),
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


@pytest.mark.parametrize(
    "name,model", [("financebench", "selected-judge"), ("financebench", None), ("language-mixing", "conflicting-judge")]
)
@pytest.mark.asyncio
async def test_judge_model_selection_respects_source_delegation(tmp_path, monkeypatch, name, model):
    task = (
        convert_finance(
            RawRow(
                "finance",
                Source(dataset="PatronusAI/financebench", revision="fixture-pin", row="0", importer_revision="1"),
                {"evidence_text": "Revenue was 42 dollars.", "question": "What was revenue?", "answer": "42 dollars"},
            ),
            ConversionContext(inputs={}, grader_environment=None),
        )
        if name == "financebench"
        else original_task(name)
    )
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    original = (directory / "task-spec.json").read_bytes()
    response = tmp_path / "response.json"
    if model == "selected-judge":
        response.write_text(json.dumps([{"role": "assistant", "content": "42 dollars"}]))
    machines = Machines(tmp_path / "sandboxes")
    bind_factory(monkeypatch, machines, tmp_path / "disks")
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: True)
    with running_review_server() as server:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        server.judge_replies.append(JudgeReply("correct"))

        async def relay_boundary(machine, stack, service_tasks, *, judge, candidate):
            assert judge.model == model
            machines.machines[-1].env[BASE_URL_ENV] = endpoint

        monkeypatch.setattr(adapter, "start_model_relays", relay_boundary)
        kwargs = {"model_endpoint": endpoint, "disk_root": str(tmp_path / "disks")}
        if model is not None:
            kwargs["model_name"] = model
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=directory),
                trial_name="judge-model",
                trials_dir=tmp_path / "trials",
                agent=AgentConfig(name="task-spec-response", kwargs={"response_path": str(response)}),
                environment=EnvironmentConfig(kwargs=kwargs),
            )
        )
        result = await trial.run()
        if model == "selected-judge":
            assert result.exception_info is None, result.exception_info
            assert result.verifier_result.rewards == {"reward": 1.0}
            assert server.requests[0]["model"] == model
            assert len(machines.machines) == 1
        else:
            assert result.exception_info.exception_type == "ValueError"
            assert result.verifier_result is None
            assert server.requests == []
            assert machines.machines == []
            assert not response.exists()  # A missing model must fail before candidate execution.
    assert (directory / "task-spec.json").read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider_outage", "input_budget"])
async def test_unserved_candidate_requests_retain_history_without_scoring(tmp_path, monkeypatch, failure):
    task = exact_answer_task(
        RawRow("outage", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), {}),
        prompt="Use the shell, then answer 42.",
        answers=("42",),
        ignore_case=False,
    ).model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                capabilities=("shell", "filesystem"),
                command_semantics=CommandSemantics.LINUX_PROCESS,
                packages_lock="fixture-lock",
                working_directory="/app",
            )
        }
    )
    directory = tmp_path / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    machines = Machines(tmp_path / "sandboxes")
    bind_factory(monkeypatch, machines, tmp_path / "disks")
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: True)
    monkeypatch.setattr("taskcompendium.runtime.environment.local_runtime", lambda path: LockedRuntime(tmp_path))
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 2:
            return httpx.Response(503, json={"error": {"message": "provider unavailable", "type": "server_error"}})
        return httpx.Response(
            200,
            json={
                "id": "shell-turn",
                "object": "chat.completion",
                "created": 0,
                "model": "candidate",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "shell-call",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": json.dumps({"command": "printf 'partial observation\\n'"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            },
        )

    real_client = agents.AsyncOpenAI
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return list(range(49153 if failure == "input_budget" else 10))

    monkeypatch.setattr(agents.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(
        agents,
        "AsyncOpenAI",
        lambda **kwargs: real_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="provider-outage",
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(
                name="task-spec-chat",
                model_name="openai/candidate",
                kwargs={
                    "api_base": "https://candidate.fixture/v1",
                    "max_turns": 3,
                    "max_tokens": 128,
                    "context_window": 65536, "max_input_tokens": 49152,
                    "request_parameters": {}, "tokenizer_model": "fixture/candidate",
                    "tokenizer_revision": ("0" if failure == "input_budget" else "f") * 40,
                },
            ),
            environment=EnvironmentConfig(kwargs={"disk_root": str(tmp_path / "disks")}),
        )
    )
    result = await trial.run()
    assert len(requests) == (0 if failure == "input_budget" else 2)
    assert result.exception_info.exception_type == "RolloutInterrupted"
    assert result.verifier_result is None
    assert not (trial.paths.verifier_dir / "verdict.json").exists()
    trace = ConversationInput.model_validate_json((trial.paths.agent_dir / "model-input.json").read_text())
    messages = conversation_messages(trace.events)
    assert any(
        message["role"] == "tool" and json.loads(message["content"])["stdout"] == "partial observation\n"
        for message in messages
    ) == (failure == "provider_outage")
    assert len(machines.machines) == 1
    assert machines.machines[0].closed


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("packaging", ["lock", "image"])
@pytest.mark.asyncio
async def test_actor_files_and_private_grader_remain_in_their_declared_roles(
    tmp_path, monkeypatch, exit_code, packaging
):
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/app",
        **(
            {"docker_image": "offline.test/task@sha256:" + "a" * 64}
            if packaging == "image"
            else {"packages_lock": "fixture-lock"}
        ),
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
    bind_factory(monkeypatch, machines, tmp_path / "disks")
    monkeypatch.setattr(adapter.StoragePath, "exists", lambda path: True)
    monkeypatch.setattr("taskcompendium.runtime.environment.local_runtime", lambda path: LockedRuntime(tmp_path))
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=directory),
            trial_name="files",
            environment=EnvironmentConfig(kwargs={"disk_root": str(tmp_path / "disks")}),
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
