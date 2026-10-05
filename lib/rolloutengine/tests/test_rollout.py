# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task execution, grading, and exact-token evidence through the public engine."""

import asyncio
import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass, field

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, MachineSpec, Result, ShellSimBuiltins
from taskcompendium.environment import (
    ArtifactKind,
    EnvironmentAsset,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    FileReward,
    HealthcheckSpec,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.grading import numeric_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    Source,
    StageRewardStrategy,
    StageVerifierSpec,
    TaskSpec,
    TaskStage,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.parquet import read_tasks, write_tasks
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from rolloutengine.assets import cached_asset
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelResponseRejected,
    ModelTurn,
    RejectedModelResponse,
    RolloutInterrupted,
    RolloutOperation,
    SessionStart,
    Transition,
)
from rolloutengine.engine import ShellboxRolloutEngine


async def run_task(runner: ShellboxRolloutEngine, task: TaskSpec):
    return await runner.run(task)


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


@dataclass
class RecordingShellSimFactory:
    machines: list = field(default_factory=list)

    async def create(self, spec):
        machine = await ShellSimMachineFactory().create(spec)
        self.machines.append(machine)
        return machine


def arithmetic_task() -> TaskSpec:
    return TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(12, tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def engine(model, factories) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(
        model.complete,
        factories,
        max_turns=3,
        command_timeout=5,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_parquet_task_produces_private_grade_and_training_tokens(tmp_path, answer, reward):
    path = str(tmp_path / "tasks.parquet")
    write_tasks(path, [arithmetic_task()])
    model = ReplayModel([{"role": "assistant", "content": answer}])
    result = await engine(model, {}).run(next(read_tasks(path)))
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert result.prompt_token_ids == (10, 11)
    assert result.response_token_ids == (20,)
    assert result.loss_mask == (1,)
    assert result.logprobs == (-0.5,)
    assert "expected" not in json.dumps(model.requests[0].messages)


async def test_executable_answer_task_keeps_submission_instruction_and_shell_tool():
    task = arithmetic_task().model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)})
    model = ReplayModel([{"role": "assistant", "content": "12"}])

    result = await run_task(engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}), task)

    assert result.grade.reward == 1.0
    assert model.requests[0].messages[-1] == {"role": "user", "content": "Give your answer as plain text."}
    assert model.requests[0].options["tools"][0]["function"]["name"] == "shell"


async def test_executable_answer_call_keeps_submission_tool_and_finishes():
    task = arithmetic_task().model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)})
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "answer",
                        "type": "function",
                        "function": {"name": "submit_answer", "arguments": '{"answer":"12"}'},
                    }
                ],
            }
        ]
    )
    runner = ShellboxRolloutEngine(
        model.complete,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=3,
        command_timeout=5,
        convention=SubmissionConvention(id="answer-call", answer_format=AnswerFormat.ANSWER_CALL),
    )

    result = await runner.run(task)

    assert result.grade.reward == 1.0
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == [
        "submit_answer",
        "shell",
    ]


async def test_executable_native_action_keeps_final_and_shell_tools():
    task = arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.NATIVE_ACTION,
            "final_tools": (FunctionDefinition(name="finish", parameters={"type": "object"}),),
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
        }
    )
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}],
            }
        ]
    )

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(task)

    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["finish", "shell"]
    assert result.response_token_ids == (20,)


@pytest.mark.parametrize(
    "runtime_inputs",
    [
        {"environment_requirements": EnvironmentRequirements(working_directory="/workspace")},
        {"interaction_tools": (FunctionDefinition(name="run", parameters={"type": "object"}),)},
        {"output_paths": ("/app/submission.py",)},
    ],
)
async def test_rollout_rejects_unsupported_runtime_inputs(runtime_inputs):
    task = arithmetic_task().model_copy(update=runtime_inputs)

    with pytest.raises(ValueError, match="machine inputs"):
        await engine(ReplayModel([]), {}).run(task)


async def test_unknown_executable_tool_returns_an_observation():
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "other", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(file_task())

    assert result.grade.reward == 0.0
    assert json.loads(model.requests[1].messages[-1]["content"])["error"]


async def test_malformed_assistant_message_is_a_graded_model_result():
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "not-json"},
                    }
                ],
            }
        ]
    )

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(file_task())

    assert result.grade.reward == 0.0
    assert result.metrics == {"invalid_assistant_message": 1.0}


def file_task() -> TaskSpec:
    verifier = ShellVerifierSpec(
        argv=("sh", "/private/grade.sh"),
        files=(
            EnvironmentFile(
                path="/private/grade.sh", content=b'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi'
            ),
        ),
        timeout=5,
    )
    return arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
        }
    )


async def test_shellbox_tools_persist_files_and_mask_observations():
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps(
                                {"command": "test ! -f /private/grade.sh && echo 12 > /workspace/answer"}
                            ),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )
    result = await run_task(engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}), file_task())
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.response_token_ids == (20, 90, 91, 21)
    assert result.loss_mask == (1, 0, 0, 1)
    assert result.logprobs == (-0.5, 0.0, 0.0, -0.5)
    observation = model.requests[1].messages[-1]
    assert observation["tool_call_id"] == "call-1"
    assert json.loads(observation["content"])["exit_code"] == 0


@pytest.mark.parametrize("completed_turns", [0, 1])
async def test_context_limit_grades_only_completed_shell_operations(completed_turns):
    class LimitedModel(ReplayModel):
        async def complete(self, request):
            if len(self.requests) == completed_turns:
                raise GenerationLimitReached((*request.prefix_token_ids, 90, 91))
            return await super().complete(request)

    model = LimitedModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "write-answer",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"echo 12 > /workspace/answer"}'},
                    }
                ],
            }
        ]
    )
    result = await run_task(engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}), file_task())
    assert result.stop_reason == "length"
    if completed_turns:
        assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
        assert result.response_token_ids == (20,)
        assert result.loss_mask == (1,)
        assert result.logprobs == (-0.5,)
        assert result.messages[-1]["role"] == "assistant"
    else:
        assert (result.grade.status, result.grade.reward) == (Outcome.UNAVAILABLE, None)
        assert result.prompt_token_ids == (90, 91)
        assert result.response_token_ids == result.loss_mask == result.logprobs == ()


async def test_failed_shell_grader_has_no_reward():
    task = file_task().model_copy(
        update={
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL, parameters_json=ShellVerifierSpec(argv=("false",), timeout=5).model_dump_json()
            )
        }
    )
    result = await run_task(
        engine(
            ReplayModel([{"role": "assistant", "content": "Completed."}]),
            {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        ),
        task,
    )
    assert (result.grade.status, result.grade.reward) == (Outcome.INFRA_ERROR, None)


async def test_agent_deadline_preserves_completed_tokens_and_closes_the_machine():
    factory = RecordingShellSimFactory()
    machines = factory.machines

    class StalledModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                await asyncio.Future()
            return await super().complete(request)

    model = StalledModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "write",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command": "echo 12 > /workspace/answer"}'},
                    }
                ],
            }
        ]
    )
    task = file_task().model_copy(update={"agent_timeout": 1.0})
    with pytest.raises(RolloutInterrupted) as failure:
        await run_task(engine(model, {EnvironmentKind.SHELLSIM: factory}), task)
    assert isinstance(failure.value.__cause__, TimeoutError)
    assert failure.value.operation == RolloutOperation.MODEL
    rollout = failure.value.rollout
    assert rollout.response_token_ids == (20,)
    assert rollout.loss_mask == (1,)
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    with pytest.raises(RuntimeError, match="closed"):
        await machines[0].run(Command(("true",)))


@pytest.mark.parametrize("completed_turns", [0, 1])
async def test_advance_failure_retains_pending_turn_without_training_data(completed_turns):
    class Session:
        def __init__(self):
            self.calls = 0

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Do the task."},), {})

        async def advance(self, turn: ModelTurn):
            if self.calls == completed_turns:
                raise OSError("Guest failed during command")
            self.calls += 1
            return Transition(False, ({"role": "user", "content": "Continue."},))

        async def grade(self, messages):
            raise AssertionError("An advance failure must not invoke grading")

        async def close(self):
            pass

    class Model(ReplayModel):
        async def complete(self, request):
            turn = await super().complete(request)
            return ModelTurn(
                turn.message,
                turn.prompt_token_ids,
                turn.response_token_ids,
                turn.logprobs,
                turn.stop_reason,
                text="exact model text",
                metadata={"request_id": "transport-evidence"},
            )

    model = Model([{"role": "assistant", "content": str(index)} for index in range(2)])
    session = Session()
    task = arithmetic_task().model_copy(
        update={"environment": EnvironmentSpec(kind=EnvironmentKind.NULL, interaction="failure")}
    )
    runner = ShellboxRolloutEngine(
        model.complete,
        {},
        max_turns=3,
        command_timeout=5,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        sessions={"failure": lambda _task: session},
    )
    with pytest.raises(RolloutInterrupted) as failure:
        await runner.run(task)
    rollout = failure.value.rollout
    assert failure.value.operation == RolloutOperation.ADVANCE
    assert isinstance(failure.value.__cause__, OSError)
    assert rollout.messages[-1] == model.messages[completed_turns]
    assert rollout.prompt_token_ids == (10, 11)
    assert rollout.response_token_ids == ((20,) if completed_turns == 0 else (20, 90, 91, 21))
    assert rollout.logprobs == ((-0.5,) if completed_turns == 0 else (-0.5, 0.0, 0.0, -0.5))
    assert rollout.loss_mask == (0,) * len(rollout.response_token_ids)
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.UNAVAILABLE, None)
    assert len(rollout.steps) == completed_turns
    assert rollout.failure is not None
    pending = rollout.failure.diagnostics["pending_turn"]
    assert pending["text"] == "exact model text"
    assert pending["metadata"] == {"request_id": "transport-evidence"}
    assert pending["response_token_ids"] == (20 + completed_turns,)
    serialized = json.loads(json.dumps(asdict(rollout)))
    assert serialized["failure"]["diagnostics"]["pending_turn"]["response_token_ids"] == [20 + completed_turns]


async def test_later_stage_advance_failure_preserves_graded_prefix_only():
    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv == ("sh", "-c", "fail-advance"):
                raise OSError("Guest unavailable")
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return Machine(await ShellSimMachineFactory().create(spec))

    first = file_task()
    task = first.model_copy(
        update={
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
            ),
            "stages": (
                TaskStage(name="first", verifier=first.verifier),
                TaskStage(
                    name="second",
                    verifier=first.verifier,
                    context=ConversationInput(events=(TextMessage(role="user", content="Continue."),)),
                ),
            ),
        }
    )
    replies = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": str(index),
                    "type": "function",
                    "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                }
            ],
        }
        for index, command in enumerate(("echo 12 > /workspace/answer", "fail-advance"))
    ]
    model = ReplayModel([replies[0], {"role": "assistant", "content": "Done."}, replies[1]])
    with pytest.raises(RolloutInterrupted) as failure:
        await engine(model, {EnvironmentKind.SHELLSIM: Factory()}).run(task)
    rollout = failure.value.rollout
    assert failure.value.operation == RolloutOperation.ADVANCE
    assert rollout.response_token_ids == (20, 90, 91, 21, 90, 91, 22)
    assert rollout.loss_mask == (1, 0, 0, 1, 0, 0, 0)
    assert len(rollout.steps) == 2
    assert rollout.steps[0].transition.grade is not None
    assert rollout.steps[0].transition.grade.status == Outcome.GRADED
    assert rollout.grade.diagnostics["stages"][1]["status"] == Outcome.UNAVAILABLE
    assert rollout.failure is not None
    assert rollout.failure.diagnostics["pending_turn"]["response_token_ids"] == (22,)


@pytest.mark.parametrize(
    "script,status,reward",
    [
        ("echo 1 > /logs/verifier/reward.txt; exit 1", Outcome.GRADED, 1.0),
        ("echo 0.5 > /logs/verifier/reward.json", Outcome.GRADED, 0.5),
        (
            "echo 1 > /logs/verifier/reward.txt; echo '{\"reward\":0}' > /logs/verifier/reward.json",
            Outcome.GRADED,
            0.0,
        ),
        ("true", Outcome.INFRA_ERROR, None),
        ("echo broken > /logs/verifier/reward.json; echo 1 > /logs/verifier/reward.txt", Outcome.INFRA_ERROR, None),
    ],
)
async def test_file_grader_preserves_priority_and_rejects_agent_scores(tmp_path, script, status, reward):
    verifier = ShellVerifierSpec(
        argv=("sh", "/tests/test.sh"),
        files=(EnvironmentFile(path="/tests/test.sh", content=script.encode()),),
        timeout=5,
        reward=FileReward(
            files=(
                RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),
                RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),
            )
        ),
    )
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/logs/verifier/reward.txt", content=b"1"),),
            ),
            "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
        }
    )
    path = str(tmp_path / "tasks.parquet")
    write_tasks(path, iter([task]))
    result = await run_task(
        engine(
            ReplayModel([{"role": "assistant", "content": "Completed."}]),
            {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        ),
        next(read_tasks(path)),
    )
    assert (result.grade.status, result.grade.reward) == (status, reward)


async def test_model_failure_releases_the_shellbox_machine():
    factory = RecordingShellSimFactory()
    machines = factory.machines

    class FailedModel:
        async def complete(self, _request):
            raise ConnectionError("Inference endpoint unavailable")

    with pytest.raises(RolloutInterrupted) as failure:
        await run_task(engine(FailedModel(), {EnvironmentKind.SHELLSIM: factory}), file_task())
    assert isinstance(failure.value.__cause__, ConnectionError)
    with pytest.raises(RuntimeError, match="closed"):
        await machines[0].run(Command(argv=("true",)))


async def test_immutable_assets_run_before_setup_and_reject_invalid_identity(tmp_path):
    payload = f"#!/bin/sh\n# {tmp_path.name}\nprintf 12 > /workspace/answer\n".encode()
    source = tmp_path / "input.sh"
    source.write_bytes(payload)
    asset = EnvironmentAsset(
        path="/workspace/input.sh",
        uri=str(source),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        mode=0o755,
    )
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                assets=(asset,),
                setup=(EnvironmentCommand(argv=(asset.path,), timeout=5),),
            )
        }
    )
    parquet = str(tmp_path / "tasks.parquet")
    write_tasks(parquet, [task])
    task = next(read_tasks(parquet))
    for _ in range(2):
        result = await run_task(
            engine(
                ReplayModel([{"role": "assistant", "content": "Completed."}]),
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
            ),
            task,
        )
        assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
        source.unlink(missing_ok=True)

    source.write_bytes(b"different file")
    wrong_identity = asset.model_copy(update={"sha256": "0" * 64})
    changed = task.model_copy(update={"environment": task.environment.model_copy(update={"assets": (wrong_identity,)})})
    with pytest.raises(RolloutInterrupted) as failure:
        await run_task(engine(ReplayModel([]), {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}), changed)
    assert failure.value.operation == RolloutOperation.START
    assert failure.value.rollout.grade.status == Outcome.UNAVAILABLE
    assert isinstance(failure.value.__cause__, ValueError)


def test_asset_cache_rejects_corruption_without_replacing_evidence(tmp_path):
    source = tmp_path / "input"
    content = b"immutable dependency"
    source.write_bytes(content)
    asset = EnvironmentAsset(
        path="/input", uri=str(source), sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )
    cache = tmp_path / "cache"
    stored = cached_asset(asset, cache)
    source.unlink()
    assert cached_asset(asset, cache).read_bytes() == content
    corrupt = b"x" * len(content)
    stored.write_bytes(corrupt)
    with pytest.raises(ValueError, match="Cached task asset differs"):
        cached_asset(asset, cache)
    assert stored.read_bytes() == corrupt


async def test_machine_setup_failure_releases_resources_and_retains_an_empty_record():
    factory = RecordingShellSimFactory()
    machines = factory.machines

    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                setup=(EnvironmentCommand(argv=("false",), timeout=5),),
            )
        }
    )
    with pytest.raises(RolloutInterrupted) as failure:
        await run_task(engine(ReplayModel([]), {EnvironmentKind.SHELLSIM: factory}), task)
    assert failure.value.operation == RolloutOperation.START
    assert failure.value.rollout.task_id == task.id
    assert failure.value.rollout.grade.status == Outcome.UNAVAILABLE
    assert failure.value.rollout.response_token_ids == ()
    assert failure.value.rollout.steps == ()
    with pytest.raises(RuntimeError, match="closed"):
        await machines[0].run(Command(("true",)))


@pytest.mark.parametrize(
    "phase",
    [
        "upload",
        "healthcheck",
        "command_timeout",
        "attempt_startup",
        "model",
        "grade",
        "cancel",
        "cleanup_cancel",
        "cancel_twice",
    ],
)
async def test_startup_and_attempt_deadlines_release_machines_without_partial_training_data(phase):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    machines = []

    async def stall():
        loop.call_soon_threadsafe(entered.set)
        await asyncio.Future()

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv == ("command-timeout",):
                loop.call_soon_threadsafe(entered.set)
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            if command.argv == ("stall",):
                await stall()
            return await self.machine.run(command)

        async def upload(self, source, target):
            if phase in {"upload", "attempt_startup", "cancel", "cleanup_cancel", "cancel_twice"}:
                await stall()
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            if phase in {"cleanup_cancel", "cancel_twice"}:
                loop.call_soon_threadsafe(closing.set)
                await asyncio.to_thread(release.wait)
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(spec)
            machines.append(machine)
            return Machine(machine)

    class Model(ReplayModel):
        async def complete(self, request):
            if phase == "model":
                await stall()
            return await super().complete(request)

    environment = EnvironmentSpec(
        kind=EnvironmentKind.SHELLSIM,
        files=(EnvironmentFile(path="/input", content=b"input"),),
        setup=(EnvironmentCommand(argv=("command-timeout",), timeout=5),) if phase == "command_timeout" else (),
        startup_timeout=1 if phase in {"upload", "healthcheck", "cleanup_cancel"} else 10,
        healthcheck=(
            HealthcheckSpec(
                command=EnvironmentCommand(argv=("stall",), timeout=5),
                interval=0,
                start_period=0,
                start_interval=0,
                retries=1,
            )
            if phase == "healthcheck"
            else None
        ),
    )
    task = file_task().model_copy(
        update={
            "environment": environment,
            "attempt_timeout": 1 if phase in {"attempt_startup", "model", "grade"} else 10,
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(argv=("stall",), timeout=5).model_dump_json(),
            ),
        }
    )
    model = Model([{"role": "assistant", "content": "Done."}])
    runner = engine(model, {EnvironmentKind.SHELLSIM: Factory()})
    pending = asyncio.create_task(run_task(runner, task))
    await asyncio.wait_for(entered.wait(), timeout=5)
    if phase in {"cancel", "cleanup_cancel", "cancel_twice"}:
        try:
            if phase == "cleanup_cancel":
                await asyncio.wait_for(closing.wait(), timeout=5)
            pending.cancel()
            if phase == "cancel_twice":
                await asyncio.wait_for(closing.wait(), timeout=5)
                pending.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(RolloutInterrupted) as failure:
            await pending
        assert isinstance(failure.value.__cause__, TimeoutError)
        assert failure.value.operation == (
            RolloutOperation.START if phase in {"upload", "healthcheck", "command_timeout"} else RolloutOperation.ATTEMPT
        )
        assert failure.value.rollout.grade.status == Outcome.UNAVAILABLE
        assert failure.value.rollout.response_token_ids == failure.value.rollout.loss_mask == ()
        assert failure.value.rollout.steps == ()
        if phase == "grade":
            assert len(model.requests) == 1
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


async def test_attempt_deadline_and_cancellation_wait_for_machine_cleanup():
    loop = asyncio.get_running_loop()
    close_started = loop.create_future()
    machine = await ShellSimMachineFactory().create(MachineSpec(source=ShellSimBuiltins()))

    class Machine:
        async def upload(self, _source, _target):
            raise OSError("Startup failed")

        async def close(self):
            cleanup_loop = asyncio.get_running_loop()
            release = asyncio.Event()
            loop.call_soon_threadsafe(close_started.set_result, (cleanup_loop, release))
            await release.wait()
            await machine.close()

    class Factory:
        async def create(self, _spec):
            return Machine()

    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/input", content=b"input"),),
            ),
            # The positive deadline expires before the first suspension in machine cleanup.
            "attempt_timeout": math.nextafter(0.0, 1.0),
        }
    )
    runner = engine(ReplayModel([]), {EnvironmentKind.SHELLSIM: Factory()})
    pending = asyncio.create_task(run_task(runner, task))
    cleanup_loop, release = await asyncio.wait_for(close_started, timeout=5)
    pending.cancel()
    cleanup_loop.call_soon_threadsafe(release.set)
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(RuntimeError, match="closed"):
        await machine.run(Command(("true",)))


@pytest.mark.parametrize("answer,expected_reward", [(b"\x00\xff\r\n", 1.0), (b"incorrect", 0.0)])
async def test_separate_grader_receives_binary_artifacts_in_a_fresh_machine(tmp_path, answer, expected_reward):
    verifier = ShellVerifierSpec(
        argv=("/private/grade.sh",),
        timeout=5,
        reward=ExitCodeReward(),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.SHELLSIM,
            setup=(EnvironmentCommand(argv=("sh", "-c", "echo clean > /workspace/baseline"), timeout=5),),
        ),
        collect=(EnvironmentCommand(argv=("cp", "/workspace/answer", "/workspace/submission"), timeout=5),),
        artifacts=(
            VerifierArtifact(source="/workspace/submission", target="/workspace/submission", kind=ArtifactKind.FILE),
        ),
        files=(
            EnvironmentFile(path="/private/expected", content=b"\x00\xff\r\n"),
            EnvironmentFile(
                path="/private/grade.sh",
                content=(
                    b'#!/bin/sh\ntest "$(cat /workspace/baseline)" = clean && '
                    b"cmp /private/expected /workspace/submission"
                ),
                mode=0o755,
            ),
        ),
    )
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/workspace/input", content=answer),),
            ),
            "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
        }
    )
    path = str(tmp_path / "tasks.parquet")
    write_tasks(path, iter([task]))
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "edit",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps(
                                {"command": "test ! -f /private/expected && cp input answer && echo tainted > baseline"}
                            ),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )
    factory = RecordingShellSimFactory()
    machines = factory.machines

    result = await run_task(engine(model, {EnvironmentKind.SHELLSIM: factory}), next(read_tasks(path)))
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected_reward)
    assert len(machines) == 2
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(argv=("true",)))


@pytest.mark.parametrize("rejected_stage", [0, 1])
async def test_staged_first_response_rejection_keeps_grade_without_invented_action(rejected_stage):
    first = file_task()
    stages = (
        TaskStage(name="first", verifier=first.verifier),
        TaskStage(
            name="second",
            verifier=VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(argv=("false",), reward=ExitCodeReward(), timeout=5).model_dump_json(),
            ),
        ),
    )
    task = first.model_copy(
        update={
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
            ),
            "stages": stages,
        }
    )
    evidence = RejectedModelResponse(
        request={"prompt": [10, 11]},
        request_sha256="authored-request",
        response_body_base64="e30=",
        response_sha256="authored-response",
    )
    prefix = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "write",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo 12 > /workspace/answer"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Done."},
        ]
    )
    requests = []

    async def model(request):
        requests.append(request)
        if rejected_stage == 1 and len(requests) <= 2:
            return await prefix.complete(request)
        raise ModelResponseRejected("Authored received-response rejection", evidence)

    runner = ShellboxRolloutEngine(
        model,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=3,
        command_timeout=5,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )
    with pytest.raises(RolloutInterrupted) as caught:
        await runner.run(task)
    record = caught.value.rollout
    assert caught.value.operation == RolloutOperation.MODEL
    assert record.grade.status == Outcome.GRADED
    assert record.grade.reward == (0 if rejected_stage == 0 else 0.5)
    assert len(requests) == (1 if rejected_stage == 0 else 3)
    assert len(record.steps) == (0 if rejected_stage == 0 else 2)
    assert record.response_token_ids == (() if rejected_stage == 0 else (20, 90, 91, 21))
    assert record.loss_mask == (() if rejected_stage == 0 else (1, 0, 0, 1))
    assert record.failure is not None
    assert record.failure.diagnostics["rejected_response"]["response_body_base64"] == "e30="
    if rejected_stage == 1:
        assert record.steps[-1].transition.grade is not None
        assert record.steps[-1].transition.grade.reward == 1
        assert record.steps[-1].transition.reward == record.grade.reward
