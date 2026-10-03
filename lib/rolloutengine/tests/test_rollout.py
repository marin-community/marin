# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task execution, grading, and exact-token evidence through the public engine."""

import asyncio
import json
import math
import threading
from dataclasses import dataclass, field

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, MachineSpec, Result, ShellSimBuiltins
from taskcompendium.environment import (
    ArtifactKind,
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
from taskcompendium.grading import Outcome, numeric_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutInterrupted,
    RolloutOperation,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.parquet import read_tasks, write_tasks


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


@pytest.mark.parametrize(
    "script,status,reward",
    [
        ("echo 1 > /logs/verifier/reward.txt; exit 1", Outcome.GRADED, 1.0),
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
