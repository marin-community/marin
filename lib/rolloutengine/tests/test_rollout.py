# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task execution, grading, and exact-token evidence through the public engine."""

import asyncio
import json
import tarfile
import threading
from dataclasses import dataclass, field, replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, MachineSpec, Result, ShellSimBuiltins
from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    ExternalVerifierSpec,
    FileReward,
    HealthcheckSpec,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerdictReward,
    VerifierArtifact,
)
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
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
from taskcompendium.submission import AnswerCall, PlainText

from rolloutengine.cleanup import finish_cleanup
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutInterrupted,
    RolloutOperation,
    SessionStart,
    Transition,
)
from rolloutengine.engine import ShellboxRolloutEngine


@pytest.mark.asyncio
async def test_resource_cleanup_retains_failure_cause_after_cancellation(tmp_path):
    resource = tmp_path / "resource"
    resource.write_text("open")
    started = asyncio.Event()
    release = asyncio.Event()

    async def close():
        started.set()
        await release.wait()
        resource.unlink()
        raise OSError("The resource closed but its cleanup report failed")

    pending = asyncio.create_task(finish_cleanup(close, timeout=5))
    await asyncio.wait_for(started.wait(), timeout=5)
    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError) as interrupted:
        await pending
    assert not resource.exists()
    assert isinstance(interrupted.value.__cause__, OSError)


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
        verifier=numeric_answer("12", tolerance_abs=0.0, tolerance_rel=0.0),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def engine(model, factories, *, cleanup_timeout=5, sessions=None, max_turns=3) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(
        model.complete,
        factories,
        max_turns=max_turns,
        command_timeout=5,
        cleanup_timeout=cleanup_timeout,
        convention=PlainText(id="plain"),
        sessions=sessions,
    )


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_serialized_task_produces_private_grade_and_training_tokens(answer, reward):
    task = TaskSpec.model_validate_json(arithmetic_task().model_dump_json())
    model = ReplayModel([{"role": "assistant", "content": answer}])
    result = await engine(model, {}).run(task, execution=TaskExecution())
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert result.prompt_token_ids == (10, 11)
    assert result.response_token_ids == (20,)
    assert result.loss_mask == (1,)
    assert result.logprobs == (-0.5,)
    assert "expected" not in json.dumps(model.requests[0].messages)


async def test_executable_answer_task_keeps_submission_instruction_and_shell_tool():
    task = arithmetic_task().model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)})
    model = ReplayModel([{"role": "assistant", "content": "12"}])

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        task, execution=TaskExecution()
    )

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
        cleanup_timeout=5,
        convention=AnswerCall(id="answer-call"),
    )

    result = await runner.run(task, execution=TaskExecution())

    assert result.grade.reward == 1.0
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == [
        "submit_answer",
        "shell",
    ]


async def test_executable_native_action_keeps_final_and_shell_tools():
    task = arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.NATIVE_ACTION,
            "verifier": VerifierSpec(
                kind=VerifierKind.PREDICTED_ACTION,
                parameters_json=json.dumps({"expected_calls": [{"name": "finish", "arguments": {}}]}),
            ),
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

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        task, execution=TaskExecution()
    )

    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["finish", "shell"]
    assert result.response_token_ids == (20,)
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)


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

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        file_task(), execution=TaskExecution()
    )

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

    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        file_task(), execution=TaskExecution()
    )

    assert result.grade.reward == 0.0
    assert result.metrics == {"invalid_assistant_message": 1.0}


def file_task() -> TaskSpec:
    verifier = ShellVerifierSpec(
        argv=("sh", "/private/grade.sh"),
        timeout=5,
    )
    return arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=verifier.model_dump_json(),
                files=(
                    EnvironmentFile(
                        path="/private/grade.sh",
                        content=b'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi',
                    ),
                ),
            ),
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
    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        file_task(), execution=TaskExecution()
    )
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.response_token_ids == (20, 90, 91, 21)
    assert result.loss_mask == (1, 0, 0, 1)
    assert result.logprobs == (-0.5, 0.0, 0.0, -0.5)
    observation = model.requests[1].messages[-1]
    assert observation["tool_call_id"] == "call-1"
    assert json.loads(observation["content"])["exit_code"] == 0


@pytest.mark.parametrize("interruption", [None, "failure", "cancel"])
async def test_custom_session_uses_prepared_machine_and_releases_it_after_session_cleanup(interruption):
    entered = asyncio.Event()
    factory = RecordingShellSimFactory()
    task = arithmetic_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                interaction="file-answer",
                files=(EnvironmentFile(path="/workspace/question", content=b"six plus six"),),
            ),
            "verifier": VerifierSpec(
                kind=VerifierKind.EXTERNAL,
                parameters_json=ExternalVerifierSpec(parameters={"expected": "12"}).model_dump_json(),
            ),
        }
    )

    class FileAnswerSession:
        def __init__(self, task, machine):
            self.task = task
            self.machine = machine

        async def prepare(self):
            result = await self.machine.run(Command(("cat", "question")))
            return SessionStart(({"role": "user", "content": result.stdout.decode()},), {})

        async def advance(self, turn):
            await self.machine.run(Command(("sh", "-c", f"echo {turn.message['content']} > answer")))
            if interruption == "failure":
                raise OSError("Execution failed")
            if interruption == "cancel":
                entered.set()
                await asyncio.Future()
            return Transition(done=True)

        async def grade(self, messages):
            verifier = ExternalVerifierSpec.model_validate_json(self.task.verifier.parameters_json)
            answer = await self.machine.run(Command(("cat", "answer")))
            return GradeResult(Outcome.GRADED, float(answer.stdout.decode().strip() == verifier.parameters["expected"]))

        async def close(self):
            # Session cleanup must retain access to the machine.
            result = await self.machine.run(Command(("test", "-f", "answer")))
            assert result.exit_code == 0

    model = ReplayModel([{"role": "assistant", "content": "12"}])
    runner = engine(
        model,
        {EnvironmentKind.SHELLSIM: factory},
        max_turns=2,
        sessions={"file-answer": FileAnswerSession},
    )
    pending = asyncio.create_task(runner.run(task, execution=TaskExecution()))
    if interruption == "cancel":
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    elif interruption == "failure":
        with pytest.raises(RolloutInterrupted) as failure:
            await pending
        assert failure.value.operation == RolloutOperation.ADVANCE
        assert isinstance(failure.value.__cause__, OSError)
    else:
        result = await pending
        assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
        assert result.response_token_ids == (20,)
        assert result.loss_mask == (1,)
    assert model.requests[0].messages == ({"role": "user", "content": "six plus six"},)
    for machine in factory.machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


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
    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        file_task(), execution=TaskExecution()
    )
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
    result = await engine(
        ReplayModel([{"role": "assistant", "content": "Completed."}]),
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
    ).run(task, execution=TaskExecution())
    assert (result.grade.status, result.grade.reward) == (Outcome.INFRA_ERROR, None)


async def test_one_task_supports_independent_execution_deadlines():
    release = asyncio.Event()

    class Model:
        async def complete(self, request):
            await release.wait()
            return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), (-0.5,), "stop")

    task = arithmetic_task()
    original = task.model_dump_json()
    runner = engine(Model(), {})
    limited = asyncio.create_task(runner.run(task, execution=TaskExecution(agent_timeout=0.05)))
    unlimited = asyncio.create_task(runner.run(task, execution=TaskExecution()))
    try:
        stopped = await asyncio.wait_for(limited, timeout=5)
        assert stopped.stop_reason == "agent_timeout"
        assert (stopped.grade.status, stopped.grade.reward) == (Outcome.UNAVAILABLE, None)
        assert not unlimited.done()
        release.set()
        completed = await asyncio.wait_for(unlimited, timeout=5)
        assert (completed.grade.status, completed.grade.reward) == (Outcome.GRADED, 1.0)
        assert completed.response_token_ids == (20,)
        assert task.model_dump_json() == original
    finally:
        release.set()
        await asyncio.gather(limited, unlimited, return_exceptions=True)


async def test_linux_files_keep_distinct_case_sensitive_paths():
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(
                    EnvironmentFile(path="/workspace/Makefile", content=b"upper\n"),
                    EnvironmentFile(path="/workspace/makefile", content=b"lower\n"),
                ),
            ),
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(
                    argv=(
                        "sh",
                        "-c",
                        'test "$(cat /workspace/Makefile)" = upper && test "$(cat /workspace/makefile)" = lower',
                    ),
                    timeout=5,
                    reward=ExitCodeReward(),
                ).model_dump_json(),
            ),
        }
    )
    model = ReplayModel([{"role": "assistant", "content": "Done."}])
    result = await engine(model, {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}).run(
        task, execution=TaskExecution()
    )
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)


async def test_explicit_timestamps_cannot_silently_change_on_shellsim():
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/workspace/answer", content=b"12\n", mtime_ns=1_234_567_890),),
            )
        }
    )
    with pytest.raises(ValueError):
        await engine(
            ReplayModel([{"role": "assistant", "content": "Done."}]),
            {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        ).run(task, execution=TaskExecution())


@pytest.mark.parametrize("timeout_phase", ["model", "advance"])
@pytest.mark.parametrize("staged", [False, True])
async def test_agent_deadline_grades_the_workspace_and_preserves_tokens(timeout_phase, staged):
    factory = RecordingShellSimFactory()
    machines = factory.machines

    class SlowMachine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            result = await self.machine.run(command)
            if timeout_phase == "advance" and command.argv == ("sh", "-c", "echo 12 > /workspace/answer"):
                await asyncio.Future()
            return result

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return SlowMachine(await factory.create(spec))

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
    task = file_task()
    if staged:
        task = task.model_copy(
            update={
                "stages": (
                    TaskStage(name="first", verifier=task.verifier),
                    TaskStage(name="not-attempted", verifier=task.verifier),
                ),
                "verifier": VerifierSpec(
                    kind=VerifierKind.STAGED,
                    parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
                ),
            }
        )
    rollout = await engine(model, {EnvironmentKind.SHELLSIM: Factory()}).run(
        task, execution=TaskExecution(agent_timeout=1, stages={stage.name: StageExecution() for stage in task.stages})
    )
    assert rollout.stop_reason == "agent_timeout"
    assert rollout.failure is None
    assert rollout.response_token_ids == (20,)
    assert rollout.loss_mask == (1,)
    assert rollout.logprobs == (-0.5,)
    assert rollout.messages[-1]["tool_calls"][0]["id"] == "write"
    assert len(rollout.steps) == 1
    assert rollout.steps[0].transition.metrics == ({"advance_incomplete": 1.0} if timeout_phase == "advance" else {})
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    if staged:
        assert [stage["name"] for stage in rollout.grade.diagnostics["stages"]] == ["first"]
    with pytest.raises(RuntimeError, match="closed"):
        await machines[0].run(Command(("true",)))


async def test_model_failure_grading_keeps_its_own_deadline_and_original_cause():
    grading_started = asyncio.Event()
    release_grade = asyncio.Event()

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Continue."},), {})

        async def advance(self, turn):
            return Transition(done=False)

        async def grade(self, messages):
            grading_started.set()
            await release_grade.wait()
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    class FailedModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise ConnectionError("Serving failed")
            return await super().complete(request)

    task = arithmetic_task().model_copy(
        update={"environment": EnvironmentSpec(kind=EnvironmentKind.NULL, interaction="fixture")}
    )
    runner = engine(
        FailedModel([{"role": "assistant", "content": "12"}]),
        {},
        sessions={"fixture": lambda task, machine: Session()},
    )
    pending = asyncio.create_task(runner.run(task, execution=TaskExecution(agent_timeout=1)))
    await asyncio.wait_for(grading_started.wait(), timeout=5)
    # The verifier remains active beyond the agent deadline.
    asyncio.get_running_loop().call_later(1.1, release_grade.set)
    with pytest.raises(RolloutInterrupted) as failure:
        await pending
    assert failure.value.operation == RolloutOperation.MODEL
    assert isinstance(failure.value.__cause__, ConnectionError)
    assert failure.value.rollout.grade.reward == 1.0
    assert failure.value.rollout.response_token_ids == (20,)


@pytest.mark.parametrize("staged", [False, True])
async def test_agent_deadline_without_a_response_does_not_grade_an_untouched_workspace(staged):
    factory = RecordingShellSimFactory()
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/workspace/answer", content=b"12\n"),),
            ),
        }
    )
    if staged:
        task = task.model_copy(
            update={
                "stages": (
                    TaskStage(name="first", verifier=task.verifier),
                    TaskStage(name="second", verifier=task.verifier),
                ),
                "verifier": VerifierSpec(
                    kind=VerifierKind.STAGED,
                    parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
                ),
            }
        )

    class Model:
        async def complete(self, _request):
            await asyncio.Future()

    record = await engine(Model(), {EnvironmentKind.SHELLSIM: factory}).run(
        task,
        execution=TaskExecution(agent_timeout=0.05, stages={stage.name: StageExecution() for stage in task.stages}),
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.UNAVAILABLE, None)
    assert record.stop_reason == "agent_timeout"
    assert record.response_token_ids == record.loss_mask == ()
    assert record.steps == ()
    if staged:
        assert [stage["name"] for stage in record.grade.diagnostics["stages"]] == ["first"]
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


async def test_agent_timeout_grades_the_recorded_transcript_without_unserved_observations():
    class Session:
        def __init__(self, _task, _machine):
            pass

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue."},))

        async def grade(self, messages):
            return GradeResult(
                Outcome.GRADED,
                float(messages[-1]["content"] == "12"),
                diagnostics={"graded_messages": messages},
            )

        async def close(self):
            pass

    class Model(ReplayModel):
        async def complete(self, request):
            if self.requests:
                await asyncio.Future()
            return await super().complete(request)

    task = arithmetic_task().model_copy(
        update={"environment": EnvironmentSpec(kind=EnvironmentKind.NULL, interaction="fixture")}
    )
    record = await engine(Model([{"role": "assistant", "content": "12"}]), {}, sessions={"fixture": Session}).run(
        task, execution=TaskExecution(agent_timeout=0.05)
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.stop_reason == "agent_timeout"
    assert record.grade.diagnostics["graded_messages"] == record.messages
    assert record.messages[-1] == {"role": "assistant", "content": "12"}
    assert record.response_token_ids == (20,)
    assert record.loss_mask == (1,)


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
async def test_file_grader_preserves_priority_and_rejects_agent_scores(script, status, reward):
    verifier = ShellVerifierSpec(
        argv=("sh", "/tests/test.sh"),
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
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=verifier.model_dump_json(),
                files=(EnvironmentFile(path="/tests/test.sh", content=script.encode()),),
            ),
        }
    )
    result = await engine(
        ReplayModel([{"role": "assistant", "content": "Completed."}]),
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
    ).run(task, execution=TaskExecution())
    assert (result.grade.status, result.grade.reward) == (status, reward)


DEEP_VERDICT = '{"reward": 1.0, "status": "scored", "detail": {"deep": ' + "[" * 100_000 + "]" * 100_000 + "}}"


def verdict_verifier(script: str) -> VerifierSpec:
    return VerifierSpec(
        kind=VerifierKind.SHELL,
        parameters_json=ShellVerifierSpec(
            argv=("sh", "/tests/grade.sh"), timeout=5, reward=VerdictReward()
        ).model_dump_json(),
        files=(EnvironmentFile(path="/tests/grade.sh", content=script.encode()),),
    )


def verdict_script(verdict: str) -> str:
    return f"echo '{verdict}' > \"$VERIFYIT_LOGS_DIR/verdict.json\""


@pytest.mark.parametrize(
    "verdict,status,reward,error,failure",
    [
        (
            '{"reward": 0.5, "status": "scored", "detail": {"criteria": {"C1": true, "C2": false}}}',
            Outcome.GRADED,
            0.5,
            None,
            None,
        ),
        (
            '{"reward": 0.0, "status": "invalid_task", "detail": {"error": "reference fails C2"}}',
            Outcome.INVALID_TASK,
            None,
            "reference fails C2",
            None,
        ),
        (
            '{"reward": 0.0, "status": "infra_error", "detail": {"error": "judge unavailable"}}',
            Outcome.INFRA_ERROR,
            None,
            "judge unavailable",
            None,
        ),
        (None, Outcome.INFRA_ERROR, None, "Grader did not write verdict.json", "missing_reward"),
        (
            '{"reward": 0.5, "status": "invalid_task", "detail": {}}',
            Outcome.INFRA_ERROR,
            None,
            "An unscored verdict must have zero reward",
            "invalid_reward",
        ),
        (
            '{"reward": 1.0, "status": "scored", "detail": {"nan": NaN}}',
            Outcome.INFRA_ERROR,
            None,
            "Out of range float values are not JSON compliant",
            "invalid_reward",
        ),
        (
            '{"reward": 1.0, "status": "scored"}',
            Outcome.INFRA_ERROR,
            None,
            "exactly reward, status, and detail",
            "invalid_reward",
        ),
        (DEEP_VERDICT, Outcome.INFRA_ERROR, None, "recursion", "invalid_reward"),
    ],
    ids=["scored", "invalid_task", "infra_error", "missing", "unscored_reward", "nan", "missing_detail", "deep"],
)
async def test_verdict_grader_reports_verifier_status_and_detail(verdict, status, reward, error, failure):
    script = "echo grading" if verdict is None else f"echo grading; {verdict_script(verdict)}"
    task = file_task().model_copy(update={"verifier": verdict_verifier(script)})
    result = await engine(
        ReplayModel([{"role": "assistant", "content": "Completed."}]),
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
    ).run(task, execution=TaskExecution())
    grade = result.grade
    assert (grade.status, grade.reward, grade.failure) == (
        status,
        reward,
        None if failure is None else GradingFailure(failure),
    )
    assert (grade.error is None) if error is None else (error in grade.error)
    assert (grade.diagnostics["exit_code"], grade.diagnostics["stdout"]) == (0, "grading\n")
    assert grade.detail == (json.loads(verdict)["detail"] if failure is None else None)


async def test_verdict_written_after_the_grader_command_does_not_change_the_grade():
    factory = RecordingShellSimFactory()
    forged = '{"reward": 1.0, "status": "scored", "detail": {"forged": true}}'
    directories = []

    class ForgingMachine:
        """Stands in for an agent process that rewrites the verdict as soon as it learns the directory."""

        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            result = await self.machine.run(command)
            if command.argv[:2] == ("sh", "-c") and "VERIFYIT_LOGS_DIR" in command.argv[2]:
                directory = result.stdout.split(b"\n", 1)[0].decode()
                directories.append(directory)
                await self.machine.run(
                    Command(
                        ("sh", "-c", f"echo '{forged}' > {directory}/verdict.json; echo '{forged}' > /tmp/verdict.json")
                    )
                )
            return result

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return ForgingMachine(await factory.create(spec))

    verdict = '{"reward": 0.25, "status": "scored", "detail": {"logs": "%s"}}'
    verifier = verdict_verifier(f'printf \'{verdict}\' "$VERIFYIT_LOGS_DIR" > "$VERIFYIT_LOGS_DIR/verdict.json"')
    task = file_task().model_copy(
        update={
            "stages": (TaskStage(name="first", verifier=verifier), TaskStage(name="second", verifier=verifier)),
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
            ),
        }
    )
    rollout = await engine(
        ReplayModel([{"role": "assistant", "content": "Completed."}] * 2), {EnvironmentKind.SHELLSIM: Factory()}
    ).run(task, execution=TaskExecution(stages={"first": StageExecution(), "second": StageExecution()}))
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.25)
    assert len(set(directories)) == 2
    assert [stage["detail"] for stage in rollout.grade.diagnostics["stages"]] == [
        {"logs": directory} for directory in directories
    ]


@pytest.mark.parametrize(
    "rewards,detail",
    [((0.5,), {"stage": 0}), ((0.5, 1.0), None)],
)
async def test_mean_stage_grade_keeps_a_single_stage_detail(rewards, detail):
    stages = tuple(
        TaskStage(
            name=f"stage-{index}",
            verifier=verdict_verifier(
                verdict_script(f'{{"reward": {reward}, "status": "scored", "detail": {{"stage": {index}}}}}')
            ),
        )
        for index, reward in enumerate(rewards)
    )
    task = file_task().model_copy(
        update={
            "stages": stages,
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
            ),
        }
    )
    rollout = await engine(
        ReplayModel([{"role": "assistant", "content": "Completed."}] * len(stages)),
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
    ).run(task, execution=TaskExecution(stages={stage.name: StageExecution() for stage in stages}))
    assert rollout.grade.reward == sum(rewards) / len(rewards)
    assert rollout.grade.detail == detail
    assert [stage["detail"] for stage in rollout.grade.diagnostics["stages"]] == [
        {"stage": index} for index in range(len(stages))
    ]


async def test_model_failure_releases_the_shellbox_machine():
    factory = RecordingShellSimFactory()
    machines = factory.machines

    class FailedModel:
        async def complete(self, _request):
            raise ConnectionError("Inference endpoint unavailable")

    with pytest.raises(RolloutInterrupted) as failure:
        await engine(FailedModel(), {EnvironmentKind.SHELLSIM: factory}).run(file_task(), execution=TaskExecution())
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
        await engine(ReplayModel([]), {EnvironmentKind.SHELLSIM: factory}).run(task, execution=TaskExecution())
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
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(argv=("stall",), timeout=5).model_dump_json(),
            ),
        }
    )
    model = Model([{"role": "assistant", "content": "Done."}])
    runner = engine(model, {EnvironmentKind.SHELLSIM: Factory()})
    pending = asyncio.create_task(
        runner.run(
            task,
            execution=TaskExecution(
                attempt_timeout=1 if phase in {"attempt_startup", "model", "grade"} else 10,
            ),
        )
    )
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


async def test_cancellation_waits_for_machine_cleanup_after_startup_failure():
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
        }
    )
    runner = engine(ReplayModel([]), {EnvironmentKind.SHELLSIM: Factory()})
    pending = asyncio.create_task(runner.run(task, execution=TaskExecution(attempt_timeout=1)))
    cleanup_loop, release = await asyncio.wait_for(close_started, timeout=5)
    pending.cancel()
    cleanup_loop.call_soon_threadsafe(release.set)
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(RuntimeError, match="closed"):
        await machine.run(Command(("true",)))


@pytest.mark.parametrize("operation", ["machine_close", "session_close"])
@pytest.mark.parametrize("model_failed", [False, True])
async def test_cleanup_errors_preserve_completed_grades_and_the_primary_failure(operation, model_failed):
    factory = RecordingShellSimFactory()
    secret = "private-cleanup-detail"

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def close(self):
            await self.machine.close()
            if operation == "machine_close":
                raise OSError(secret)

    class Factory:
        async def create(self, spec):
            return Machine(await factory.create(spec))

    class Session:
        def __init__(self, _task, _machine):
            pass

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=not model_failed)

        async def grade(self, _messages):
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            if operation == "session_close":
                raise OSError(secret)

    class Model(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise TimeoutError("Model server deadline expired")
            return await super().complete(request)

    task = arithmetic_task().model_copy(
        update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, interaction="fixture")}
    )
    model = Model([{"role": "assistant", "content": "12"}])
    runner = engine(
        model,
        {EnvironmentKind.SHELLSIM: Factory()},
        cleanup_timeout=1,
        sessions={"fixture": Session},
    )
    if model_failed:
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run(task, execution=TaskExecution())
        assert caught.value.operation == RolloutOperation.MODEL
        assert isinstance(caught.value.__cause__, TimeoutError)
        record = caught.value.rollout
    else:
        record = await runner.run(task, execution=TaskExecution())
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.response_token_ids == (20,)
    assert record.loss_mask == (1,)
    assert record.grade.diagnostics["cleanup_errors"] == [{"operation": operation, "exception_type": "OSError"}]
    assert record.metrics["cleanup_error_count"] == 1.0
    assert secret not in json.dumps(record.grade.diagnostics)
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("cancel", [False, True])
async def test_cleanup_deadline_bounds_a_close_that_suppresses_cancellation(cancel):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    factory = RecordingShellSimFactory()

    class Machine:
        async def close(self):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            await factory.machines[0].close()
            closed.set()

    class Factory:
        async def create(self, spec):
            await factory.create(spec)
            return Machine()

    runner = engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        {EnvironmentKind.SHELLSIM: Factory()},
        cleanup_timeout=0.05,
    )
    task = arithmetic_task().model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)})
    pending = asyncio.create_task(runner.run(task, execution=TaskExecution()))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if cancel:
            pending.cancel()
            asyncio.get_running_loop().call_soon(pending.cancel)
        done, _ = await asyncio.wait((pending,), timeout=1)
        assert pending in done
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            record = pending.result()
            assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
            assert record.response_token_ids == (20,)
            assert record.grade.diagnostics["cleanup_errors"] == [
                {"operation": "machine_close", "exception_type": "TimeoutError"}
            ]
    finally:
        release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)
        if not pending.done():
            pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("interruption", ["startup", "attempt", "cancel"])
async def test_cancelled_creation_keeps_build_files_and_disposes_the_late_machine(interruption):
    started = asyncio.Event()
    closed = asyncio.Event()
    release = threading.Event()
    machines = []
    contexts = []

    class Machine:
        async def close(self):
            await machines[0].close()
            closed.set()

    class Factory:
        async def create(self, spec):
            contexts.append(spec.source.context)
            machine = await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))
            machines.append(machine)
            started.set()

            def finish_creation():
                release.wait()
                assert spec.source.dockerfile.read_bytes() == b"FROM fixture\n"
                return Machine()

            return await asyncio.to_thread(finish_creation)

    task = arithmetic_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.DOCKER,
                image=DockerBuild(files=(EnvironmentFile(path="/Dockerfile", content=b"FROM fixture\n"),)),
                startup_timeout=1 if interruption == "startup" else None,
            ),
        }
    )
    runner = engine(ReplayModel([]), {EnvironmentKind.DOCKER: Factory()})
    pending = asyncio.create_task(
        runner.run(task, execution=TaskExecution(attempt_timeout=1 if interruption == "attempt" else None))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if interruption == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(RolloutInterrupted) as caught:
                await pending
            assert caught.value.operation == (
                RolloutOperation.START if interruption == "startup" else RolloutOperation.ATTEMPT
            )
            assert isinstance(caught.value.__cause__, TimeoutError)
            assert caught.value.rollout.response_token_ids == ()
        assert contexts[0].exists()
    finally:
        release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)
    assert not contexts[0].exists()
    with pytest.raises(RuntimeError, match="closed"):
        await machines[0].run(Command(("true",)))


@pytest.mark.parametrize("model_failed", [False, True])
@pytest.mark.parametrize("stage_chain", ["next", "last", "failed_gate"])
async def test_private_grader_removal_failure_stops_only_a_continuing_chain_and_preserves_the_cause(
    model_failed, stage_chain
):
    factory = RecordingShellSimFactory()

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv == ("rm", "-f", "/private/grade"):
                raise OSError("Cannot remove private grader")
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return Machine(await factory.create(spec))

    class Model(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise TimeoutError("Model server deadline expired")
            return await super().complete(request)

    verifier = VerifierSpec(
        kind=VerifierKind.SHELL,
        parameters_json=ShellVerifierSpec(
            argv=("cat", "/private/grade"),
            timeout=5,
        ).model_dump_json(),
        files=(EnvironmentFile(path="/private/grade", content=b"0.3\n"),),
    )
    task = arithmetic_task().model_copy(
        update={
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "stages": (
                (
                    TaskStage(
                        name="first",
                        verifier=verifier,
                        minimum_rewards={"reward": 1} if stage_chain == "failed_gate" else {},
                    ),
                )
                + (() if stage_chain == "last" else (TaskStage(name="second", verifier=verifier),))
            ),
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.FINAL).model_dump_json(),
            ),
        }
    )
    message = (
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "work", "type": "function", "function": {"name": "shell", "arguments": '{"command":"true"}'}}
            ],
        }
        if model_failed
        else {"role": "assistant", "content": "Completed."}
    )
    model = Model([message])
    runner = engine(model, {EnvironmentKind.SHELLSIM: Factory()})
    execution = TaskExecution(stages={stage.name: StageExecution() for stage in task.stages})
    if model_failed or stage_chain == "next":
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run(task, execution=execution)
        assert caught.value.operation == (RolloutOperation.MODEL if model_failed else RolloutOperation.CLEANUP)
        assert isinstance(caught.value.__cause__, TimeoutError if model_failed else OSError)
        record = caught.value.rollout
    else:
        record = await runner.run(task, execution=execution)
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 0.3)
    assert record.response_token_ids == (20,)
    assert record.loss_mask == (1,)
    assert [stage["name"] for stage in record.grade.diagnostics["stages"]] == ["first"]
    assert record.grade.diagnostics["cleanup_errors"] == [
        {"operation": "stage_grader_remove", "exception_type": "OSError"}
    ]
    assert len(model.requests) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("answer,expected_reward", [(b"\x00\xff\r\n", 1.0), (b"incorrect", 0.0)])
async def test_separate_grader_receives_binary_artifacts_in_a_fresh_machine(answer, expected_reward):
    private_environment = EnvironmentSpec(
        kind=EnvironmentKind.SHELLSIM,
        setup=(EnvironmentCommand(argv=("sh", "-c", "echo clean > /workspace/baseline"), timeout=5),),
    )
    private_files = (
        EnvironmentFile(path="/private/expected", content=b"\x00\xff\r\n"),
        EnvironmentFile(
            path="/private/grade.sh",
            content=(
                b'#!/bin/sh\ntest "$(cat /workspace/baseline)" = clean && '
                b"cmp /private/expected /workspace/submission"
            ),
            mode=0o755,
        ),
    )
    verifier = ShellVerifierSpec(
        argv=("/private/grade.sh",),
        timeout=5,
        reward=ExitCodeReward(),
        collect=(EnvironmentCommand(argv=("cp", "/workspace/answer", "/workspace/submission"), timeout=5),),
        artifacts=(
            VerifierArtifact(source="/workspace/submission", target="/workspace/submission", kind=ArtifactKind.FILE),
        ),
    )
    task = file_task().model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/workspace/input", content=answer),),
            ),
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=verifier.model_dump_json(),
                environment=private_environment,
                files=private_files,
            ),
        }
    )
    task = TaskSpec.model_validate_json(task.model_dump_json())
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

    result = await engine(model, {EnvironmentKind.SHELLSIM: factory}).run(task, execution=TaskExecution())
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected_reward)
    assert len(machines) == 2
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(argv=("true",)))


@pytest.mark.parametrize("download_failed", [False, True])
async def test_artifact_archive_cleanup_failure_retains_the_grade_or_primary_error(tmp_path, download_failed):
    answer = tmp_path / "answer"
    answer.write_bytes(b"12\n")
    factory = RecordingShellSimFactory()

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv[:2] == ("tar", "-cf"):
                return Result(0, b"", b"", False, False, ExitReason.EXITED)
            if command.argv[:2] == ("rm", "-f") and command.argv[2].startswith("/tmp/taskcompendium-artifact-"):
                raise OSError("Cannot remove artifact archive")
            return await self.machine.run(command)

        async def download(self, source, target):
            if source.startswith("/tmp/taskcompendium-artifact-"):
                if download_failed:
                    raise ConnectionError("Artifact download failed")
                with tarfile.open(target, "w") as archive:
                    archive.add(answer, arcname="answer")
                return
            await self.machine.download(source, target)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return Machine(await factory.create(spec))

    verifier = ShellVerifierSpec(
        argv=("sh", "-c", 'test "$(cat /workspace/project/answer)" = 12'),
        timeout=5,
        reward=ExitCodeReward(),
        artifacts=(
            VerifierArtifact(
                source="/workspace/project", target="/workspace/project", kind=ArtifactKind.DIRECTORY, exclude=("cache",)
            ),
        ),
    )
    task = file_task().model_copy(
        update={
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=verifier.model_dump_json(),
                environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            )
        }
    )
    runner = engine(ReplayModel([{"role": "assistant", "content": "Completed."}]), {EnvironmentKind.SHELLSIM: Factory()})
    if download_failed:
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run(task, execution=TaskExecution())
        assert caught.value.operation == RolloutOperation.GRADE
        assert isinstance(caught.value.__cause__, ConnectionError)
        record = caught.value.rollout
        assert (record.grade.status, record.grade.reward) == (Outcome.UNAVAILABLE, None)
    else:
        record = await runner.run(task, execution=TaskExecution())
        assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.response_token_ids == (20,)
    assert record.grade.diagnostics["cleanup_errors"] == [
        {"operation": "artifact_archive_remove", "exception_type": "OSError"}
    ]
    for machine in factory.machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))
