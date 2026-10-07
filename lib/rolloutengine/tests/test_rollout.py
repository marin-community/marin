# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage execution, private grading, deadlines, and exact token evidence."""

import asyncio
import json
import tarfile
from dataclasses import dataclass, field, replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, Machine, NetworkPolicy, Result, ShellSimBuiltins
from taskcompendium.grader import grader_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.shell_verifier import (
    ArtifactKind,
    ExitCodeReward,
    FileReward,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.submission import AnswerCall, FinalAction, PlainText
from verifyit.candidate import grade_text_candidate
from verifyit.spec import NumericSpec, StdioSpec, StructuredExactSpec, parse_spec

from rolloutengine.cleanup import finish_cleanup
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutInterrupted,
    RolloutOperation,
    SessionStart,
    Transition,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.lowering import lower_task
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


class FixtureImageFactory:
    """Execute fixture image commands on the built-in filesystem."""

    async def create(self, spec):
        return await ShellSimMachineFactory().create(
            replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
        )


@dataclass
class RecordingShellSimFactory:
    machines: list[Machine] = field(default_factory=list)

    async def create(self, spec):
        machine = await FixtureImageFactory().create(spec)
        self.machines.append(machine)
        return machine


def arithmetic_task() -> TaskSpec:
    return TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=grader_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).verifier,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def file_task(script: bytes = b'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi') -> TaskSpec:
    return arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "environment_requirements": EnvironmentRequirements(capabilities=("shell", "filesystem")),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image=FIXTURE_IMAGE),
                parameters_json=ShellVerifierSpec(
                    argv=("sh", "/tests/grade.sh"),
                    artifacts=(
                        VerifierArtifact(source="/workspace/answer", target="/workspace/answer", kind=ArtifactKind.FILE),
                    ),
                ).model_dump_json(),
            ),
            "resources": ResourceGroups(verifier=(inline_resource("grade.sh", script),)),
        }
    )


def shell_call(command: str, *, call_id: str = "write") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def machine_runtime(**overrides) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        **{
            "backend": "local",
            "network": NetworkPolicy.DENY,
            "cpus": None,
            "memory_mb": None,
            "storage_mb": None,
            "gpus": 0,
            "user": None,
            "startup_timeout": None,
            "cleanup_timeout": None,
            **overrides,
        }
    )


def lowered(task: TaskSpec, *, machine=None, verifier_machine=None, **limits) -> LoweredTaskSpec:
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=machine, verifier_machine=verifier_machine),
        session=TaskSessionSpec(
            **{
                "task_session": "shellbox",
                "max_turns": 3,
                "model_turn_timeout": None,
                "command_timeout": None,
                "tool_turn_timeout": None,
                "total_turn_timeout": None,
                "attempt_timeout": None,
                "verifier_timeout": 5,
                "cleanup_timeout": 5,
                **limits,
            }
        ),
    )


def engine(model, factories=None, *, sessions=None, convention=None) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(
        model.complete,
        {} if factories is None else factories,
        convention=convention or PlainText(id="plain"),
        sessions=sessions,
    )


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_lowering_preserves_task_and_produces_private_grade_with_training_tokens(answer, reward):
    task = arithmetic_task()
    original = task.model_dump_json()
    spec = lowered(task)
    model = ReplayModel([{"role": "assistant", "content": answer}])
    runner = engine(model)
    selected = lower_task(task, spec.runtime, spec.session, factories=runner.factories, sessions=runner.sessions)
    reloaded = LoweredTaskSpec.model_validate_json(selected.model_dump_json())

    result = await runner.run(reloaded)

    assert task.model_dump_json() == reloaded.task.model_dump_json() == original
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert result.prompt_token_ids == (10, 11)
    assert result.response_token_ids == (20,)
    assert result.loss_mask == (1,)
    assert result.logprobs == (-0.5,)
    assert "12" not in json.dumps(model.requests[0].messages)


@pytest.mark.parametrize(
    "answer,status,reward",
    [
        ('{"value":12}', Outcome.GRADED, 1.0),
        ('{"value":13}', Outcome.GRADED, 0.0),
        ('{"value":12,"value":13}', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
async def test_json_answer_grades_typed_evidence_and_rejects_duplicate_keys(answer, status, reward):
    task = arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.JSON,
            "verifier": grader_package(StructuredExactSpec(expected={"value": 12})).verifier,
        }
    )
    model = ReplayModel([{"role": "assistant", "content": answer}])
    record = await engine(model).run(lowered(task))
    assert (record.grade.status, record.grade.reward) == (status, reward)
    assert "12" not in json.dumps(model.requests[0].messages)


@pytest.mark.parametrize("provider", ["machine", "session"])
async def test_unknown_provider_fails_before_startup(provider):
    factory = RecordingShellSimFactory()
    spec = lowered(
        arithmetic_task(),
        machine=machine_runtime(backend="missing") if provider == "machine" else None,
        task_session="missing" if provider == "session" else "shellbox",
    )
    with pytest.raises(ValueError):
        await engine(ReplayModel([]), {"local": factory}).run(spec)
    assert factory.machines == []


async def test_machine_user_applies_to_custom_sessions_without_replacing_explicit_command_users():
    closed = asyncio.Event()

    class Machine:
        async def run(self, command):
            return Result(0, (command.user or "image-user").encode(), b"", False, False, ExitReason.EXITED)

        async def close(self):
            closed.set()

    class Factory:
        async def create(self, spec):
            return Machine()

    class Session:
        def __init__(self, lowered, machine):
            self.machine = machine

        async def prepare(self):
            agent = await self.machine.run(Command(("whoami",)))
            trusted = await self.machine.run(Command(("whoami",), user="0"))
            return SessionStart(
                ({"role": "user", "content": agent.stdout.decode() + ":" + trusted.stdout.decode()},), {}
            )

        async def advance(self, turn):
            return Transition(done=True)

        async def grade(self, messages):
            return GradeResult(Outcome.GRADED, float(messages[0]["content"] == "learner:0"))

        async def close(self):
            pass

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()}, sessions={"identity": Session}
    ).run(lowered(arithmetic_task(), machine=machine_runtime(user="learner"), task_session="identity"))
    assert record.grade.reward == 1.0
    assert closed.is_set()


async def test_shell_calls_keep_private_files_hidden_and_mask_tool_observations():
    model = ReplayModel(
        [
            shell_call("test ! -f /tests/grade.sh && echo 12 > /workspace/answer"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    result = await engine(model, {"local": FixtureImageFactory()}).run(
        lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.response_token_ids == (20, 90, 91, 21)
    assert result.loss_mask == (1, 0, 0, 1)
    assert result.logprobs == (-0.5, 0.0, 0.0, -0.5)
    assert model.requests[1].messages[-1]["tool_call_id"] == "write"
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0


async def test_command_timeouts_return_observations_and_allow_the_model_to_finish():
    class TimeoutMachine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv == ("sh", "-c", "hang"):
                try:
                    async with asyncio.timeout(command.timeout):
                        await asyncio.Future()
                except TimeoutError:
                    return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            return TimeoutMachine(await FixtureImageFactory().create(spec))

    message = shell_call("hang", call_id="first")
    message["tool_calls"].extend(
        [
            *shell_call("hang", call_id="second")["tool_calls"],
            *shell_call("echo 12 > /workspace/answer", call_id="repair")["tool_calls"],
        ]
    )
    model = ReplayModel([message, {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": Factory()}).run(
        lowered(
            file_task(),
            machine=machine_runtime(),
            verifier_machine=machine_runtime(),
            command_timeout=0.05,
            tool_turn_timeout=5,
        )
    )
    observations = model.requests[1].messages[-3:]
    assert [item["tool_call_id"] for item in observations] == ["first", "second", "repair"]
    assert [json.loads(item["content"])["reason"] for item in observations] == ["timed_out", "timed_out", "exited"]
    assert json.loads(observations[-1]["content"])["exit_code"] == 0
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.failure is None
    assert record.response_token_ids == (20, 90, 91, 21)
    assert record.loss_mask == (1, 0, 0, 1)
    assert record.logprobs == (-0.5, 0.0, 0.0, -0.5)


@pytest.mark.parametrize("tool_turn_timeout", [0.05, 0.01])
async def test_tool_turn_cannot_preempt_command_timeout_feedback(tool_turn_timeout):
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}).run(
            lowered(
                file_task(),
                machine=machine_runtime(),
                verifier_machine=machine_runtime(),
                command_timeout=0.05,
                tool_turn_timeout=tool_turn_timeout,
            )
        )
    assert factory.machines == []
    assert model.requests == []


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_candidate_verifier_grades_captured_file_without_text_submission(answer, reward):
    task = arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "environment_requirements": EnvironmentRequirements(capabilities=("shell", "filesystem")),
            "output_paths": ("/app/answer.txt",),
        }
    )
    model = ReplayModel(
        [shell_call(f"mkdir -p /app && echo {answer} > /app/answer.txt"), {"role": "assistant", "content": "Done."}]
    )
    record = await engine(model, {"local": FixtureImageFactory()}).run(lowered(task, machine=machine_runtime()))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert record.loss_mask == (1, 0, 0, 1)
    assert "12" not in json.dumps(model.requests[0].messages)


async def test_answer_call_retains_submission_tool_and_finishes():
    task = arithmetic_task().model_copy(
        update={"environment_requirements": EnvironmentRequirements(capabilities=("shell",))}
    )
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "answer",
                "type": "function",
                "function": {"name": "submit_answer", "arguments": '{"answer":"12"}'},
            }
        ],
    }
    model = ReplayModel([message])
    result = await engine(
        model,
        {"local": FixtureImageFactory()},
        convention=AnswerCall(id="answer"),
    ).run(lowered(task, machine=machine_runtime()))
    assert result.grade.reward == 1.0
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["submit_answer", "shell"]


@pytest.mark.parametrize(
    "calls,expected",
    [(["finish"], Outcome.GRADED), (["finish", "finish"], Outcome.SUBMISSION_FAILURE), ([], Outcome.SUBMISSION_FAILURE)],
)
async def test_native_action_preserves_configured_call_limits(calls, expected):
    task = arithmetic_task().model_copy(
        update={
            "answer_type": AnswerType.NATIVE_ACTION,
            "final_tools": (FunctionDefinition(name="finish", parameters={"type": "object"}),),
            "verifier": VerifierSpec(
                kind="predicted_action",
                parameters_json=json.dumps(
                    {
                        "expected_calls": [{"name": "finish", "arguments": {}}],
                    }
                ),
            ),
        }
    )
    message = {"role": "assistant", "content": "Done."}
    if calls:
        message["tool_calls"] = [
            {"id": str(index), "type": "function", "function": {"name": call, "arguments": "{}"}}
            for index, call in enumerate(calls)
        ]
    model = ReplayModel([message])
    record = await engine(model, convention=FinalAction(id="limited", require_call=True, max_calls=1)).run(lowered(task))
    assert record.grade.status == expected
    assert model.requests[0].options["tool_choice"] == "required"
    assert model.requests[0].options["parallel_tool_calls"] is False


async def test_workspace_state_does_not_receive_a_text_submission_instruction():
    task = file_task().model_copy(update={"answer_type": AnswerType.WORKSPACE_STATE})
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": FixtureImageFactory()}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert record.grade.reward == 1.0
    assert model.requests[0].messages == ({"role": "user", "content": "What is six plus six?"},)


async def test_shared_shell_grading_is_rejected_before_task_startup():
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}).run(lowered(file_task(), machine=machine_runtime()))
    assert factory.machines == []
    assert model.requests == []


async def test_private_shell_verifier_can_run_without_a_task_machine():
    task = arithmetic_task().model_copy(
        update={
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image=FIXTURE_IMAGE),
                parameters_json=ShellVerifierSpec(argv=("cat", "/tests/reward")).model_dump_json(),
            ),
            "resources": ResourceGroups(verifier=(inline_resource("reward", b"0.75"),)),
        }
    )
    factory = RecordingShellSimFactory()
    record = await engine(ReplayModel([{"role": "assistant", "content": "12"}]), {"local": factory}).run(
        lowered(task, verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 0.75)
    assert len(factory.machines) == 1
    with pytest.raises(RuntimeError):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("answer,reward", [(b"\x00\xff\r\n", 1.0), (b"incorrect", 0.0)])
async def test_private_verifier_receives_binary_artifacts_in_a_fresh_workspace(answer, reward):
    verifier = ShellVerifierSpec(
        argv=("sh", "/tests/grade.sh"),
        reward=ExitCodeReward(),
        artifacts=(
            VerifierArtifact(source="/workspace/answer", target="/workspace/submission", kind=ArtifactKind.FILE),
        ),
    )
    task = file_task().model_copy(
        update={
            "verifier": VerifierSpec(
                kind="shell",
                parameters_json=verifier.model_dump_json(),
                environment_requirements=EnvironmentRequirements(
                    docker_image=FIXTURE_IMAGE, setup_commands=("echo clean > /workspace/baseline",)
                ),
            ),
            "resources": ResourceGroups(
                worker=(inline_resource("workspace/input", answer),),
                verifier=(
                    inline_resource("expected", b"\x00\xff\r\n"),
                    inline_resource(
                        "grade.sh",
                        b'test "$(cat /workspace/baseline)" = clean && cmp /tests/expected /workspace/submission',
                    ),
                ),
            ),
        }
    )
    factory = RecordingShellSimFactory()
    model = ReplayModel(
        [
            shell_call("test ! -f /tests/expected && cp input answer && echo tainted > baseline"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    record = await engine(model, {"local": factory}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    assert len(factory.machines) == 2
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize("completed_turns", [0, 1])
async def test_context_limit_keeps_served_evidence_and_grades_only_completed_operations(completed_turns):
    class LimitedModel(ReplayModel):
        async def complete(self, request):
            if len(self.requests) == completed_turns:
                raise GenerationLimitReached((*request.prefix_token_ids, 90, 91))
            return await super().complete(request)

    record = await engine(
        LimitedModel([shell_call("echo 12 > /workspace/answer")]), {"local": FixtureImageFactory()}
    ).run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()))
    assert record.stop_reason == "length"
    assert (record.grade.status, record.grade.reward) == (
        (Outcome.GRADED, 1.0) if completed_turns else (Outcome.UNAVAILABLE, None)
    )
    assert record.response_token_ids == ((20,) if completed_turns else ())
    assert record.loss_mask == ((1,) if completed_turns else ())


@pytest.mark.parametrize("violation", ["prefix", "logprobs", "empty"])
async def test_model_transport_must_preserve_exact_token_evidence(violation):
    async def complete(request):
        if request.prefix_token_ids:
            return ModelTurn({"role": "assistant", "content": "Done."}, (99,), (30,), (-0.5,), "stop")
        return ModelTurn(
            shell_call("echo 12 > /workspace/answer"),
            (10, 11),
            () if violation == "empty" else (20,),
            (-0.5, -0.5) if violation == "logprobs" else () if violation == "empty" else (-0.5,),
            "stop",
        )

    runner = ShellboxRolloutEngine(
        complete,
        {"local": FixtureImageFactory()},
        convention=PlainText(id="plain"),
    )
    with pytest.raises(RolloutContractError):
        await runner.run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()))


@pytest.mark.parametrize(
    "script,status,reward,failure",
    [
        ("echo 1 > /logs/verifier/reward.txt; exit 1", Outcome.GRADED, 1.0, None),
        (
            'echo \'{"reward":0,"extra":0.5}\' > /logs/verifier/reward.json; echo 1 > /logs/verifier/reward.txt',
            Outcome.GRADED,
            0.0,
            None,
        ),
        ("true", Outcome.INFRA_ERROR, None, GradingFailure.MISSING_REWARD),
        (
            "echo broken > /logs/verifier/reward.json; echo 1 > /logs/verifier/reward.txt",
            Outcome.INFRA_ERROR,
            None,
            GradingFailure.INVALID_REWARD,
        ),
    ],
)
async def test_reward_file_priority_rejects_fallback_and_agent_scores(script, status, reward, failure):
    verifier = ShellVerifierSpec(
        argv=("sh", "/tests/grade.sh"),
        reward=FileReward(
            files=(
                RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),
                RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),
            ),
            pass_above=0,
        ),
    )
    task = file_task().model_copy(
        update={
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image=FIXTURE_IMAGE),
                parameters_json=verifier.model_dump_json(),
            ),
            "resources": ResourceGroups(
                worker=(inline_resource("logs/verifier/reward.txt", b"1"),),
                verifier=(inline_resource("grade.sh", script.encode()),),
            ),
        }
    )
    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": FixtureImageFactory()}
    ).run(lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime()))
    assert (record.grade.status, record.grade.reward, record.grade.failure) == (status, reward, failure)
    if reward == 0:
        assert record.grade.passed is False


async def test_one_task_supports_independent_turn_deadlines():
    release = asyncio.Event()

    class Model(ReplayModel):
        async def complete(self, request):
            await release.wait()
            return await super().complete(request)

    task = arithmetic_task()
    original = task.model_dump_json()
    runner = engine(Model([{"role": "assistant", "content": "12"}]))
    limited = asyncio.create_task(runner.run(lowered(task, total_turn_timeout=0.05)))
    unlimited = asyncio.create_task(runner.run(lowered(task)))
    try:
        stopped = await asyncio.wait_for(limited, timeout=5)
        assert (stopped.stop_reason, stopped.grade.status) == ("total_turn_timeout", Outcome.UNAVAILABLE)
        assert not unlimited.done()
        release.set()
        record = await asyncio.wait_for(unlimited, timeout=5)
        assert record.grade.reward == 1.0
        assert task.model_dump_json() == original
    finally:
        release.set()
        await asyncio.gather(limited, unlimited, return_exceptions=True)


@pytest.mark.parametrize(
    "phase,budget",
    [
        ("model", "attempt"),
        ("advance", "attempt"),
        ("grade", "attempt"),
        ("model", "per_phase"),
        ("advance", "per_phase"),
        ("grade", "per_phase"),
        ("model", "total_turn"),
        ("advance", "total_turn"),
    ],
)
async def test_deadlines_keep_generated_tokens_and_close_sessions(phase, budget):
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, turn):
            if phase == "advance":
                entered.set()
                await asyncio.Future()
            return Transition(done=phase == "grade")

        async def grade(self, messages):
            if phase == "grade":
                entered.set()
                await asyncio.Future()
            assert messages[-1]["content"] == "12"
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            closed.set()

    class Model(ReplayModel):
        async def complete(self, request):
            if phase == "model" and self.requests:
                entered.set()
                await asyncio.Future()
            return await super().complete(request)

    limits = {"task_session": "fixture"}
    if budget == "attempt":
        limits["attempt_timeout"] = 0.1
    elif budget == "total_turn":
        limits["total_turn_timeout"] = 0.1
    else:
        limits[{"model": "model_turn_timeout", "advance": "tool_turn_timeout", "grade": "verifier_timeout"}[phase]] = 0.1
    pending = asyncio.create_task(
        engine(
            Model([{"role": "assistant", "content": "12"}]), sessions={"fixture": lambda task, machine: Session()}
        ).run(lowered(arithmetic_task(), **limits))
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    if budget == "total_turn":
        record = await pending
        assert record.stop_reason == "total_turn_timeout"
        assert record.grade.reward == 1.0
    else:
        with pytest.raises(RolloutInterrupted) as caught:
            await pending
        record = caught.value.rollout
        assert caught.value.operation == (
            RolloutOperation.ATTEMPT
            if budget == "attempt"
            else {"model": RolloutOperation.MODEL, "advance": RolloutOperation.ADVANCE, "grade": RolloutOperation.GRADE}[
                phase
            ]
        )
        assert isinstance(caught.value.__cause__, TimeoutError)
        if phase == "grade" and budget == "per_phase":
            assert record.grade.failure == GradingFailure.TIMEOUT
    assert record.response_token_ids == (20,)
    assert record.loss_mask == (1,)
    assert record.logprobs == (-0.5,)
    assert closed.is_set()


async def test_turn_budget_covers_all_turns_and_excludes_final_grading():
    release_grade = asyncio.Event()
    grading = asyncio.Event()
    close = asyncio.Event()

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue."},))

        async def grade(self, messages):
            assert messages[-1]["role"] == "assistant"
            grading.set()
            await release_grade.wait()
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            close.set()

    class Model(ReplayModel):
        async def complete(self, request):
            if self.requests:
                await asyncio.Future()
            return await super().complete(request)

    pending = asyncio.create_task(
        engine(
            Model([{"role": "assistant", "content": "12"}]), sessions={"fixture": lambda task, machine: Session()}
        ).run(lowered(arithmetic_task(), task_session="fixture", total_turn_timeout=0.05))
    )
    await asyncio.wait_for(grading.wait(), timeout=5)
    assert not pending.done()
    release_grade.set()
    record = await pending
    assert record.grade.reward == 1.0
    assert record.stop_reason == "total_turn_timeout"
    assert record.messages[-1]["role"] == "assistant"
    assert close.is_set()


@pytest.mark.parametrize("interruption", ["startup", "attempt", "cancel"])
async def test_late_creation_remains_owned_and_closes_after_cancellation(interruption):
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class LateMachine:
        async def close(self):
            closed.set()

    class Factory:
        async def create(self, spec):
            entered.set()
            await release.wait()
            return LateMachine()

    spec = lowered(
        arithmetic_task(),
        machine=machine_runtime(startup_timeout=0.05 if interruption == "startup" else None),
        attempt_timeout=0.05 if interruption == "attempt" else None,
    )
    pending = asyncio.create_task(engine(ReplayModel([]), {"local": Factory()}).run(spec))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
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
        assert not closed.is_set()
    finally:
        release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)
    assert closed.is_set()


@pytest.mark.parametrize("cancel", [False, True])
async def test_cleanup_outside_attempt_is_bounded_despite_repeated_cancellation(cancel):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Machine:
        async def close(self):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            closed.set()

    class Factory:
        async def create(self, spec):
            return Machine()

    spec = lowered(
        arithmetic_task(), machine=machine_runtime(cleanup_timeout=0.05), cleanup_timeout=5, attempt_timeout=0.01
    )
    pending = asyncio.create_task(
        engine(ReplayModel([{"role": "assistant", "content": "12"}]), {"local": Factory()}).run(spec)
    )
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
            assert record.grade.reward == 1.0
            assert record.grade.diagnostics["cleanup_errors"] == [
                {"operation": "machine_close", "exception_type": "TimeoutError"}
            ]
    finally:
        release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)


async def test_startup_failure_cleanup_retains_primary_error_without_private_messages():
    closed = asyncio.Event()

    class Machine:
        async def upload(self, source, target):
            raise ConnectionError("private-startup-detail")

        async def close(self):
            closed.set()
            raise OSError("private-cleanup-detail")

    class Factory:
        async def create(self, spec):
            return Machine()

    task = arithmetic_task().model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("input", b"input"),))}
    )
    with pytest.raises(RolloutInterrupted) as caught:
        await engine(ReplayModel([]), {"local": Factory()}).run(lowered(task, machine=machine_runtime()))
    record = caught.value.rollout
    assert caught.value.operation == RolloutOperation.START
    assert isinstance(caught.value.__cause__, ConnectionError)
    assert closed.is_set()
    assert record.response_token_ids == record.loss_mask == ()
    assert record.grade.diagnostics["cleanup_errors"] == [{"operation": "machine_close", "exception_type": "OSError"}]
    assert "private" not in json.dumps(record.grade.diagnostics)


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


@pytest.mark.parametrize("phase", ["advance", "grade"])
async def test_external_cancellation_at_attempt_deadline_remains_cancellation(phase):
    closed = asyncio.Event()

    async def wait_for_deadline():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            asyncio.current_task().cancel()
            raise

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return twelve."},), {})

        async def advance(self, turn):
            if phase == "advance":
                await wait_for_deadline()
            return Transition(done=True)

        async def grade(self, messages):
            await wait_for_deadline()

        async def close(self):
            closed.set()

    with pytest.raises(asyncio.CancelledError):
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}]),
            sessions={"fixture": lambda lowered, machine: Session()},
        ).run(lowered(arithmetic_task(), task_session="fixture", attempt_timeout=0.01))
    assert closed.is_set()


async def test_environment_setup_runs_as_root_before_agent_commands():
    commands = []
    closed = asyncio.Event()

    class Machine:
        async def run(self, command):
            commands.append(command)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def close(self):
            closed.set()

    class Factory:
        async def create(self, spec):
            return Machine()

    class Session:
        def __init__(self, lowered, machine):
            self.machine = machine

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Run the task."},), {})

        async def advance(self, turn):
            await self.machine.run(Command(("whoami",)))
            return Transition(done=True)

        async def grade(self, messages):
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    task = arithmetic_task().model_copy(
        update={"environment_requirements": EnvironmentRequirements(setup_commands=("mkdir -p /logs/agent",))}
    )
    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()}, sessions={"fixture": Session}
    ).run(lowered(task, machine=machine_runtime(user="learner"), task_session="fixture"))
    assert record.grade.reward == 1.0
    assert [command.user for command in commands] == ["0", "learner"]
    assert closed.is_set()


@pytest.mark.parametrize("download_failed", [False, True])
async def test_artifact_archive_cleanup_failure_retains_grade_or_primary_error(tmp_path, download_failed):
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
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image=FIXTURE_IMAGE),
                parameters_json=verifier.model_dump_json(),
            )
        }
    )
    runner = engine(ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()})
    spec = lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    if download_failed:
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run(spec)
        assert caught.value.operation == RolloutOperation.GRADE
        assert isinstance(caught.value.__cause__, ConnectionError)
        record = caught.value.rollout
        assert (record.grade.status, record.grade.reward) == (Outcome.UNAVAILABLE, None)
    else:
        record = await runner.run(spec)
        assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.response_token_ids == (20,)
    assert record.grade.diagnostics["cleanup_errors"] == [
        {"operation": "artifact_archive_remove", "exception_type": "OSError"}
    ]
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize(
    "answer_type,answer,status,expected_status,expected_reward",
    [
        (AnswerType.FILE, "12", "scored", Outcome.GRADED, 1.0),
        (AnswerType.FILE, "13", "scored", Outcome.GRADED, 0.0),
        (AnswerType.FILE, "12", "unknown", Outcome.INFRA_ERROR, None),
        (AnswerType.NUMBER, "12", "scored", Outcome.GRADED, 1.0),
        (AnswerType.NUMBER, "13", "scored", Outcome.GRADED, 0.0),
        (AnswerType.NUMBER, "12", "unknown", Outcome.INFRA_ERROR, None),
        (AnswerType.NUMBER, " ", "scored", Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
async def test_separate_verifyit_grader_uses_typed_submissions_without_worker_files(
    answer_type, answer, status, expected_status, expected_reward
):
    factory = RecordingShellSimFactory()

    class VerifierMachine:
        def __init__(self, machine):
            self.machine = machine
            self.verdict = None

        async def run(self, command):
            if command.argv[0] != "python3":
                return await self.machine.run(command)
            visibility = await self.machine.run(
                Command(
                    (
                        "sh",
                        "-c",
                        "test -f /workspace/common && test ! -f /workspace/worker && test -f /workspace/verifier-only",
                    )
                )
            )
            assert visibility.exit_code == 0
            spec = await self.machine.run(Command(("cat", "/tests/verifier.toml")))
            candidate = await self.machine.run(Command(("cat", "/app/answer.txt")))
            reward = grade_text_candidate(parse_spec(spec.stdout.decode()), candidate.stdout.decode())
            self.verdict = {"status": status, "reward": reward.reward, "detail": reward.detail}
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            if source == "/logs/verifier/verdict.json":
                target.write_text(json.dumps(self.verdict))
            else:
                await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            machine = await factory.create(spec)
            return VerifierMachine(machine) if len(factory.machines) == 2 else machine

    task = arithmetic_task().model_copy(
        update={
            "answer_type": answer_type,
            "verifier": (
                arithmetic_task().verifier.model_copy(
                    update={
                        "environment_requirements": EnvironmentRequirements(
                            setup_commands=("echo private > /workspace/verifier-only",)
                        )
                    }
                )
            ),
            "environment_requirements": EnvironmentRequirements(capabilities=("shell", "filesystem")),
            "output_paths": ("/app/answer.txt",) if answer_type == AnswerType.FILE else (),
            "resources": ResourceGroups(
                all=(inline_resource("workspace/common", b"public"),),
                worker=(inline_resource("workspace/worker", b"task-only"),),
            ),
        }
    )
    model = ReplayModel(
        [
            shell_call(f"test ! -f /tests/verifier.toml && mkdir -p /app && echo {answer} > /app/answer.txt"),
            {"role": "assistant", "content": "Done."},
        ]
        if answer_type == AnswerType.FILE
        else [{"role": "assistant", "content": answer}]
    )
    record = await engine(model, {"local": Factory()}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (expected_status, expected_reward)
    if expected_status == Outcome.INFRA_ERROR:
        assert record.grade.failure == GradingFailure.INVALID_REWARD
    if answer_type == AnswerType.FILE:
        assert record.loss_mask == (1, 0, 0, 1)
        assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    else:
        assert record.loss_mask == (1,)
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize("invalid", ["text_executable", "private_output"])
async def test_invalid_private_grading_inputs_fail_before_machine_acquisition(invalid):
    task = arithmetic_task().model_copy(
        update={
            "verifier": (
                grader_package(StdioSpec(command="python answer.py")).verifier
                if invalid == "text_executable"
                else arithmetic_task().verifier
            ),
            "output_paths": ("/tests/verifier.toml",) if invalid == "private_output" else (),
        }
    )
    factory = RecordingShellSimFactory()
    with pytest.raises(ValueError):
        await engine(ReplayModel([]), {"local": factory}).run(lowered(task, verifier_machine=machine_runtime()))
    assert factory.machines == []
