# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage execution, grading, deadlines, and exact token evidence."""

import asyncio
import json
import tarfile
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    ExecutionRequirements,
    ExitReason,
    Machine,
    MachineSpec,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AnswerCall,
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    FinalAction,
    FunctionDefinition,
    JsonValueAnswer,
    NoGrader,
    ProviderRequirement,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    SessionGrader,
    ShellToolBinding,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.output_capture import MAX_OUTPUT_FILES
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellToolConfig
from verifyit.grade import grade as verifyit_grade
from verifyit.spec import FunctionCall, PredictedActionSpec, StructuredExactSpec, parse_spec

from rolloutengine.agent_session import AgentTaskSession
from rolloutengine.cleanup import finish_cleanup
from rolloutengine.contracts import (
    AttemptEnded,
    AttemptStarted,
    GenerationLimitReached,
    Graded,
    MachineReady,
    ModelRequest,
    ModelTurn,
    ModelTurnCompleted,
    RolloutContractError,
    RolloutEvent,
    RolloutEventPayload,
    RolloutEventType,
    RolloutInterrupted,
    RolloutOperation,
    SessionPrepared,
    SessionStart,
    TokenUsage,
    ToolCallCompleted,
    ToolCallResult,
    ToolContext,
    Transition,
    TransitionCompleted,
)
from rolloutengine.engine import RolloutEngine
from rolloutengine.lowering import lower_task
from rolloutengine.spec import LoweredTaskSpec, TraceMode
from rolloutengine.task_session import WORKSPACE_INSTRUCTION
from rolloutengine.test_support import (
    TWELVE,
    ReplayModel,
    arithmetic_task,
    engine,
    lowered,
    machine_runtime,
)

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE
)


@pytest.mark.docker
@pytest.mark.parametrize("attack", [None, "source", "archive"])
async def test_artifact_collection_cannot_read_root_files_through_candidate_path_changes(attack):
    machines = []

    class CandidateMachine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv[0] == "tar":
                if attack == "source":
                    mutation = await self.machine.run(
                        Command(
                            (
                                "sh",
                                "-c",
                                "mv /workspace/artifacts /workspace/submitted && ln -s /private /workspace/artifacts",
                            ),
                            user="nobody",
                        )
                    )
                    assert mutation.exit_code == 0
                elif attack == "archive":
                    mutation = await self.machine.run(
                        Command(("ln", "-sf", "/private/answer", command.argv[2]), user="nobody")
                    )
                    assert mutation.exit_code != 0
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            machine = await DockerMachineFactory().create(replace(spec, source=DockerImage("busybox:1.36")))
            machines.append(machine)
            if spec.env.get("ARTIFACT_TASK_MACHINE") != "1":
                return machine
            prepared = await machine.run(
                Command(
                    (
                        "sh",
                        "-c",
                        "mkdir -m 700 /private && printf secret > /private/answer && "
                        "chmod 600 /private/answer && chmod 777 /workspace && "
                        "mkdir -m 777 /workspace/artifacts",
                    ),
                    cwd="/",
                    user="0",
                )
            )
            assert prepared.exit_code == 0
            written = await machine.run(
                Command(("sh", "-c", "printf public > /workspace/artifacts/answer"), user="nobody")
            )
            assert written.exit_code == 0
            return CandidateMachine(machine)

    task = file_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=FIXTURE_IMAGE,
            working_directory="/workspace",
            environment_variables={"ARTIFACT_TASK_MACHINE": "1"},
        ),
        grader=workspace_grader(
            argv=("sh", "-c", "cmp /workspace/artifacts/answer /tests/expected"),
            reward=ExitCodeReward(),
            artifacts=(
                VerifierArtifact(
                    source="/workspace/artifacts",
                    target="/workspace/artifacts",
                    kind=ArtifactKind.DIRECTORY,
                    exclude=("cache",),
                ),
            ),
        ),
        resources=ResourceGroups(verifier=(inline_resource("expected", b"public"),)),
    )
    runtime = lowered(task, machine=machine_runtime(user="nobody"), verifier_machine=machine_runtime())
    rollout_engine = engine(ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()})
    record = await rollout_engine.run(runtime)
    if attack == "source":
        assert (record.grade.status, record.grade.reward, record.grade.failure) == (
            Outcome.INFRA_ERROR,
            None,
            GradingFailure.EXECUTION,
        )
    else:
        assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    for machine in machines:
        assert (await docker("inspect", machine.name)).exit_code != 0


class FixtureImageFactory(DockerMachineFactory):
    """Execute fixture image commands on the built-in filesystem."""

    backend = Backend.DOCKER

    async def create(self, spec):
        return await ShellSimMachineFactory().create(
            replace(spec, source=PackageSpec(), workdir=spec.workdir or "/workspace")
        )


@dataclass
class RecordingShellSimFactory(DockerMachineFactory):
    backend: Backend = Backend.DOCKER

    machines: list[Machine] = field(default_factory=list)

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        factory = ShellSimMachineFactory() if self.backend == Backend.SHELLSIM else DockerMachineFactory()
        factory.check_environment(requirements, spec)

    async def create(self, spec):
        machine = await FixtureImageFactory().create(spec)
        self.machines.append(machine)
        return machine


def workspace_grader(**fields) -> ScriptGrader:
    """A script grader that reads the agent's workspace files rather than an extracted answer."""
    return ScriptGrader.model_validate(
        {"cwd": "/workspace", "environment": GRADER_ENVIRONMENT, "answer_path": None, **fields}
    )


def file_task(
    script: bytes = b'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi', **update
) -> TaskSpec:
    return arithmetic_task(
        **{
            "answer_type": AnswerType.FILE,
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=FIXTURE_IMAGE,
                capabilities=("shell", "filesystem"),
            ),
            "grader": workspace_grader(
                argv=("sh", "/tests/grade.sh"),
                artifacts=(
                    VerifierArtifact(source="/workspace/answer", target="/workspace/answer", kind=ArtifactKind.FILE),
                ),
            ),
            "resources": ResourceGroups(verifier=(inline_resource("grade.sh", script),)),
            **update,
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


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
@pytest.mark.parametrize(
    "content", ["Solve the task.", [{"type": "text", "text": "Solve "}, {"type": "text", "text": "the task."}]]
)
async def test_agent_session_uses_isolated_task_grading_and_engine_cleanup(answer, reward, content):
    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": answer}])
    released = asyncio.Event()

    async def agent(machine, complete):
        assert machine is not None
        try:
            grader_script_probe = await machine.run(Command(("test", "-f", "/tests/grade.sh")))
            assert grader_script_probe.exit_code != 0
            response = await complete(SessionStart(({"role": "user", "content": content},), {}))
            written = await machine.run(Command(("sh", "-c", f"echo {response.message['content']} > /workspace/answer")))
            assert written.exit_code == 0
        finally:
            released.set()

    record = await RolloutEngine(
        model.complete,
        {"local": factory},
        sessions={
            "agent": lambda context: AgentTaskSession(partial(agent, context.machine), context.require_task_grader()),
        },
    ).run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), task_session="agent"))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert model.requests[0].messages == ({"role": "user", "content": content},)
    assert record.response_token_ids == (20,)
    assert record.loss_mask == (1,)
    assert released.is_set()
    assert len(factory.machines) == 2
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_agent_session_file_grade_preserves_rejected_nonobject_tool_arguments(answer, reward):
    rejected = shell_call("unused", call_id="rejected")
    rejected["tool_calls"][0]["function"]["arguments"] = '["echo incorrect > /workspace/answer"]'
    model = ReplayModel([rejected, {"role": "assistant", "content": answer}])

    async def agent(machine, complete):
        assert machine is not None
        prefix = ({"role": "user", "content": "Write the answer."},)
        first = await complete(SessionStart(prefix, {}))
        feedback = {"role": "tool", "tool_call_id": "rejected", "content": "Arguments must be an object"}
        response = await complete(SessionStart((*prefix, first.message, feedback), {}))
        result = await machine.run(Command(("sh", "-c", f"echo {response.message['content']} > /workspace/answer")))
        assert result.exit_code == 0

    record = await RolloutEngine(
        model.complete,
        {"local": RecordingShellSimFactory()},
        sessions={
            "agent": lambda context: AgentTaskSession(partial(agent, context.machine), context.require_task_grader()),
        },
    ).run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), task_session="agent"))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert record.messages[1] == rejected
    assert record.messages[2]["tool_call_id"] == "rejected"


@pytest.mark.parametrize("arguments", ['["invalid"]', '"invalid"', "null", "42"])
async def test_shell_rejects_nonobject_arguments_and_grades_repaired_file(arguments):
    rejected = shell_call("unused", call_id="rejected")
    rejected["tool_calls"][0]["function"]["arguments"] = arguments
    model = ReplayModel([rejected, shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done"}])
    record = await RolloutEngine(model.complete, {"local": RecordingShellSimFactory()}).run(
        lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.messages[1] == rejected
    assert json.loads(record.messages[2]["content"]) == {"error": "Tool arguments must be a JSON object"}


async def test_declared_outputs_return_in_the_record_and_reach_the_grader():
    task = file_task(
        b"""test "$(cat /workspace/out/task.json)" = '{"id": 1}' && echo 1 || echo 0""",
        output_paths=("/workspace/out",),
        grader=workspace_grader(argv=("sh", "/tests/grade.sh")),
    )
    model = ReplayModel(
        [
            shell_call("""mkdir -p /workspace/out && echo '{"id": 1}' > /workspace/out/task.json"""),
            shell_call("echo notes > /workspace/scratch.txt", call_id="scratch"),
            {"role": "assistant", "content": "Done"},
        ]
    )
    record = await RolloutEngine(model.complete, {"local": RecordingShellSimFactory()}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert record.outputs == {"/workspace/out/task.json": b'{"id": 1}\n'}
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_session_graded_task_returns_and_grades_declared_outputs():
    class Session:
        def __init__(self, context):
            self.machine = context.machine

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Write the report."},), {})

        async def advance(self, turn):
            written = await self.machine.run(
                Command(("sh", "-c", f"echo {turn.message['content']} > /workspace/report.txt; echo x > /workspace/x"))
            )
            assert written.exit_code == 0
            return Transition(done=True)

        async def grade(self, _messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, float(outputs == {"/workspace/report.txt": b"ready\n"}))

        async def close(self):
            pass

    task = file_task(grader=SessionGrader(), resources=ResourceGroups(), output_paths=("/workspace/report.txt",))
    record = await RolloutEngine(
        ReplayModel([{"role": "assistant", "content": "ready"}]).complete,
        {"local": RecordingShellSimFactory()},
        sessions={"fixture": Session},
    ).run(lowered(task, machine=machine_runtime(), task_session="fixture"))
    assert record.outputs == {"/workspace/report.txt": b"ready\n"}
    assert record.grade.reward == 1.0


class OutputWritingSession:
    """Write ``command``'s files in the first transition, then grade only after ``grading`` is released."""

    def __init__(self, context, command: str, grading: asyncio.Future | None = None):
        self.machine = context.machine
        self.command = command
        self.grading = grading
        self.entered = asyncio.Event()

    async def prepare(self):
        return SessionStart(({"role": "user", "content": "Write the report."},), {})

    async def advance(self, turn):
        assert (await self.machine.run(Command(("sh", "-c", self.command)))).exit_code == 0
        return Transition(done=True)

    async def grade(self, _messages, *, stop_reason, outputs):
        self.entered.set()
        if self.grading is not None:
            await self.grading
        return GradeResult(Outcome.GRADED, 1.0)

    async def close(self):
        pass


@pytest.mark.parametrize("deadline", ["attempt_timeout", "verifier_timeout"])
async def test_deadline_during_grading_keeps_captured_outputs(deadline):
    task = file_task(grader=SessionGrader(), resources=ResourceGroups(), output_paths=("/workspace/report.txt",))
    with pytest.raises(RolloutInterrupted) as caught:
        await RolloutEngine(
            ReplayModel([{"role": "assistant", "content": "ready"}]).complete,
            {"local": RecordingShellSimFactory()},
            sessions={
                "fixture": lambda context: OutputWritingSession(
                    context, "echo ready > /workspace/report.txt", asyncio.Future()
                )
            },
        ).run(lowered(task, machine=machine_runtime(), task_session="fixture", **{deadline: 1.0}))
    expected = RolloutOperation.ATTEMPT if deadline == "attempt_timeout" else RolloutOperation.GRADE
    assert caught.value.operation == expected
    assert caught.value.rollout.outputs == {"/workspace/report.txt": b"ready\n"}


async def test_output_capture_over_the_file_budget_interrupts_grading():
    sessions = []

    def session(context):
        command = f"mkdir -p /workspace/out && cd /workspace/out && touch $(seq {MAX_OUTPUT_FILES + 1})"
        sessions.append(OutputWritingSession(context, command))
        return sessions[-1]

    task = file_task(grader=SessionGrader(), resources=ResourceGroups(), output_paths=("/workspace/out",))
    with pytest.raises(RolloutInterrupted) as caught:
        await RolloutEngine(
            ReplayModel([{"role": "assistant", "content": "ready"}]).complete,
            {"local": RecordingShellSimFactory()},
            sessions={"fixture": session},
        ).run(lowered(task, machine=machine_runtime(), task_session="fixture"))
    assert caught.value.operation == RolloutOperation.GRADE
    assert "file or byte budget" in str(caught.value.__cause__)
    assert caught.value.rollout.outputs == {}
    assert not sessions[0].entered.is_set()


@pytest.mark.parametrize("compact", [False, True])
async def test_agent_session_receives_engine_turns_and_controls_next_history(compact):
    first = ({"role": "user", "content": "Return 12."},)
    observation = {"role": "tool", "tool_call_id": "write", "content": "Written."}
    received = []
    finished = asyncio.Event()

    async def agent(complete):
        turn = await complete(SessionStart(first, {"tool_choice": "auto"}))
        received.append(turn)
        history = (
            ({"role": "user", "content": "File written. Finish."},) if compact else (*first, turn.message, observation)
        )
        options = {} if compact else {"tool_choice": "auto"}
        received.append(await complete(SessionStart(history, options)))
        finished.set()

    async def grade(messages, *, stop_reason, outputs):
        assert finished.is_set()
        return GradeResult(Outcome.GRADED, float(messages[-1]["content"] == "12"))

    model = ReplayModel([shell_call("echo 12 > answer"), {"role": "assistant", "content": "12"}])
    result = await engine(model, sessions={"agent": lambda _context: AgentTaskSession(agent, grade)}).run(
        lowered(arithmetic_task(), task_session="agent")
    )

    assert result.grade.reward == 1.0
    assert [turn.response_token_ids for turn in received] == [(20,), (21,)]
    assert [turn.logprobs for turn in received] == [(-0.5,), (-0.5,)]
    assert result.loss_mask == ((1,) if compact else (1, 0, 0, 1))
    assert model.requests[1].options == ({} if compact else {"tool_choice": "auto"})
    assert model.requests[1].messages == (
        ({"role": "user", "content": "File written. Finish."},)
        if compact
        else (*first, received[0].message, observation)
    )


@pytest.mark.parametrize("compact", [False, True])
async def test_agent_session_replays_exact_tokens_on_later_turns(compact):
    first = ({"role": "user", "content": "Start."},)
    observation = {"role": "tool", "tool_call_id": "write", "content": "Done."}
    replay = (501, 502) if compact else (10, 11, 20, 90, 91)
    requests = []

    async def agent(complete):
        turn = await complete(SessionStart(first, {}))
        messages = ({"role": "user", "content": "Finish."},) if compact else (*first, turn.message, observation)
        await complete(SessionStart(messages, {}, prompt_token_ids=replay))

    async def grade(messages, *, stop_reason, outputs):
        return GradeResult(Outcome.GRADED, float(messages[-1]["content"] == "12"))

    async def model(request):
        requests.append(request)
        if len(requests) == 1:
            assert request.prompt_token_ids is None
            return ModelTurn({"role": "assistant", "content": "Working."}, (10, 11), (20,), (-0.5,), "stop")
        assert request.prompt_token_ids == replay
        return ModelTurn({"role": "assistant", "content": "12"}, replay, (21,), (-0.5,), "stop")

    rollout_engine = RolloutEngine(model, {}, sessions={"agent": lambda _context: AgentTaskSession(agent, grade)})
    result = await rollout_engine.run(lowered(arithmetic_task(), task_session="agent"))

    assert result.grade.reward == 1.0
    assert result.prompt_token_ids == (replay if compact else (10, 11))
    assert result.response_token_ids == ((21,) if compact else (20, 90, 91, 21))
    assert result.loss_mask == ((1,) if compact else (1, 0, 0, 1))


async def test_agent_session_is_cancelled_when_engine_generation_is_cancelled():
    model_entered = asyncio.Event()
    agent_closed = asyncio.Event()

    async def agent(complete):
        try:
            await complete(SessionStart(({"role": "user", "content": "Return 12."},), {}))
        finally:
            agent_closed.set()

    async def model(_request):
        model_entered.set()
        await asyncio.Future()

    async def grade(_messages, *, stop_reason, outputs):
        raise AssertionError("Cancellation must not grade an unfinished model call")

    runner = RolloutEngine(model, {}, sessions={"agent": lambda _context: AgentTaskSession(agent, grade)})
    pending = asyncio.create_task(runner.run(lowered(arithmetic_task(), task_session="agent")))
    await asyncio.wait_for(model_entered.wait(), timeout=5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert agent_closed.is_set()


async def test_agent_session_failure_retains_the_generated_turn():
    async def agent(complete):
        await complete(SessionStart(({"role": "user", "content": "Return 12."},), {}))
        raise RuntimeError("Agent failed after receiving its response")

    async def grade(_messages, *, stop_reason, outputs):
        raise AssertionError("The agent failure must be reported before grading")

    runner = engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        sessions={"agent": lambda _context: AgentTaskSession(agent, grade)},
    )
    with pytest.raises(RolloutInterrupted) as caught:
        await runner.run(lowered(arithmetic_task(), task_session="agent"))
    assert caught.value.operation == RolloutOperation.ADVANCE
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.rollout.response_token_ids == (20,)
    assert caught.value.rollout.loss_mask == (1,)


async def test_session_grader_uses_session_owned_grade():
    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=True)

        async def grade(self, messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, float(messages[-1]["content"] == "12"))

        async def close(self):
            pass

    def create(context):
        assert context.task_grader is None
        return Session()

    record = await engine(ReplayModel([{"role": "assistant", "content": "12"}]), sessions={"fixture": create}).run(
        lowered(arithmetic_task(grader=SessionGrader()), task_session="fixture")
    )
    assert record.grade.reward == 1.0


@pytest.mark.parametrize("next_options", [{}, {"tool_choice": "none"}])
async def test_session_replaces_model_options_after_conversation_reset(next_options):
    initial_options = {"tool_choice": "required", "parallel_tool_calls": False}
    revised_messages = ({"role": "user", "content": "Return 12 without tools."},)

    class Session:
        def __init__(self):
            self.turns = 0

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Use a tool."},), initial_options)

        async def advance(self, _turn):
            self.turns += 1
            if self.turns == 1:
                return Transition(done=False, reset_conversation=revised_messages, next_options=next_options)
            return Transition(done=self.turns == 3)

        async def grade(self, messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, float(messages[-1]["content"] == "12"))

        async def close(self):
            pass

    model = ReplayModel([{"role": "assistant", "content": "12"}] * 3)
    result = await engine(model, sessions={"fixture": lambda _context: Session()}).run(
        lowered(arithmetic_task(), task_session="fixture")
    )

    assert [request.options for request in model.requests] == [initial_options, next_options, next_options]
    assert model.requests[1].messages == revised_messages
    assert result.grade.reward == 1.0
    assert result.loss_mask == (1, 0, 0, 1)


@pytest.mark.parametrize("trace", list(TraceMode))
@pytest.mark.parametrize("alter_prefix", [False, True])
async def test_session_replays_exact_initial_tokens_and_rejects_transport_changes(alter_prefix, trace):
    prefix = (128000, 42, 128005)

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Replay case"},), {}, prompt_token_ids=prefix)

        async def advance(self, _turn):
            return Transition(done=True)

        async def grade(self, _messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    async def model(request):
        assert request.prompt_token_ids == prefix
        served = (*prefix[:-1], 99) if alter_prefix else prefix
        return ModelTurn({"role": "assistant", "content": "done"}, served, (5, 6), (-0.1, -0.2), "stop")

    runner = RolloutEngine(model, {}, sessions={"fixture": lambda _context: Session()})
    task = lowered(arithmetic_task(), task_session="fixture", trace=trace)
    if alter_prefix:
        # A chat endpoint cannot serve pinned tokens either, so the replay check holds in every trace mode.
        with pytest.raises(RolloutContractError, match="replay token prefix"):
            await runner.run(task)
        return
    result = await runner.run(task)
    assert (result.prompt_token_ids, result.response_token_ids) == (prefix, (5, 6))
    if trace == TraceMode.EXACT_TOKENS:
        assert (result.loss_mask, result.logprobs) == ((1, 1), (-0.1, -0.2))
    else:
        assert (result.loss_mask, result.logprobs) == ((0, 0), None)


@pytest.mark.parametrize("stop_reason", ["stop", "length"])
async def test_reasoning_only_response_grades_missing_answer_or_existing_files(stop_reason):
    empty = {"role": "assistant", "content": None, "reasoning": "Still working."}

    async def run(task, messages, **runtime):
        replay = ReplayModel(messages)

        async def complete(request):
            turn = await replay.complete(request)
            return replace(turn, stop_reason=stop_reason) if turn.message is empty else turn

        return await RolloutEngine(complete, {"local": FixtureImageFactory()}).run(lowered(task, **runtime))

    missing = await run(arithmetic_task(), [empty])
    assert missing.grade.reward == 0.0
    assert missing.messages[-1] == empty
    task = file_task(
        b'if [ "$(cat /workspace/answer)" = 12 ] && '
        b'[ "$TASKCOMPENDIUM_STOP_REASON" = "' + stop_reason.encode() + b'" ]; then echo 1; else echo 0; fi'
    )
    submitted = await run(
        task,
        [shell_call("echo 12 > /workspace/answer"), empty],
        machine=machine_runtime(),
        verifier_machine=machine_runtime(),
    )
    assert (submitted.grade.status, submitted.grade.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_lowering_preserves_task_and_produces_private_grade_with_training_tokens(answer, reward):
    task = arithmetic_task()
    original = task.model_dump_json()
    spec = lowered(task)
    model = ReplayModel([{"role": "assistant", "content": answer}])
    runner = engine(model)
    selected = lower_task(
        task,
        spec.runtime,
        spec.session,
        factories=runner.factories,
        sessions=runner.sessions,
        providers=runner.providers,
    )
    reloaded = LoweredTaskSpec.model_validate_json(selected.model_dump_json())

    result = await runner.run(reloaded)

    assert task.model_dump_json() == reloaded.task.model_dump_json() == original
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert result.prompt_token_ids == (10, 11)
    assert result.response_token_ids == (20,)
    assert result.loss_mask == (1,)
    assert result.logprobs == (-0.5,)
    assert "12" not in json.dumps(model.requests[0].messages)


@pytest.mark.parametrize("unresolved_role", ["actor", "grader"])
@pytest.mark.parametrize("entrypoint", ["lower", "run"])
async def test_unresolved_recipe_rejected_before_fallback_machine_or_model_start(unresolved_role, entrypoint):
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest\n"),)),
    )
    task = file_task()
    if unresolved_role == "actor":
        task = task.model_copy(update={"environment_requirements": environment})
    else:
        task = task.model_copy(
            update={"grader": workspace_grader(argv=("sh", "/tests/grade.sh"), environment=environment)}
        )
    spec = lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    reloaded = LoweredTaskSpec.model_validate_json(spec.model_dump_json())
    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": "Done."}])
    runner = engine(model, {"local": factory})
    with pytest.raises(UnsupportedMachineSpec):
        if entrypoint == "lower":
            lower_task(
                reloaded.task,
                reloaded.runtime,
                reloaded.session,
                factories=runner.factories,
                sessions=runner.sessions,
                providers=runner.providers,
            )
        else:
            await runner.run(reloaded)
    assert factory.machines == []
    assert model.requests == []


async def test_native_task_without_dependencies_never_falls_back_to_simulator():
    task = file_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, capabilities=("shell", "filesystem")
        )
    )
    model = ReplayModel([{"role": "assistant", "content": "Done."}])
    factory = RecordingShellSimFactory()
    with pytest.raises(UnsupportedMachineSpec):
        await engine(model, {"local": factory}).run(
            lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
        )
    assert model.requests == []
    assert factory.machines == []


@pytest.mark.parametrize(
    "answer,status,reward",
    [
        ('{"value":12}', Outcome.GRADED, 1.0),
        ('{"value":13}', Outcome.GRADED, 0.0),
        ('{"value":12,"value":13}', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
async def test_json_answer_grades_typed_evidence_and_rejects_duplicate_keys(answer, status, reward):
    task = arithmetic_task(
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=verifyit_package(StructuredExactSpec(expected={"value": 12})).grader,
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return Machine()

    class Session:
        def __init__(self, context):
            self.machine = context.machine

        async def prepare(self):
            agent = await self.machine.run(Command(("whoami",)))
            trusted = await self.machine.run(Command(("whoami",), user="0"))
            return SessionStart(
                ({"role": "user", "content": agent.stdout.decode() + ":" + trusted.stdout.decode()},), {}
            )

        async def advance(self, turn):
            return Transition(done=True)

        async def grade(self, messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, float(messages[0]["content"] == "learner:0"))

        async def close(self):
            pass

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()}, sessions={"identity": Session}
    ).run(
        lowered(
            arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
            machine=machine_runtime(user="learner"),
            task_session="identity",
        )
    )
    assert record.grade.reward == 1.0
    assert closed.is_set()


@pytest.mark.parametrize("interface", ["default", "renamed", "task-owned"])
async def test_shell_calls_keep_private_files_hidden_and_mask_tool_observations(interface):
    binding = ShellToolConfig() if interface == "default" else ShellToolConfig(name="terminal", command_parameter="cmd")
    task = file_task()
    name, parameter = binding.name, binding.command_parameter
    if interface == "task-owned":
        name, parameter = "Bash", "script"
        definition = FunctionDefinition(
            name=name,
            description="Use Bash to write the answer file.",
            parameters={
                "type": "object",
                "properties": {parameter: {"type": "string", "description": "Bash source"}},
                "required": [parameter],
                "additionalProperties": False,
            },
        )
        task = file_task(
            context=ConversationInput(
                events=(TextMessage(role="user", content="Use Bash to write 12 to /workspace/answer."),)
            ),
            interaction_tools=(definition,),
            tool_bindings={name: ShellToolBinding(command_parameter=parameter)},
        )
        task = TaskSpec.model_validate_json(task.model_dump_json())
    command = "values=(12); [[ ! -f /tests/grade.sh ]] && echo ${values[0]} > /workspace/answer"
    message = shell_call(command)
    message["tool_calls"][0]["function"] = {
        "name": name,
        "arguments": json.dumps({parameter: command}),
    }
    model = ReplayModel(
        [
            message,
            {"role": "assistant", "content": "Done."},
        ]
    )
    result = await engine(model, {"local": FixtureImageFactory()}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime(), shell_tool=binding)
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.response_token_ids == (20, 90, 91, 21)
    assert result.loss_mask == (1, 0, 0, 1)
    assert result.logprobs == (-0.5, 0.0, 0.0, -0.5)
    assert model.requests[1].messages[-1]["tool_call_id"] == "write"
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    tools = model.requests[0].options["tools"]
    assert [tool["function"]["name"] for tool in tools] == [name]
    assert tools[0]["function"]["parameters"]["required"] == [parameter]
    if interface == "task-owned":
        assert tools[0]["function"] == task.interaction_tools[0].model_dump(exclude_none=True)
        assert model.requests[0].messages[0]["content"] == "Use Bash to write 12 to /workspace/answer."
    response = next(message for message in result.messages if message["role"] == "assistant")
    assert response["tool_calls"][0]["function"]["name"] == name


async def test_shell_binding_collision_rejects_before_machine_or_model_start():
    task = file_task(final_tools=(FunctionDefinition(name="terminal", parameters={"type": "object"}),))
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}).run(
            lowered(
                task,
                machine=machine_runtime(),
                verifier_machine=machine_runtime(),
                shell_tool=ShellToolConfig(name="terminal"),
            )
        )
    assert factory.machines == []
    assert model.requests == []


CHECK_TOOL = FunctionDefinition(
    name="check",
    description="Read a workspace file and submit it.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)
CHECKER = ProviderRequirement(action_interface="checker:v1", initial_state={"attempts": 1})


@dataclass
class CheckSession:
    context: ToolContext
    state: dict
    calls: list[tuple[str, dict]] = field(default_factory=list)
    closed: bool = False

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.context.machine is None:
            self.state["attempts"] += 1
            return ToolCallResult(json.dumps(self.state))
        result = await self.context.machine.run(Command(("cat", arguments["path"])))
        return ToolCallResult(result.stdout.decode(), done=True)

    async def close(self):
        self.closed = True


@dataclass
class CheckProvider:
    definition: FunctionDefinition = CHECK_TOOL
    sessions: list[CheckSession] = field(default_factory=list)

    def tools(self, requirement):
        return (self.definition,)

    async def open(self, context):
        session = CheckSession(context, dict(context.requirement.initial_state))
        self.sessions.append(session)
        return session


def tool_message(*calls: tuple[str, str, str]) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            for call_id, name, arguments in calls
        ],
    }


async def test_shell_and_provider_calls_share_one_dispatch_and_a_done_result_ends_the_rollout():
    task = file_task(
        environment_requirements=file_task().environment_requirements.model_copy(
            update={"tool_providers": {"checker": CHECKER}}
        )
    )
    provider = CheckProvider()
    model = ReplayModel(
        [
            tool_message(
                ("write", "shell", json.dumps({"command": "echo 12 > /workspace/answer"})),
                ("rejected", "check", '["/workspace/answer"]'),
                ("missing", "lookup", "{}"),
            ),
            tool_message(("submit", "check", json.dumps({"path": "/workspace/answer"}))),
            {"role": "assistant", "content": "Not requested."},
        ]
    )
    record = await engine(model, {"local": FixtureImageFactory()}, providers={"checker:v1": provider}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )

    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["shell", "check"]
    observations = model.requests[1].messages[-3:]
    assert json.loads(observations[0]["content"])["exit_code"] == 0
    assert json.loads(observations[1]["content"]) == {"error": "Tool arguments must be a JSON object"}
    assert json.loads(observations[2]["content"]) == {"error": "Unknown tool: lookup"}
    [session] = provider.sessions
    assert (session.context.requirement, session.calls) == (CHECKER, [("check", {"path": "/workspace/answer"})])
    assert session.closed
    assert len(model.requests) == 2
    assert record.steps[-1].transition.done
    assert record.steps[-1].transition.observations[0]["content"] == "12\n"
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_provider_tools_run_on_the_host_without_a_task_machine():
    task = arithmetic_task(environment_requirements=EnvironmentRequirements(tool_providers={"checker": CHECKER}))
    provider = CheckProvider()
    model = ReplayModel(
        [
            tool_message(("lookup", "check", json.dumps({"path": "/state"}))),
            {"role": "assistant", "content": "12"},
        ]
    )
    record = await engine(model, providers={"checker:v1": provider}).run(lowered(task))

    assert provider.sessions[0].context.machine is None
    assert json.loads(model.requests[1].messages[-1]["content"]) == {"attempts": 2}
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_each_attempt_opens_and_closes_its_own_provider_session():
    task = arithmetic_task(environment_requirements=EnvironmentRequirements(tool_providers={"checker": CHECKER}))
    provider = CheckProvider()
    for _ in range(2):
        model = ReplayModel(
            [tool_message(("lookup", "check", json.dumps({"path": "/state"}))), {"role": "assistant", "content": "12"}]
        )
        await engine(model, providers={"checker:v1": provider}).run(lowered(task))
        # State from the earlier attempt does not leak: each attempt starts from initial_state.
        assert json.loads(model.requests[1].messages[-1]["content"]) == {"attempts": 2}

    assert [session.closed for session in provider.sessions] == [True, True]


@pytest.mark.parametrize("interface,tool_name", [("missing:v1", "check"), ("checker:v1", "shell")])
async def test_unregistered_or_colliding_provider_rejects_before_machine_or_model_start(interface, tool_name):
    requirement = ProviderRequirement(action_interface=interface, initial_state={})
    task = file_task(
        environment_requirements=file_task().environment_requirements.model_copy(
            update={"tool_providers": {"checker": requirement}}
        )
    )
    provider = CheckProvider(CHECK_TOOL.model_copy(update={"name": tool_name}))
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}, providers={"checker:v1": provider}).run(
            lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
        )
    assert factory.machines == []
    assert model.requests == []


async def test_command_timeouts_return_observations_and_allow_the_model_to_finish():
    class TimeoutMachine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv == ("bash", "-c", "hang"):
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

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


@pytest.mark.parametrize("spill_bytes", [None, 100])
async def test_large_shell_output_spills_to_a_machine_file_only_when_enabled(spill_bytes):
    full_output = "".join(f"{line}\n" for line in range(1, 2001))
    model = ReplayModel([shell_call("seq 1 2000; echo warning >&2"), {"role": "assistant", "content": "Done."}])

    @dataclass
    class InspectingModel:
        """Read the spilled file in a follow-up command, as an agent would."""

        requests: list[ModelRequest] = field(default_factory=list)

        async def complete(self, request: ModelRequest) -> ModelTurn:
            self.requests.append(request)
            if spill_bytes is not None and len(self.requests) == 2:
                path = json.loads(request.messages[-1]["content"])["stdout_path"]
                model.messages.insert(1, shell_call(f"wc -c < {path}; tail -n 1 {path}", call_id="read"))
            return await model.complete(request)

    inspecting = InspectingModel()
    await engine(inspecting, {"local": FixtureImageFactory()}).run(
        lowered(
            file_task(),
            machine=machine_runtime(),
            verifier_machine=machine_runtime(),
            tool_output_spill_bytes=spill_bytes,
        )
    )

    observation = json.loads(model.requests[1].messages[-1]["content"])
    if spill_bytes is None:
        assert observation == {
            "exit_code": 0,
            "reason": "exited",
            "stdout": full_output,
            "stderr": "warning\n",
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
        return
    assert observation["stdout"] == full_output[:spill_bytes]
    assert observation["stdout_truncated"] is True
    assert observation["stdout_path"].startswith("/tmp/tool-output/")
    assert (observation["stderr"], observation["stderr_truncated"], observation["stderr_path"]) == (
        "warning\n",
        False,
        None,
    )
    read = json.loads(model.requests[2].messages[-1]["content"])
    assert read["stdout"].split() == [str(len(full_output)), "2000"]


async def test_answer_call_retains_submission_tool_and_finishes():
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE, capabilities=("shell",)
        ),
        answer_format=AnswerCall(),
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
    result = await engine(model, {"local": FixtureImageFactory()}).run(lowered(task, machine=machine_runtime()))
    assert result.grade.reward == 1.0
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["submit_answer", "shell"]


@pytest.mark.parametrize(
    "calls,expected",
    [(["finish"], Outcome.GRADED), (["finish", "finish"], Outcome.SUBMISSION_FAILURE), ([], Outcome.SUBMISSION_FAILURE)],
)
async def test_native_action_preserves_configured_call_limits(calls, expected):
    task = arithmetic_task(
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(require_call=True, max_calls=1),
        final_tools=(FunctionDefinition(name="finish", parameters={"type": "object"}),),
        grader=verifyit_package(PredictedActionSpec(expected_calls=(FunctionCall("finish", {}),))).grader,
    )
    message = {"role": "assistant", "content": "Done."}
    if calls:
        message["tool_calls"] = [
            {"id": str(index), "type": "function", "function": {"name": call, "arguments": "{}"}}
            for index, call in enumerate(calls)
        ]
    model = ReplayModel([message])
    record = await engine(model).run(lowered(task))
    assert record.grade.status == expected
    assert model.requests[0].options["tool_choice"] == "required"
    assert model.requests[0].options["parallel_tool_calls"] is False


async def test_workspace_state_receives_shell_presentation_without_answer_tools():
    task = file_task(answer_type=AnswerType.WORKSPACE_STATE, answer_format=AnswerCall())
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": FixtureImageFactory()}).run(
        lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert record.grade.reward == 1.0
    assert model.requests[0].messages[:-1] == ({"role": "user", "content": "What is six plus six?"},)
    assert model.requests[0].messages[-1] == {"role": "user", "content": WORKSPACE_INSTRUCTION.format(tool_name="shell")}
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["shell"]


@pytest.mark.parametrize(
    "grader,verifier_machine",
    [
        (verifyit_package(TWELVE, environment=GRADER_ENVIRONMENT).grader, None),
        (ScriptGrader(argv=("true",), environment=GRADER_ENVIRONMENT), None),
        (verifyit_package(TWELVE).grader, machine_runtime()),
        (NoGrader(reason="No evaluator"), machine_runtime()),
        (SessionGrader(), None),
    ],
    ids=["verifyit_environment", "script", "in_process", "none", "session"],
)
async def test_grader_without_matching_runtime_is_rejected_before_task_startup(grader, verifier_machine):
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}).run(
            lowered(arithmetic_task(grader=grader), machine=machine_runtime(), verifier_machine=verifier_machine)
        )
    assert factory.machines == []
    assert model.requests == []


async def test_state_answer_is_rejected_because_the_shellbox_session_cannot_capture_state():
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(NotImplementedError):
        await engine(model, {"local": factory}).run(
            lowered(
                file_task(answer_type=AnswerType.STATE), machine=machine_runtime(), verifier_machine=machine_runtime()
            )
        )
    assert factory.machines == []
    assert model.requests == []


async def test_ungraded_task_reports_reason_without_a_verifier_machine():
    reason = "The source evaluator is unavailable"
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=FIXTURE_IMAGE,
            capabilities=("shell", "filesystem"),
        ),
        grader=NoGrader(reason=reason),
    )
    factory = RecordingShellSimFactory()
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "12"}])
    record = await engine(model, {"local": factory}).run(lowered(task, machine=machine_runtime()))
    assert (record.grade.status, record.grade.reward, record.grade.error) == (Outcome.UNAVAILABLE, None, reason)
    assert record.loss_mask == (1, 0, 0, 1)
    assert len(factory.machines) == 1
    with pytest.raises(RuntimeError):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_script_grader_reads_answer_and_conversation_files_without_a_task_machine(answer, reward):
    script = b"""answer=$(cat /app/answer.txt)
jq --arg answer "$answer" 'if .[0].content == "What is six plus six?" and
    .[-1].content == $answer and $answer == "12" then 1 else 0 end' /tests/conversation.json
"""
    task = arithmetic_task(
        grader=ScriptGrader(argv=("sh", "/tests/grade.sh"), environment=GRADER_ENVIRONMENT),
        resources=ResourceGroups(verifier=(inline_resource("grade.sh", script),)),
    )
    factory = RecordingShellSimFactory()
    record = await engine(ReplayModel([{"role": "assistant", "content": answer}]), {"local": factory}).run(
        lowered(task, verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert len(factory.machines) == 1
    with pytest.raises(RuntimeError):
        await factory.machines[0].run(Command(("true",)))


@pytest.mark.parametrize("answer,reward", [(b"\x00\xff\r\n", 1.0), (b"incorrect", 0.0)])
@pytest.mark.parametrize("command_semantics", [CommandSemantics.LINUX_PROCESS, CommandSemantics.SHELL_SIMULATOR])
async def test_private_verifier_receives_binary_artifacts_in_a_fresh_workspace(answer, reward, command_semantics):
    task = file_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=command_semantics,
            docker_image=FIXTURE_IMAGE if command_semantics == CommandSemantics.LINUX_PROCESS else None,
            capabilities=("shell", "filesystem"),
        ),
        grader=workspace_grader(
            argv=("sh", "/tests/grade.sh"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=FIXTURE_IMAGE,
                setup_commands=("echo clean > /workspace/baseline",),
            ),
            reward=ExitCodeReward(),
            artifacts=(
                VerifierArtifact(source="/workspace/answer", target="/workspace/submission", kind=ArtifactKind.FILE),
            ),
        ),
        resources=ResourceGroups(
            worker=(inline_resource("workspace/input", answer),),
            verifier=(
                inline_resource("expected", b"\x00\xff\r\n"),
                inline_resource(
                    "grade.sh",
                    b'test "$(cat /workspace/baseline)" = clean && cmp /tests/expected /workspace/submission',
                ),
            ),
        ),
    )
    factory = RecordingShellSimFactory()
    simulator = RecordingShellSimFactory(backend=Backend.SHELLSIM)
    model = ReplayModel(
        [
            shell_call("printf '12\\n' && test ! -f /tests/expected && cp input answer && echo tainted > baseline"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    record = await engine(model, {"local": factory, "simulator": simulator}).run(
        lowered(
            task,
            machine=machine_runtime(
                backend="simulator" if command_semantics == CommandSemantics.SHELL_SIMULATOR else "local"
            ),
            verifier_machine=machine_runtime(),
        )
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    tool_result = json.loads(model.requests[1].messages[-1]["content"])
    assert tool_result["exit_code"] == 0
    assert tool_result["stdout"] == "12\n"
    machines = [*factory.machines, *simulator.machines]
    assert len(machines) == 2
    for machine in machines:
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


@pytest.mark.parametrize(
    "usage, budget, requests, stop_reason",
    [
        # Token IDs charge 3 tokens, then 6: the second request reaches the budget.
        (None, 9, 2, "budget_reached"),
        # Requests charge 2 each; after the third, 6 + 2 < 10, so only the turn limit stops the rollout.
        (TokenUsage(prompt_tokens=1, completion_tokens=1), 10, 3, "max_turns"),
        # The third and last permitted request leaves no room for another: the budget takes precedence.
        (TokenUsage(prompt_tokens=1, completion_tokens=1), 8, 3, "budget_reached"),
        # After two requests of 3, a third prompt holds at least 3 more tokens and would reach 9, so it is not sent.
        (TokenUsage(prompt_tokens=2, completion_tokens=1), 9, 2, "budget_reached"),
        (TokenUsage(prompt_tokens=8, completion_tokens=1), 9, 1, "budget_reached"),
    ],
)
async def test_token_budget_stops_model_requests_and_grades_the_last_response(usage, budget, requests, stop_reason):
    graded = []

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue."},))

        async def grade(self, messages, *, stop_reason, outputs):
            graded.append((messages[-1]["role"], stop_reason))
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    class Model(ReplayModel):
        async def complete(self, request):
            return replace(await super().complete(request), usage=usage)

    model = Model([{"role": "assistant", "content": "12"}] * 3)
    record = await engine(model, sessions={"fixture": lambda _context: Session()}).run(
        lowered(arithmetic_task(), task_session="fixture", max_total_tokens=budget)
    )
    assert len(model.requests) == requests
    assert record.stop_reason == stop_reason
    assert graded == [("assistant", stop_reason)]


async def test_token_budget_skips_a_conversation_reset_and_grades_the_last_response():
    graded = []

    class Session:
        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=False, reset_conversation=({"role": "user", "content": "Start again."},))

        async def grade(self, messages, *, stop_reason, outputs):
            graded.append((messages[-1]["role"], stop_reason))
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    model = ReplayModel([{"role": "assistant", "content": "12"}] * 3)
    # Each request after a reset charges 3 tokens; after the second, a third would reach the budget of 9.
    record = await engine(model, sessions={"fixture": lambda _context: Session()}).run(
        lowered(arithmetic_task(), task_session="fixture", max_total_tokens=9)
    )
    assert len(model.requests) == 2
    assert record.stop_reason == "budget_reached"
    assert graded == [("assistant", "budget_reached")]


@pytest.mark.parametrize("usage", [None, TokenUsage(prompt_tokens=4, completion_tokens=1)])
async def test_token_budget_under_messages_trace_needs_usage_or_token_ids(usage):
    class UsageChatModel(ChatModel):
        async def complete(self, request):
            return replace(await super().complete(request), usage=usage)

    model = UsageChatModel([{"role": "assistant", "content": "12"}])
    run = engine(model).run(lowered(arithmetic_task(), trace=TraceMode.MESSAGES, max_total_tokens=100))
    if usage is None:
        with pytest.raises(RolloutContractError, match="token budget"):
            await run
        return
    record = await run
    assert (record.stop_reason, record.grade.reward) == ("stop", 1.0)


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

    runner = RolloutEngine(complete, {"local": FixtureImageFactory()})
    with pytest.raises(RolloutContractError):
        await runner.run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()))


@dataclass
class ChatModel:
    """Return parsed messages without token evidence, as a hosted chat endpoint does."""

    replies: list[dict]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return ModelTurn(self.replies[len(self.requests) - 1], (), (), None, "stop")


async def test_message_trace_runs_chat_models():
    replies = [shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    spec = lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), trace=TraceMode.MESSAGES)
    model = ChatModel(replies)

    record = await engine(model, {"local": FixtureImageFactory()}).run(spec)

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.stop_reason == "stop"
    assert record.messages[-1] == replies[-1]
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    assert [request.prefix_token_ids for request in model.requests] == [(), ()]
    assert len(record.steps) == 2
    assert (record.prompt_token_ids, record.response_token_ids, record.loss_mask, record.logprobs) == ((), (), (), None)
    assert record.metrics["trace"] == "messages"


@dataclass
class RetokenizingChatModel:
    """Report token IDs whose later prompts do not extend the earlier sequence, as a re-tokenizing endpoint does."""

    replies: list[dict]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (10, 11) if index == 0 else (10, 99, 98, 97)
        return ModelTurn(self.replies[index], prompt, (20 + index,), (-0.5,), "stop")


async def test_message_trace_keeps_reported_tokens_without_making_them_trainable():
    replies = [shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    spec = lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), trace=TraceMode.MESSAGES)

    record = await engine(RetokenizingChatModel(replies), {"local": FixtureImageFactory()}).run(spec)

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    # The last served sequence minus the first prompt's length: re-tokenization makes it informational only.
    assert (record.prompt_token_ids, record.response_token_ids) == ((10, 11), (98, 97, 21))
    assert (record.loss_mask, record.logprobs) == ((0, 0, 0), None)
    assert record.metrics["trace"] == "messages"


async def test_interrupted_message_trace_record_is_marked():
    class UnavailableModel:
        async def complete(self, request):
            raise ConnectionError("chat endpoint unavailable")

    spec = lowered(arithmetic_task(), trace=TraceMode.MESSAGES)
    with pytest.raises(RolloutInterrupted) as interrupted:
        await engine(UnavailableModel()).run(spec)

    assert interrupted.value.operation == RolloutOperation.MODEL
    assert interrupted.value.rollout.metrics["trace"] == "messages"


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
            "echo > /logs/verifier/reward.json; echo 1 > /logs/verifier/reward.txt",
            Outcome.INFRA_ERROR,
            None,
            GradingFailure.EMPTY_REWARD,
        ),
        (
            "echo broken > /logs/verifier/reward.json; echo 1 > /logs/verifier/reward.txt",
            Outcome.INFRA_ERROR,
            None,
            GradingFailure.INVALID_REWARD,
        ),
    ],
)
async def test_reward_file_priority_rejects_fallback_and_agent_scores(script, status, reward, failure):
    task = file_task(
        grader=workspace_grader(
            argv=("sh", "/tests/grade.sh"),
            reward=FileReward(
                files=(
                    RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),
                    RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),
                ),
                pass_above=0,
            ),
        ),
        # The grading machine also receives worker resources, so this planted score reaches it.
        resources=ResourceGroups(
            worker=(inline_resource("logs/verifier/reward.txt", b"1"),),
            verifier=(inline_resource("grade.sh", script.encode()),),
        ),
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

        async def grade(self, messages, *, stop_reason, outputs):
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
            Model([{"role": "assistant", "content": "12"}]),
            sessions={"fixture": lambda _context: Session()},
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

        async def grade(self, messages, *, stop_reason, outputs):
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
            Model([{"role": "assistant", "content": "12"}]),
            sessions={"fixture": lambda _context: Session()},
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            entered.set()
            await release.wait()
            return LateMachine()

    spec = lowered(
        arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return Machine()

    spec = lowered(
        arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
        machine=machine_runtime(cleanup_timeout=0.05),
        cleanup_timeout=5,
        attempt_timeout=0.01,
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return Machine()

    task = arithmetic_task(
        environment_requirements=GRADER_ENVIRONMENT,
        resources=ResourceGroups(worker=(inline_resource("input", b"input"),)),
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

        async def grade(self, messages, *, stop_reason, outputs):
            await wait_for_deadline()

        async def close(self):
            closed.set()

    with pytest.raises(asyncio.CancelledError):
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}]),
            sessions={"fixture": lambda _context: Session()},
        ).run(lowered(arithmetic_task(), task_session="fixture", attempt_timeout=0.01))
    assert closed.is_set()


async def test_environment_setup_runs_as_root_before_agent_commands():
    closed = asyncio.Event()

    class Machine:
        def __init__(self):
            self.learner_ready = False

        async def run(self, command):
            if command.user == "0" and command.argv == ("sh", "-c", "mkdir -p /logs/agent"):
                self.learner_ready = True
            if command.user != "0" and not self.learner_ready:
                return Result(126, b"", b"User is not prepared", False, False, ExitReason.EXITED)
            output = command.user.encode() if command.argv == ("whoami",) else b""
            return Result(0, output, b"", False, False, ExitReason.EXITED)

        async def close(self):
            closed.set()

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return Machine()

    class Session:
        def __init__(self, context):
            self.machine = context.machine

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Run the task."},), {})

        async def advance(self, turn):
            result = await self.machine.run(Command(("whoami",)))
            assert (result.exit_code, result.stdout) == (0, b"learner")
            return Transition(done=True)

        async def grade(self, messages, *, stop_reason, outputs):
            return GradeResult(Outcome.GRADED, 1.0)

        async def close(self):
            pass

    task = arithmetic_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=FIXTURE_IMAGE,
                setup_commands=("mkdir -p /logs/agent",),
            )
        }
    )
    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()}, sessions={"fixture": Session}
    ).run(lowered(task, machine=machine_runtime(user="learner"), task_session="fixture"))
    assert record.grade.reward == 1.0
    assert closed.is_set()


async def test_a_failed_setup_command_names_itself_and_keeps_its_output_tail():
    """A setup command that exits non-zero fails the start with the command and its output in the error."""
    stderr = b"mkdir: cannot create directory '/opt/gen': Read-only file system\n"

    class ReadOnlyMachine:
        async def run(self, command):
            if command.argv[:2] == ("sh", "-c"):
                return Result(1, b"made /workspace", stderr, False, False, ExitReason.EXITED)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def close(self):
            pass

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return ReadOnlyMachine()

    task = arithmetic_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=FIXTURE_IMAGE,
                setup_commands=("mkdir -p /workspace /opt/gen",),
            )
        }
    )
    with pytest.raises(RolloutInterrupted) as caught:
        await engine(ReplayModel([]), {"local": Factory()}).run(lowered(task, machine=machine_runtime()))

    assert caught.value.operation == RolloutOperation.START
    cause = caught.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == (
        "Environment setup command failed: exited, exit=1; command: 'mkdir -p /workspace /opt/gen'; "
        "stdout: 'made /workspace'; stderr: \"mkdir: cannot create directory '/opt/gen': Read-only file system\""
    )


@pytest.mark.parametrize(
    "result,cause_type",
    [
        (Result(126, b"", b"", False, False, ExitReason.EXITED), RuntimeError),
        (Result(None, b"", b"", False, False, ExitReason.TIMED_OUT), TimeoutError),
    ],
)
async def test_execution_user_preflight_fails_during_start_before_model_inference(result, cause_type):
    closed = asyncio.Event()

    class FailedProbeMachine:
        async def run(self, command):
            return result

        async def close(self):
            closed.set()

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return FailedProbeMachine()

    model = ReplayModel([])
    task = arithmetic_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE
            )
        }
    )
    with pytest.raises(RolloutInterrupted) as caught:
        await engine(model, {"local": Factory()}).run(lowered(task, machine=machine_runtime(user="learner")))
    assert caught.value.operation == RolloutOperation.START
    assert isinstance(caught.value.__cause__, cause_type)
    assert model.requests == []
    assert closed.is_set()


@pytest.mark.parametrize("failure", [None, "download", "remove", "download_and_remove"])
async def test_artifact_archive_failures_fail_grading_and_close_machines(tmp_path, failure):
    answer = tmp_path / "answer"
    answer.write_bytes(b"12\n")
    factory = RecordingShellSimFactory()

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv[:2] == ("tar", "-cf"):
                return Result(0, b"", b"", False, False, ExitReason.EXITED)
            if (
                failure in {"remove", "download_and_remove"}
                and command.argv[:2] == ("rm", "-rf")
                and command.argv[2].startswith("/tmp/taskcompendium-artifact-")
            ):
                return Result(1, b"", b"Cannot remove artifact archive", False, False, ExitReason.EXITED)
            return await self.machine.run(command)

        async def download(self, source, target):
            if source.startswith("/tmp/taskcompendium-artifact-"):
                if failure in {"download", "download_and_remove"}:
                    raise ConnectionError("Artifact download failed")
                with tarfile.open(target, "w") as archive:
                    archive.add(answer, arcname="answer")
                return
            await self.machine.download(source, target)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def close(self):
            await self.machine.close()

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            machine = await factory.create(spec)
            await machine.run(Command(("mkdir", "-p", "/workspace/project")))
            await machine.upload(answer, "/workspace/project/answer")
            return Machine(machine)

    task = file_task(
        grader=workspace_grader(
            argv=("sh", "-c", 'test "$(cat /workspace/project/answer)" = 12'),
            reward=ExitCodeReward(),
            artifacts=(
                VerifierArtifact(
                    source="/workspace/project",
                    target="/workspace/project",
                    kind=ArtifactKind.DIRECTORY,
                    exclude=("cache",),
                ),
            ),
        )
    )
    runner = engine(ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()})
    spec = lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    if failure in {"download", "download_and_remove"}:
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run(spec)
        assert caught.value.operation == RolloutOperation.GRADE
        assert isinstance(caught.value.__cause__, ConnectionError)
        record = caught.value.rollout
        assert (record.grade.status, record.grade.reward) == (Outcome.UNAVAILABLE, None)
    elif failure == "remove":
        record = await runner.run(spec)
        assert (record.grade.status, record.grade.reward, record.grade.failure) == (
            Outcome.INFRA_ERROR,
            None,
            GradingFailure.EXECUTION,
        )
        # The archive stays on the task machine, so the grader never starts.
        assert len(factory.machines) == 1
    else:
        record = await runner.run(spec)
        assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.response_token_ids == (20,)
    assert "cleanup_errors" not in record.grade.diagnostics
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize(
    "answer_type,answer,status,expected_status,expected_reward",
    [
        (AnswerType.FILE, "12", "scored", Outcome.GRADED, 1.0),
        (AnswerType.FILE, "13", "scored", Outcome.GRADED, 0.0),
        (AnswerType.FILE, "garbage", "scored", Outcome.SUBMISSION_FAILURE, 0.0),
        (AnswerType.FILE, "12", "unknown", Outcome.INFRA_ERROR, None),
        (AnswerType.NUMBER, "12", "scored", Outcome.GRADED, 1.0),
        (AnswerType.NUMBER, "13", "scored", Outcome.GRADED, 0.0),
        (AnswerType.NUMBER, "garbage", "scored", Outcome.SUBMISSION_FAILURE, 0.0),
        (AnswerType.NUMBER, "12", "unknown", Outcome.INFRA_ERROR, None),
        (AnswerType.NUMBER, " ", "scored", Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
async def test_separate_verifyit_grader_uses_typed_submissions_and_task_resources(
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
                        "test -f /workspace/common && test -f /workspace/worker && test -f /workspace/verifier-only",
                    )
                )
            )
            assert visibility.exit_code == 0
            spec = await self.machine.run(Command(("cat", "/tests/verifier.toml")))
            candidate = await self.machine.run(Command(("cat", "/app/answer.txt")))
            with TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / "answer.txt").write_bytes(candidate.stdout)
                reward = verifyit_grade(parse_spec(spec.stdout.decode()), workspace, workspace)
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

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            machine = await factory.create(spec)
            return VerifierMachine(machine) if len(factory.machines) == 2 else machine

    task = arithmetic_task(
        answer_type=answer_type,
        grader=verifyit_package(
            TWELVE,
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=FIXTURE_IMAGE,
                setup_commands=("echo private > /workspace/verifier-only",),
            ),
        ).grader,
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=FIXTURE_IMAGE,
            capabilities=("shell", "filesystem"),
        ),
        output_paths=("/app/answer.txt",) if answer_type == AnswerType.FILE else (),
        resources=ResourceGroups(
            all=(inline_resource("workspace/common", b"public"),),
            worker=(inline_resource("workspace/worker", b"task-only"),),
        ),
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
    if answer_type == AnswerType.NUMBER and status == "scored":
        candidate_record = await engine(ReplayModel([{"role": "assistant", "content": answer}])).run(
            lowered(arithmetic_task())
        )
        assert (record.grade.status, record.grade.reward, record.grade.error) == (
            candidate_record.grade.status,
            candidate_record.grade.reward,
            candidate_record.grade.error,
        )
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


@dataclass
class RecordingObserver:
    events: list[RolloutEvent] = field(default_factory=list)

    async def on_event(self, event: RolloutEvent) -> None:
        self.events.append(event)

    def payloads(self, run_key: str) -> list[RolloutEventPayload]:
        """The untimed payloads, after checking that every event belongs to ``run_key`` of the arithmetic task."""
        assert {(event.run_key, event.task_id) for event in self.events} == {(run_key, "arithmetic")}
        return [
            (
                replace(event.payload, duration=0.0)
                if "duration" in {item.name for item in fields(event.payload)}
                else event.payload
            )
            for event in self.events
        ]


def shell_turn_payloads(model: ReplayModel) -> list[RolloutEventPayload]:
    """Payloads through the first turn of a rollout whose first response writes the answer file."""
    observation = model.requests[1].messages[-1]["content"]
    return [
        AttemptStarted(),
        MachineReady(0.0),
        SessionPrepared(len(model.requests[0].messages), 0.0),
        ModelTurnCompleted(0, "stop", 2, 1, {}, 0.0),
        ToolCallCompleted("shell", len(observation.encode()), 0.0),
        TransitionCompleted(0, False, 1, 0.0),
    ]


async def test_observer_receives_ordered_events_for_a_multi_turn_shell_rollout():
    observer = RecordingObserver()
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await RolloutEngine(model.complete, {"local": FixtureImageFactory()}, observer=observer).run(
        lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()), run_key="arithmetic/3"
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert observer.payloads("arithmetic/3") == [
        *shell_turn_payloads(model),
        ModelTurnCompleted(1, "stop", 5, 1, {}, 0.0),
        TransitionCompleted(1, True, 0, 0.0),
        Graded(Outcome.GRADED, 1.0),
        AttemptEnded("stop", None),
    ]
    assert asdict(observer.events[-1]) == {
        "run_key": "arithmetic/3",
        "task_id": "arithmetic",
        "payload": {"stop_reason": "stop", "operation": None, "type": RolloutEventType.ATTEMPT_ENDED},
    }


async def test_observer_receives_interrupted_operation_after_grading_completed_turns():
    class FailingModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                self.requests.append(request)
                raise ConnectionError("inference server unavailable")
            return await super().complete(request)

    observer = RecordingObserver()
    model = FailingModel([shell_call("echo 12 > /workspace/answer")])
    with pytest.raises(RolloutInterrupted) as interrupted:
        await RolloutEngine(model.complete, {"local": FixtureImageFactory()}, observer=observer).run(
            lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
        )
    assert interrupted.value.operation == RolloutOperation.MODEL
    # Without a run key, events carry the task id.
    assert observer.payloads("arithmetic") == [
        *shell_turn_payloads(model),
        Graded(Outcome.GRADED, 1.0),
        AttemptEnded(None, RolloutOperation.MODEL),
    ]


async def test_observer_failure_leaves_the_rollout_unchanged():
    class FailingObserver:
        async def on_event(self, event: RolloutEvent) -> None:
            if event.payload.type == RolloutEventType.MODEL_TURN_COMPLETED:
                raise RuntimeError("telemetry sink unavailable")

    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await RolloutEngine(model.complete, {"local": FixtureImageFactory()}, observer=FailingObserver()).run(
        lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )

    assert (record.grade.status, record.grade.reward, record.stop_reason) == (Outcome.GRADED, 1.0, "stop")
    # Three events precede the failing first model turn; the six after it are never delivered.
    assert (record.metrics["observer_error_count"], record.metrics["observer_undelivered_events"]) == (1.0, 6.0)


async def test_blocked_observer_does_not_hold_the_rollout():
    class BlockedObserver:
        async def on_event(self, event: RolloutEvent) -> None:
            await asyncio.Event().wait()

    model = ReplayModel([{"role": "assistant", "content": "12"}])
    record = await RolloutEngine(model.complete, {}, observer=BlockedObserver()).run(
        lowered(arithmetic_task(), attempt_timeout=5, cleanup_timeout=0.1)
    )

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.metrics["observer_undelivered_events"] > 0
