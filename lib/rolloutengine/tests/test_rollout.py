# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage execution, grading, deadlines, and exact token evidence."""

import asyncio
import json
import pickle
import tarfile
from dataclasses import asdict, dataclass, field, fields, replace
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
from taskcompendium.grading_result import GradingFailure, Outcome
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
    ShellToolBinding,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.output_capture import MAX_OUTPUT_FILES
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellToolConfig
from verifyit.grade import grade as verifyit_grade
from verifyit.modes.grade_judge import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, JudgeConnection
from verifyit.spec import (
    RUBRIC_LABELS,
    FunctionCall,
    JudgeSpec,
    PredictedActionSpec,
    StructuredExactSpec,
    parse_spec,
)

from rolloutengine.cleanup import finish_cleanup
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.shell import WORKSPACE_INSTRUCTION, ShellHarness, ToolCallResult, ToolContext
from rolloutengine.harness.step import StepHarness
from rolloutengine.machines import task_machine
from rolloutengine.models import (
    AttemptEnded,
    AttemptStarted,
    ConversationStarted,
    GenerationLimitReached,
    Graded,
    GradedRolloutInterrupted,
    LimitReached,
    MachineReady,
    ModelTurn,
    ModelTurnCompleted,
    Prompt,
    RolloutContractError,
    RolloutEnded,
    RolloutEvent,
    RolloutEventPayload,
    RolloutEventType,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
    TokenUsage,
    ToolCallCompleted,
    TraceMode,
    Transition,
    TransitionCompleted,
)
from rolloutengine.recorder import harness_rollout
from rolloutengine.test_support import (
    SHELL,
    TWELVE,
    ReplayModel,
    arithmetic_task,
    engine,
    machine_runtime,
    rollout_limits,
    rollout_spec,
    shell_call,
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
    runtime = rollout_spec(task, machine=machine_runtime(user="nobody"), verifier_machine=machine_runtime())
    rollout_engine = engine(ReplayModel([{"role": "assistant", "content": "Done."}]), {"local": Factory()})
    record = await rollout_engine.run_task(runtime)
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


def judge_task(answer_type: AnswerType = AnswerType.TEXT, *, model: str = "fixture-judge") -> TaskSpec:
    spec = JudgeSpec(
        references=("12",),
        rubric=RUBRIC_LABELS,
        system_prompt="Grade the response.",
        prompt_template="Reference: {reference}\nCandidate: {candidate}",
        label_scores={"PASS": 1.0, "FAIL": 0.0},
        model=model,
    )
    return arithmetic_task(answer_type=answer_type, grader=verifyit_package(spec, environment=GRADER_ENVIRONMENT).grader)


def judge_url(judge_server) -> str:
    return f"http://127.0.0.1:{judge_server.server_port}/v1"


def shell_harnesses(**options) -> dict:
    return {SHELL: ShellHarness(**{"command_timeout": None, **options})}


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
@pytest.mark.parametrize(
    "content", ["Solve the task.", [{"type": "text", "text": "Solve "}, {"type": "text", "text": "the task."}]]
)
async def test_agent_harness_uses_isolated_task_grading_and_engine_cleanup(answer, reward, content):
    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": answer}])
    released = asyncio.Event()

    async def agent(machine, complete):
        assert machine is not None
        try:
            grader_script_probe = await machine.run(Command(("test", "-f", "/tests/grade.sh")))
            assert grader_script_probe.exit_code != 0
            response = await complete(Prompt(({"role": "user", "content": content},), {}))
            written = await machine.run(Command(("sh", "-c", f"echo {response.message['content']} > /workspace/answer")))
            assert written.exit_code == 0
        finally:
            released.set()

    record = await engine(
        model,
        {"local": factory},
        harnesses={"agent": AgentHarness(lambda context, complete: agent(context.machine, complete))},
    ).run_task(rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), harness="agent"))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert model.requests[0].messages == ({"role": "user", "content": content},)
    assert record.rollout.conversations[-1].response_token_ids == (20,)
    assert record.rollout.conversations[-1].loss_mask == (1,)
    assert released.is_set()
    assert len(factory.machines) == 2
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize("arguments", ['["invalid"]', '"invalid"', "null", "42"])
async def test_shell_rejects_nonobject_arguments_and_grades_repaired_file(arguments):
    rejected = shell_call("unused", call_id="rejected")
    rejected["tool_calls"][0]["function"]["arguments"] = arguments
    model = ReplayModel([rejected, shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done"}])
    record = await engine(model, {"local": RecordingShellSimFactory()}).run_task(
        rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.rollout.conversations[-1].messages[1] == rejected
    assert json.loads(record.rollout.conversations[-1].messages[2]["content"]) == {
        "error": "Tool arguments must be a JSON object"
    }


async def test_shell_harness_cuts_each_command_output_at_its_limit():
    model = ReplayModel(
        [shell_call("seq 1 1000; echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done"}]
    )
    harnesses = shell_harnesses(output_limit_bytes=100)
    record = await engine(model, {"local": RecordingShellSimFactory()}, harnesses=harnesses).run_task(
        rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    observation = json.loads(record.rollout.conversations[-1].messages[2]["content"])
    assert (len(observation["stdout"]), observation["stdout_truncated"]) == (100, True)
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


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
    record = await engine(model, {"local": RecordingShellSimFactory()}).run_task(
        rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert record.rollout.outputs == {"/workspace/out/task.json": b'{"id": 1}\n'}
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


class OutputWritingEpisode:
    """Write ``command``'s files in the first transition, and end."""

    def __init__(self, context, command: str):
        self.machine = context.machine
        self.command = command

    async def prepare(self):
        return Prompt(({"role": "user", "content": "Write the report."},), {})

    async def advance(self, turn):
        assert (await self.machine.run(Command(("sh", "-c", self.command)))).exit_code == 0
        return Transition(done=True)

    async def close(self):
        pass


@dataclass
class BlockingFactory(RecordingShellSimFactory):
    """Block the creation of machine number ``blocked``, counted from 0, until ``release`` is set."""

    blocked: int = 1
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def create(self, spec):
        if len(self.machines) == self.blocked:
            self.entered.set()
            await self.release.wait()
        return await super().create(spec)


@dataclass
class FirstTurnModel(ReplayModel):
    """Answer the first request, and wait on each later one until it is cancelled. ``blocked`` is set then."""

    blocked: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, request):
        if self.requests:
            self.blocked.set()
            await asyncio.Future()
        return await super().complete(request)


def report_task(**update) -> TaskSpec:
    return file_task(b"echo 1", grader=workspace_grader(argv=("sh", "/tests/grade.sh")), **update)


@pytest.mark.parametrize("deadline", ["attempt_timeout", "verifier_timeout"])
async def test_deadline_during_grading_keeps_captured_outputs(deadline):
    # The verifier machine is the second machine, so grading blocks after the outputs are captured.
    factory = BlockingFactory()
    harness = StepHarness(lambda context: OutputWritingEpisode(context, "echo ready > /workspace/report.txt"))
    spec = rollout_spec(
        report_task(output_paths=("/workspace/report.txt",)),
        machine=machine_runtime(),
        verifier_machine=machine_runtime(),
        harness="fixture",
        **{deadline: 1.0},
    )
    try:
        with pytest.raises(RolloutInterrupted) as caught:
            await engine(
                ReplayModel([{"role": "assistant", "content": "ready"}]),
                {"local": factory},
                harnesses={"fixture": harness},
            ).run_task(spec)
    finally:
        factory.release.set()
    expected = RolloutOperation.ATTEMPT if deadline == "attempt_timeout" else RolloutOperation.GRADE
    assert caught.value.operation == expected
    assert caught.value.rollout.outputs == {"/workspace/report.txt": b"ready\n"}


async def test_output_capture_over_the_file_budget_interrupts_grading():
    command = f"mkdir -p /workspace/out && cd /workspace/out && touch $(seq {MAX_OUTPUT_FILES + 1})"
    factory = RecordingShellSimFactory()
    with pytest.raises(RolloutInterrupted) as caught:
        await engine(
            ReplayModel([{"role": "assistant", "content": "ready"}]),
            {"local": factory},
            harnesses={"fixture": StepHarness(lambda context: OutputWritingEpisode(context, command))},
        ).run_task(
            rollout_spec(
                report_task(output_paths=("/workspace/out",)),
                machine=machine_runtime(),
                verifier_machine=machine_runtime(),
                harness="fixture",
            )
        )
    assert caught.value.operation == RolloutOperation.GRADE
    assert "file or byte budget" in str(caught.value.__cause__)
    assert caught.value.rollout.outputs == {}
    # The grader never starts, so only the task machine exists.
    assert len(factory.machines) == 1


@pytest.mark.parametrize("compact", [False, True])
async def test_agent_harness_receives_engine_turns_and_controls_next_history(compact):
    first = ({"role": "user", "content": "Return 12."},)
    observation = {"role": "tool", "tool_call_id": "write", "content": "Written."}
    received = []
    finished = asyncio.Event()

    async def agent(complete):
        turn = await complete(Prompt(first, {"tool_choice": "auto"}))
        received.append(turn)
        history = (
            ({"role": "user", "content": "File written. Finish."},) if compact else (*first, turn.message, observation)
        )
        options = {} if compact else {"tool_choice": "auto"}
        received.append(await complete(Prompt(history, options)))
        finished.set()

    model = ReplayModel([shell_call("echo 12 > answer"), {"role": "assistant", "content": "12"}])
    result = await engine(model, harnesses={"agent": AgentHarness(lambda _context, complete: agent(complete))}).run_task(
        rollout_spec(arithmetic_task(), harness="agent")
    )

    assert finished.is_set()
    assert result.grade.reward == 1.0
    assert len(result.rollout.conversations) == (2 if compact else 1)
    assert [turn.response_token_ids for turn in received] == [(20,), (21,)]
    assert [turn.logprobs for turn in received] == [(-0.5,), (-0.5,)]
    assert result.rollout.conversations[-1].loss_mask == ((1,) if compact else (1, 0, 0, 1))
    assert model.requests[1].options == ({} if compact else {"tool_choice": "auto"})
    assert model.requests[1].messages == (
        ({"role": "user", "content": "File written. Finish."},)
        if compact
        else (*first, received[0].message, observation)
    )


@pytest.mark.parametrize("compact", [False, True])
async def test_agent_harness_replays_exact_tokens_on_later_turns(compact):
    first = ({"role": "user", "content": "Start."},)
    # The task's grader reads this chat, so the observation is a user message: the turn makes no tool call.
    observation = {"role": "user", "content": "Done."}
    replay = (501, 502) if compact else (10, 11, 20, 90, 91)
    requests = []

    async def agent(complete):
        turn = await complete(Prompt(first, {}))
        messages = ({"role": "user", "content": "Finish."},) if compact else (*first, turn.message, observation)
        await complete(Prompt(messages, {}, token_ids=replay))

    async def model(request):
        requests.append(request)
        if len(requests) == 1:
            assert request.prompt_token_ids is None
            return ModelTurn({"role": "assistant", "content": "Working."}, (10, 11), (20,), (-0.5,), "stop")
        # A pinned prompt replaces the continuation: the transport serves it unchanged.
        assert (request.prompt_token_ids, request.continuation) == (replay, None)
        return ModelTurn({"role": "assistant", "content": "12"}, replay, (21,), (-0.5,), "stop")

    rollout_engine = RolloutEngine({}, {"agent": AgentHarness(lambda _context, complete: agent(complete))})
    result = await rollout_engine.run_task(rollout_spec(arithmetic_task(), harness="agent"), model)

    assert result.grade.reward == 1.0
    assert result.rollout.conversations[-1].prompt_token_ids == (replay if compact else (10, 11))
    assert result.rollout.conversations[-1].response_token_ids == ((21,) if compact else (20, 90, 91, 21))
    assert result.rollout.conversations[-1].loss_mask == ((1,) if compact else (1, 0, 0, 1))


async def test_agent_harness_is_cancelled_when_engine_generation_is_cancelled():
    model_entered = asyncio.Event()
    agent_closed = asyncio.Event()

    async def agent(complete):
        try:
            await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        finally:
            agent_closed.set()

    async def model(_request):
        model_entered.set()
        await asyncio.Future()

    runner = RolloutEngine({}, {"agent": AgentHarness(lambda _context, complete: agent(complete))})
    pending = asyncio.create_task(runner.run_task(rollout_spec(arithmetic_task(), harness="agent"), model))
    await asyncio.wait_for(model_entered.wait(), timeout=5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert agent_closed.is_set()


async def test_agent_harness_failure_retains_the_generated_turn():
    async def agent(complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        raise RuntimeError("Agent failed after receiving its response")

    runner = engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        harnesses={"agent": AgentHarness(lambda _context, complete: agent(complete))},
    )
    with pytest.raises(GradedRolloutInterrupted) as caught:
        await runner.run_task(rollout_spec(arithmetic_task(), harness="agent"))
    assert caught.value.grade.status == Outcome.UNAVAILABLE
    assert caught.value.operation == RolloutOperation.ADVANCE
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.rollout.conversations[-1].response_token_ids == (20,)
    assert caught.value.rollout.conversations[-1].loss_mask == (1,)
    assert caught.value.rollout.failure == RolloutFailure(
        "RuntimeError", {"operation": "advance", "error": "Agent failed after receiving its response"}
    )
    assert "cleanup_errors" not in caught.value.grade.diagnostics
    # A Zephyr or Ray worker sends the exception to another process, which still gets the record and the grade.
    restored = pickle.loads(pickle.dumps(caught.value))
    assert type(restored) is GradedRolloutInterrupted
    assert (restored.rollout, restored.operation, restored.grade) == (
        caught.value.rollout,
        caught.value.operation,
        caught.value.grade,
    )


@pytest.mark.parametrize("cancel", ["child", "self"])
async def test_an_agent_that_ends_cancelled_fails_its_attempt_instead_of_cancelling_the_caller(cancel):
    async def agent(complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        if cancel == "child":
            child = asyncio.create_task(asyncio.sleep(60))
            child.cancel()
            await child  # The child's CancelledError ends the agent, although no task cancelled the agent.
        else:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(1)

    runner = engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        harnesses={"agent": AgentHarness(lambda _context, complete: agent(complete))},
    )
    with pytest.raises(GradedRolloutInterrupted) as caught:
        await runner.run_task(rollout_spec(arithmetic_task(), harness="agent"))
    assert caught.value.operation == RolloutOperation.ADVANCE
    assert caught.value.rollout.failure == RolloutFailure(
        "RuntimeError",
        {"operation": "advance", "error": "The agent ended with a CancelledError before the engine stopped it"},
    )
    assert caught.value.rollout.conversations[-1].response_token_ids == (20,)


async def test_agent_request_open_at_the_turn_limit_fails_at_the_attempt_cleanup():
    handlers = []

    async def agent(_context, complete):
        messages = ({"role": "user", "content": "Return 12."},)
        while True:
            # A server's request handler owns each request; waiting for it does not cancel it.
            handler = asyncio.create_task(complete(Prompt(messages, {})))
            handlers.append(handler)
            await asyncio.wait({handler})
            messages = (*messages, handler.result().message, {"role": "user", "content": "Continue."})

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", max_turns=1))
    assert record.rollout.stop_reason == "max_turns"
    await asyncio.wait({handlers[-1]}, timeout=5)
    assert len(handlers) == 2
    assert isinstance(handlers[-1].exception(), RolloutEnded)


@pytest.mark.parametrize("retry_content", ["Return 12.", "Return twelve."])
async def test_a_call_cancelled_while_the_model_generates_changes_nothing(retry_content):
    generating = asyncio.Event()
    first = Prompt(({"role": "user", "content": "Return 12."},), {})
    retry = Prompt(({"role": "user", "content": retry_content},), {})
    requests = []

    async def agent(_context, complete):
        call = asyncio.create_task(complete(first))
        await generating.wait()
        # The HTTP client disconnects while the model generates, then sends a request again.
        call.cancel()
        await asyncio.wait({call})
        await complete(retry)

    async def model(request):
        requests.append(request)
        if len(requests) == 1:
            generating.set()
            await asyncio.Future()
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), (-0.5,), "stop")

    record = await RolloutEngine({}, {"agent": AgentHarness(agent)}).run_task(
        rollout_spec(arithmetic_task(), harness="agent"), model
    )
    assert len(requests) == 2
    [conversation] = record.rollout.conversations
    assert conversation.messages == (*retry.messages, {"role": "assistant", "content": "12"})
    assert (conversation.response_token_ids, conversation.loss_mask) == ((20,), (1,))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_a_reply_cancelled_while_the_model_generates_can_be_sent_again():
    generating = asyncio.Event()
    prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
    observation = {"role": "user", "content": "Check it."}
    requests = []

    async def agent(_context, complete):
        turn = await complete(prompt)
        follow_up = Prompt((*prompt.messages, turn.message, observation), {})
        call = asyncio.create_task(complete(follow_up))
        await generating.wait()
        call.cancel()
        await asyncio.wait({call})
        await complete(follow_up)

    async def model(request):
        requests.append(request)
        if len(requests) == 2:
            generating.set()
            await asyncio.Future()
        served = (*request.continuation.served_token_ids, 90) if request.continuation is not None else (10, 11)
        return ModelTurn({"role": "assistant", "content": "12"}, served, (20 + len(requests),), (-0.5,), "stop")

    record = await RolloutEngine({}, {"agent": AgentHarness(agent)}).run_task(
        rollout_spec(arithmetic_task(), harness="agent"), model
    )
    assert requests[1] == requests[2]
    [conversation] = record.rollout.conversations
    assert [step.transition.observations for step in conversation.steps] == [(observation,), ()]
    assert (conversation.response_token_ids, conversation.loss_mask) == ((21, 90, 23), (1, 0, 1))


@pytest.mark.parametrize("next_options", [{}, {"tool_choice": "none"}])
async def test_a_new_conversation_replaces_the_model_options(next_options):
    initial_options = {"tool_choice": "required", "parallel_tool_calls": False}
    revised_messages = ({"role": "user", "content": "Return 12 without tools."},)

    class Restarting:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Use a tool."},), initial_options))
            await session.start(Prompt(revised_messages, next_options))
            await session.reply(())

    model = ReplayModel([{"role": "assistant", "content": "12"}] * 3)
    result = await engine(model, harnesses={"fixture": Restarting()}).run_task(
        rollout_spec(arithmetic_task(), harness="fixture")
    )

    assert [request.options for request in model.requests] == [initial_options, next_options, next_options]
    assert model.requests[1].messages == revised_messages
    assert result.grade.reward == 1.0
    first, revised = result.rollout.conversations
    assert (first.loss_mask, revised.loss_mask) == ((1,), (1, 0, 0, 1))


@pytest.mark.parametrize("trace", list(TraceMode))
@pytest.mark.parametrize("alter_prefix", [False, True])
async def test_session_replays_exact_initial_tokens_and_rejects_transport_changes(alter_prefix, trace):
    prefix = (128000, 42, 128005)

    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Replay case"},), {}, token_ids=prefix)

        async def advance(self, _turn):
            return Transition(done=True)

        async def close(self):
            pass

    async def model(request):
        assert request.prompt_token_ids == prefix
        served = (*prefix[:-1], 99) if alter_prefix else prefix
        return ModelTurn({"role": "assistant", "content": "done"}, served, (5, 6), (-0.1, -0.2), "stop")

    runner = RolloutEngine({}, {"fixture": StepHarness(lambda _context: Episode())})
    task = rollout_spec(arithmetic_task(), harness="fixture", trace=trace)
    if alter_prefix:
        # A chat endpoint cannot serve pinned tokens either, so the replay check holds in every trace mode.
        with pytest.raises(RolloutContractError, match="replay token prefix"):
            await runner.run_task(task, model)
        return
    result = await runner.run_task(task, model)
    assert (result.rollout.conversations[-1].prompt_token_ids, result.rollout.conversations[-1].response_token_ids) == (
        prefix,
        (5, 6),
    )
    if trace == TraceMode.EXACT_TOKENS:
        assert (result.rollout.conversations[-1].loss_mask, result.rollout.conversations[-1].logprobs) == (
            (1, 1),
            (-0.1, -0.2),
        )
    else:
        assert (result.rollout.conversations[-1].loss_mask, result.rollout.conversations[-1].logprobs) == ((0, 0), None)


@pytest.mark.parametrize("stop_reason", ["stop", "length"])
async def test_reasoning_only_response_grades_missing_answer_or_existing_files(stop_reason):
    empty = {"role": "assistant", "content": None, "reasoning": "Still working."}

    async def run(task, messages, **runtime):
        replay = ReplayModel(messages)

        async def complete(request):
            turn = await replay.complete(request)
            return replace(turn, stop_reason=stop_reason) if turn.message is empty else turn

        return await RolloutEngine({"local": FixtureImageFactory()}, shell_harnesses()).run_task(
            rollout_spec(task, **runtime), complete
        )

    missing = await run(arithmetic_task(), [empty])
    assert missing.grade.reward == 0.0
    assert missing.rollout.conversations[-1].messages[-1] == empty
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
async def test_spec_preserves_task_and_produces_private_grade_with_training_tokens(answer, reward):
    task = arithmetic_task()
    original = task.model_dump_json()
    spec = rollout_spec(task)
    model = ReplayModel([{"role": "assistant", "content": answer}])
    runner = engine(model)
    runner.check(spec)
    reloaded = TaskRolloutSpec.model_validate_json(spec.model_dump_json())

    result = await runner.run_task(reloaded)

    assert task.model_dump_json() == reloaded.task.model_dump_json() == original
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert result.rollout.conversations[-1].prompt_token_ids == (10, 11)
    assert result.rollout.conversations[-1].response_token_ids == (20,)
    assert result.rollout.conversations[-1].loss_mask == (1,)
    assert result.rollout.conversations[-1].logprobs == (-0.5,)
    assert "12" not in json.dumps(model.requests[0].messages)


@pytest.mark.parametrize("unresolved_role", ["actor", "grader"])
@pytest.mark.parametrize("entrypoint", ["check", "run"])
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
    spec = rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    reloaded = TaskRolloutSpec.model_validate_json(spec.model_dump_json())
    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": "Done."}])
    runner = engine(model, {"local": factory})
    with pytest.raises(UnsupportedMachineSpec):
        if entrypoint == "check":
            runner.check(reloaded)
        else:
            await runner.run_task(reloaded)
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
        await engine(model, {"local": factory}).run_task(
            rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
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
    record = await engine(model).run_task(rollout_spec(task))
    assert (record.grade.status, record.grade.reward) == (status, reward)
    assert "12" not in json.dumps(model.requests[0].messages)


@pytest.mark.parametrize("provider", ["machine", "harness"])
async def test_unknown_provider_fails_before_startup(provider):
    factory = RecordingShellSimFactory()
    spec = rollout_spec(
        arithmetic_task(),
        machine=machine_runtime(backend="missing") if provider == "machine" else None,
        harness="missing" if provider == "harness" else SHELL,
    )
    with pytest.raises(ValueError):
        await engine(ReplayModel([]), {"local": factory}).run_task(spec)
    assert factory.machines == []


async def test_machine_user_applies_to_custom_episodes_without_replacing_explicit_command_users():
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

    class Episode:
        def __init__(self, context):
            self.machine = context.machine

        async def prepare(self):
            agent = await self.machine.run(Command(("whoami",)))
            trusted = await self.machine.run(Command(("whoami",), user="0"))
            return Prompt(({"role": "user", "content": agent.stdout.decode() + ":" + trusted.stdout.decode()},), {})

        async def advance(self, turn):
            return Transition(done=True)

        async def close(self):
            pass

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]),
        {"local": Factory()},
        harnesses={"identity": StepHarness(Episode)},
    ).run_task(
        rollout_spec(
            arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
            machine=machine_runtime(user="learner"),
            harness="identity",
        )
    )
    assert record.rollout.conversations[0].messages[0]["content"] == "learner:0"
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
    result = await engine(
        model, {"local": FixtureImageFactory()}, harnesses=shell_harnesses(shell_tool=binding)
    ).run_task(rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime()))

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.rollout.conversations[-1].response_token_ids == (20, 90, 91, 21)
    assert result.rollout.conversations[-1].loss_mask == (1, 0, 0, 1)
    assert result.rollout.conversations[-1].logprobs == (-0.5, 0.0, 0.0, -0.5)
    assert model.requests[1].messages[-1]["tool_call_id"] == "write"
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    tools = model.requests[0].options["tools"]
    assert [tool["function"]["name"] for tool in tools] == [name]
    assert tools[0]["function"]["parameters"]["required"] == [parameter]
    if interface == "task-owned":
        assert tools[0]["function"] == task.interaction_tools[0].model_dump(exclude_none=True)
        assert model.requests[0].messages[0]["content"] == "Use Bash to write 12 to /workspace/answer."
    response = next(message for message in result.rollout.conversations[-1].messages if message["role"] == "assistant")
    assert response["tool_calls"][0]["function"]["name"] == name


async def test_shell_binding_collision_rejects_before_machine_or_model_start():
    task = file_task(final_tools=(FunctionDefinition(name="terminal", parameters={"type": "object"}),))
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(
            model, {"local": factory}, harnesses=shell_harnesses(shell_tool=ShellToolConfig(name="terminal"))
        ).run_task(rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime()))
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
    observer = RecordingObserver()
    record = await engine(
        model,
        {"local": FixtureImageFactory()},
        harnesses=shell_harnesses(tools={"checker:v1": provider}),
        observer=observer,
    ).run_task(rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime()))

    tool_events = [event.payload for event in observer.events if isinstance(event.payload, ToolCallCompleted)]
    assert [(event.call_id, event.name, event.done) for event in tool_events] == [
        ("write", "shell", False),
        ("rejected", "check", False),
        ("missing", "lookup", False),
        ("submit", "check", True),
    ]
    transitions = [event.payload for event in observer.events if isinstance(event.payload, TransitionCompleted)]
    assert transitions[-1].observations[-1] == {"role": "tool", "tool_call_id": "submit", "content": "12\n"}
    assert [tool["function"]["name"] for tool in model.requests[0].options["tools"]] == ["shell", "check"]
    observations = model.requests[1].messages[-3:]
    assert json.loads(observations[0]["content"])["exit_code"] == 0
    assert json.loads(observations[1]["content"]) == {"error": "Tool arguments must be a JSON object"}
    assert json.loads(observations[2]["content"]) == {"error": "Unknown tool: lookup"}
    [session] = provider.sessions
    assert (session.context.requirement, session.calls) == (CHECKER, [("check", {"path": "/workspace/answer"})])
    assert session.closed
    assert len(model.requests) == 2
    assert record.rollout.conversations[-1].steps[-1].transition.done
    assert record.rollout.conversations[-1].steps[-1].transition.observations[0]["content"] == "12\n"
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
    record = await engine(model, harnesses=shell_harnesses(tools={"checker:v1": provider})).run_task(rollout_spec(task))

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
        await engine(model, harnesses=shell_harnesses(tools={"checker:v1": provider})).run_task(rollout_spec(task))
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
        await engine(model, {"local": factory}, harnesses=shell_harnesses(tools={"checker:v1": provider})).run_task(
            rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
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
    record = await engine(model, {"local": Factory()}, harnesses=shell_harnesses(command_timeout=0.05)).run_task(
        rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), tool_turn_timeout=5)
    )
    observations = model.requests[1].messages[-3:]
    assert [item["tool_call_id"] for item in observations] == ["first", "second", "repair"]
    assert [json.loads(item["content"])["reason"] for item in observations] == ["timed_out", "timed_out", "exited"]
    assert json.loads(observations[-1]["content"])["exit_code"] == 0
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.rollout.failure is None
    assert record.rollout.conversations[-1].response_token_ids == (20, 90, 91, 21)
    assert record.rollout.conversations[-1].loss_mask == (1, 0, 0, 1)
    assert record.rollout.conversations[-1].logprobs == (-0.5, 0.0, 0.0, -0.5)


@pytest.mark.parametrize("tool_turn_timeout", [0.05, 0.01])
async def test_tool_turn_cannot_preempt_command_timeout_feedback(tool_turn_timeout):
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}, harnesses=shell_harnesses(command_timeout=0.05)).run_task(
            rollout_spec(
                file_task(),
                machine=machine_runtime(),
                verifier_machine=machine_runtime(),
                tool_turn_timeout=tool_turn_timeout,
            )
        )
    assert factory.machines == []
    assert model.requests == []


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
    result = await engine(model, {"local": FixtureImageFactory()}).run_task(
        rollout_spec(task, machine=machine_runtime())
    )
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
    record = await engine(model).run_task(rollout_spec(task))
    assert record.grade.status == expected
    assert model.requests[0].options["tool_choice"] == "required"
    assert model.requests[0].options["parallel_tool_calls"] is False


async def test_workspace_state_receives_shell_presentation_without_answer_tools():
    task = file_task(answer_type=AnswerType.WORKSPACE_STATE, answer_format=AnswerCall())
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": FixtureImageFactory()}).run_task(
        rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
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
    ],
    ids=["verifyit_environment", "script", "in_process", "none"],
)
async def test_grader_without_matching_runtime_is_rejected_before_task_startup(grader, verifier_machine):
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}).run_task(
            rollout_spec(arithmetic_task(grader=grader), machine=machine_runtime(), verifier_machine=verifier_machine)
        )
    assert factory.machines == []
    assert model.requests == []


def workspace_task() -> TaskSpec:
    return arithmetic_task(
        answer_type=AnswerType.FILE,
        grader=workspace_grader(
            argv=("sh", "/tests/grade.sh"),
            artifacts=(
                VerifierArtifact(source="/workspace/answer", target="/workspace/answer", kind=ArtifactKind.FILE),
            ),
        ),
        resources=ResourceGroups(verifier=(inline_resource("grade.sh", b"echo 1"),)),
    )


def bash_binding_task() -> TaskSpec:
    definition = FunctionDefinition(
        name="Bash",
        parameters={
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
            "additionalProperties": False,
        },
    )
    return arithmetic_task(
        interaction_tools=(definition,), tool_bindings={"Bash": ShellToolBinding(command_parameter="script")}
    )


JUDGE_CONNECTION = JudgeConnection("http://judge.invalid/v1", "private-test-key")


@pytest.mark.parametrize(
    "spec,error,message",
    [
        # The engine does not capture state, so it rejects a graded state task for every harness.
        pytest.param(
            lambda: rollout_spec(
                file_task(answer_type=AnswerType.STATE), machine=machine_runtime(), verifier_machine=machine_runtime()
            ),
            NotImplementedError,
            "state answer",
            id="state-shell",
        ),
        pytest.param(
            lambda: rollout_spec(
                file_task(answer_type=AnswerType.STATE),
                harness="step",
                machine=machine_runtime(),
                verifier_machine=machine_runtime(),
            ),
            NotImplementedError,
            "state answer",
            id="state-step",
        ),
        pytest.param(
            lambda: rollout_spec(workspace_task(), verifier_machine=machine_runtime()),
            ValueError,
            "Workspace tasks require a task machine",
            id="workspace-shell",
        ),
        pytest.param(
            lambda: rollout_spec(workspace_task(), verifier_machine=machine_runtime(), harness="agent"),
            ValueError,
            "Workspace tasks require a task machine",
            id="workspace-agent",
        ),
        # The agent never gets the submit_answer tool, so the grader would reject every answer.
        pytest.param(
            lambda: rollout_spec(arithmetic_task(answer_format=AnswerCall()), harness="agent"),
            NotImplementedError,
            "task tools",
            id="answer-call-agent",
        ),
        pytest.param(lambda: rollout_spec(judge_task()), ValueError, BASE_URL_ENV, id="judge-without-service"),
        pytest.param(
            lambda: rollout_spec(judge_task(model=""), judge_connection=JUDGE_CONNECTION),
            ValueError,
            MODEL_ENV,
            id="judge-without-model",
        ),
        pytest.param(
            lambda: rollout_spec(judge_task(AnswerType.FILE), judge_connection=JUDGE_CONNECTION),
            NotImplementedError,
            "conversation answer",
            id="judge-file-answer",
        ),
        pytest.param(
            lambda: rollout_spec(judge_task(), verifier_machine=machine_runtime(), judge_connection=JUDGE_CONNECTION),
            ValueError,
            "does not use a verifier machine",
            id="judge-verifier-machine",
        ),
        pytest.param(
            lambda: rollout_spec(bash_binding_task()),
            ValueError,
            "A shell binding requires a task machine",
            id="shell-binding-without-machine",
        ),
        pytest.param(
            lambda: rollout_spec(arithmetic_task(output_paths=("/workspace/answer.txt",))),
            ValueError,
            "Output paths require a task machine",
            id="output-paths-without-machine",
        ),
        pytest.param(
            lambda: rollout_spec(
                arithmetic_task(environment_requirements=EnvironmentRequirements(capabilities=("shell",)))
            ),
            ValueError,
            "need a selected machine",
            id="shell-capability-without-machine",
        ),
    ],
)
async def test_a_spec_that_the_engine_cannot_run_is_rejected_before_machines_and_model_requests(
    spec, error, message, monkeypatch
):
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)

    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))

    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    harnesses = {"step": StepHarness(lambda _context: FinishingEpisode()), "agent": AgentHarness(agent)}
    with pytest.raises(error, match=message):
        await engine(model, {"local": factory}, harnesses=harnesses).run_task(spec())
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
    record = await engine(model, {"local": factory}).run_task(rollout_spec(task, machine=machine_runtime()))
    assert (record.grade.status, record.grade.reward, record.grade.error) == (Outcome.UNAVAILABLE, None, reason)
    assert record.rollout.conversations[-1].loss_mask == (1, 0, 0, 1)
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
    record = await engine(ReplayModel([{"role": "assistant", "content": answer}]), {"local": factory}).run_task(
        rollout_spec(task, verifier_machine=machine_runtime())
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
    record = await engine(model, {"local": factory, "simulator": simulator}).run_task(
        rollout_spec(
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
                served = () if request.continuation is None else request.continuation.served_token_ids
                raise GenerationLimitReached((*served, 90, 91))
            return await super().complete(request)

    record = await engine(
        LimitedModel([shell_call("echo 12 > /workspace/answer")]), {"local": FixtureImageFactory()}
    ).run_task(rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()))
    assert record.rollout.stop_reason == "length"
    assert (record.grade.status, record.grade.reward) == (
        (Outcome.GRADED, 1.0) if completed_turns else (Outcome.UNAVAILABLE, None)
    )
    assert record.rollout.conversations[-1].response_token_ids == ((20,) if completed_turns else ())
    assert record.rollout.conversations[-1].loss_mask == ((1,) if completed_turns else ())


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
    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, _turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue."},))

        async def close(self):
            pass

    class Model(ReplayModel):
        async def complete(self, request):
            return replace(await super().complete(request), usage=usage)

    model = Model([{"role": "assistant", "content": "12"}] * 3)
    record = await engine(model, harnesses={"fixture": StepHarness(lambda _context: Episode())}).run_task(
        rollout_spec(arithmetic_task(), harness="fixture", max_total_tokens=budget)
    )
    assert len(model.requests) == requests
    assert record.rollout.stop_reason == stop_reason
    assert record.rollout.conversations[-1].messages[-1]["role"] == "assistant"
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_token_budget_skips_a_new_conversation_and_grades_the_last_response():
    class Restarting:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
            while True:
                await session.start(prompt)
                prompt = Prompt(({"role": "user", "content": "Start again."},), {})

    model = ReplayModel([{"role": "assistant", "content": "12"}] * 3)
    # Each request of a new conversation charges 3 tokens; after the second, a third would reach the budget of 9.
    observer = RecordingObserver()
    record = await engine(model, harnesses={"fixture": Restarting()}, observer=observer).run_task(
        rollout_spec(arithmetic_task(), harness="fixture", max_total_tokens=9)
    )
    assert len(model.requests) == 2
    assert record.rollout.stop_reason == "budget_reached"
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert [conversation.messages[0]["content"] for conversation in record.rollout.conversations] == [
        "Return 12.",
        "Start again.",
    ]
    started = [event.payload for event in observer.events if isinstance(event.payload, ConversationStarted)]
    assert len(started) == 2


@pytest.mark.parametrize("usage", [None, TokenUsage(prompt_tokens=4, completion_tokens=1)])
async def test_token_budget_under_messages_trace_needs_usage_or_token_ids(usage):
    class UsageChatModel(ChatModel):
        async def complete(self, request):
            return replace(await super().complete(request), usage=usage)

    model = UsageChatModel([{"role": "assistant", "content": "12"}])
    run = engine(model).run_task(rollout_spec(arithmetic_task(), trace=TraceMode.MESSAGES, max_total_tokens=100))
    if usage is None:
        with pytest.raises(RolloutContractError, match="token budget"):
            await run
        return
    record = await run
    assert (record.rollout.stop_reason, record.grade.reward) == ("stop", 1.0)


@pytest.mark.parametrize("violation", ["prefix", "logprobs", "empty"])
async def test_model_transport_must_preserve_exact_token_evidence(violation):
    async def complete(request):
        if request.continuation is not None:
            return ModelTurn({"role": "assistant", "content": "Done."}, (99,), (30,), (-0.5,), "stop")
        return ModelTurn(
            shell_call("echo 12 > /workspace/answer"),
            (10, 11),
            () if violation == "empty" else (20,),
            (-0.5, -0.5) if violation == "logprobs" else () if violation == "empty" else (-0.5,),
            "stop",
        )

    runner = RolloutEngine({"local": FixtureImageFactory()}, shell_harnesses())
    with pytest.raises(RolloutContractError):
        await runner.run_task(
            rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()), complete
        )


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
    spec = rollout_spec(
        file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), trace=TraceMode.MESSAGES
    )
    model = ChatModel(replies)

    record = await engine(model, {"local": FixtureImageFactory()}).run_task(spec)

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.rollout.stop_reason == "stop"
    assert record.rollout.conversations[-1].messages[-1] == replies[-1]
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    assert [request.continuation for request in model.requests] == [None, None]
    assert len(record.rollout.conversations[-1].steps) == 2
    assert (
        record.rollout.conversations[-1].prompt_token_ids,
        record.rollout.conversations[-1].response_token_ids,
        record.rollout.conversations[-1].loss_mask,
        record.rollout.conversations[-1].logprobs,
    ) == ((), (), (), None)
    assert record.rollout.metrics["trace"] == "messages"


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
    spec = rollout_spec(
        file_task(), machine=machine_runtime(), verifier_machine=machine_runtime(), trace=TraceMode.MESSAGES
    )

    record = await engine(RetokenizingChatModel(replies), {"local": FixtureImageFactory()}).run_task(spec)

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    # The last served sequence minus the first prompt's length: re-tokenization makes it informational only.
    assert (record.rollout.conversations[-1].prompt_token_ids, record.rollout.conversations[-1].response_token_ids) == (
        (10, 11),
        (98, 97, 21),
    )
    assert (record.rollout.conversations[-1].loss_mask, record.rollout.conversations[-1].logprobs) == ((0, 0, 0), None)
    assert record.rollout.metrics["trace"] == "messages"


async def test_interrupted_message_trace_record_is_marked():
    class UnavailableModel:
        async def complete(self, request):
            raise ConnectionError("chat endpoint unavailable")

    spec = rollout_spec(arithmetic_task(), trace=TraceMode.MESSAGES)
    with pytest.raises(RolloutInterrupted) as interrupted:
        await engine(UnavailableModel()).run_task(spec)

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
    ).run_task(rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime()))
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
    limited = asyncio.create_task(runner.run_task(rollout_spec(task, total_turn_timeout=0.05)))
    unlimited = asyncio.create_task(runner.run_task(rollout_spec(task)))
    try:
        stopped = await asyncio.wait_for(limited, timeout=5)
        assert (stopped.rollout.stop_reason, stopped.grade.status) == ("total_turn_timeout", Outcome.UNAVAILABLE)
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
async def test_deadlines_keep_generated_tokens_and_close_episodes(phase, budget):
    closed = asyncio.Event()
    # A sandbox grader blocks while its verifier machine starts, the only machine of this attempt.
    factory = BlockingFactory(blocked=0)
    entered = factory.entered

    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, turn):
            if phase == "advance":
                entered.set()
                await asyncio.Future()
            return Transition(done=phase == "grade")

        async def close(self):
            closed.set()

    class Model(ReplayModel):
        async def complete(self, request):
            if phase == "model" and self.requests:
                entered.set()
                await asyncio.Future()
            return await super().complete(request)

    limits = {"harness": "fixture"}
    if budget == "attempt":
        limits["attempt_timeout"] = 0.1
    elif budget == "total_turn":
        limits["total_turn_timeout"] = 0.1
    else:
        limits[{"model": "model_turn_timeout", "advance": "tool_turn_timeout", "grade": "verifier_timeout"}[phase]] = 0.1
    task, verifier = arithmetic_task(), None
    if phase == "grade":
        task, verifier = (
            arithmetic_task(grader=ScriptGrader(argv=("true",), environment=GRADER_ENVIRONMENT)),
            machine_runtime(),
        )
    pending = asyncio.create_task(
        engine(
            Model([{"role": "assistant", "content": "12"}]),
            {"local": factory},
            harnesses={"fixture": StepHarness(lambda _context: Episode())},
        ).run_task(rollout_spec(task, verifier_machine=verifier, **limits))
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if budget == "total_turn":
            result = await pending
            rollout = result.rollout
            assert rollout.stop_reason == "total_turn_timeout"
            assert result.grade.reward == 1.0
        else:
            with pytest.raises(RolloutInterrupted) as caught:
                await pending
            rollout = caught.value.rollout
            assert caught.value.operation == (
                RolloutOperation.ATTEMPT
                if budget == "attempt"
                else {
                    "model": RolloutOperation.MODEL,
                    "advance": RolloutOperation.ADVANCE,
                    "grade": RolloutOperation.GRADE,
                }[phase]
            )
            assert isinstance(caught.value.__cause__, TimeoutError)
            if phase == "grade" and budget == "per_phase":
                assert caught.value.grade.failure == GradingFailure.TIMEOUT
    finally:
        factory.release.set()
    assert rollout.conversations[-1].response_token_ids == (20,)
    assert rollout.conversations[-1].loss_mask == (1,)
    assert rollout.conversations[-1].logprobs == (-0.5,)
    assert closed.is_set()


async def test_turn_budget_covers_all_turns_and_excludes_final_grading():
    close = asyncio.Event()

    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue."},))

        async def close(self):
            close.set()

    class Model(ReplayModel):
        async def complete(self, request):
            if self.requests:
                await asyncio.Future()
            return await super().complete(request)

    factory = BlockingFactory(blocked=0)
    task = arithmetic_task(grader=ScriptGrader(argv=("sh", "-c", "echo 1"), environment=GRADER_ENVIRONMENT))
    pending = asyncio.create_task(
        engine(
            Model([{"role": "assistant", "content": "12"}]),
            {"local": factory},
            harnesses={"fixture": StepHarness(lambda _context: Episode())},
        ).run_task(rollout_spec(task, verifier_machine=machine_runtime(), harness="fixture", total_turn_timeout=0.05))
    )
    try:
        await asyncio.wait_for(factory.entered.wait(), timeout=5)
        assert not pending.done()
    finally:
        factory.release.set()
    record = await pending
    assert record.grade.reward == 1.0
    assert record.rollout.stop_reason == "total_turn_timeout"
    assert record.rollout.conversations[-1].messages[-1]["role"] == "assistant"
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

    spec = rollout_spec(
        arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
        machine=machine_runtime(startup_timeout=0.05 if interruption == "startup" else None),
        attempt_timeout=0.05 if interruption == "attempt" else None,
    )
    pending = asyncio.create_task(engine(ReplayModel([]), {"local": Factory()}).run_task(spec))
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

    spec = rollout_spec(
        arithmetic_task(environment_requirements=GRADER_ENVIRONMENT),
        machine=machine_runtime(cleanup_timeout=0.05),
        cleanup_timeout=5,
        attempt_timeout=0.01,
    )
    pending = asyncio.create_task(
        engine(ReplayModel([{"role": "assistant", "content": "12"}]), {"local": Factory()}).run_task(spec)
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
        await engine(ReplayModel([]), {"local": Factory()}).run_task(rollout_spec(task, machine=machine_runtime()))
    rollout, grade = caught.value.rollout, caught.value.grade
    assert caught.value.operation == RolloutOperation.START
    assert isinstance(caught.value.__cause__, ConnectionError)
    assert closed.is_set()
    assert rollout.conversations == ()
    assert grade.diagnostics["cleanup_errors"] == [{"operation": "machine_close", "exception_type": "OSError"}]
    assert "private" not in json.dumps(grade.diagnostics)


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

    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Return twelve."},), {})

        async def advance(self, turn):
            if phase == "advance":
                await wait_for_deadline()
            return Transition(done=True)

        async def close(self):
            closed.set()

    class CaptureMachine:
        """A task machine whose output capture, the first grading step, waits for the deadline."""

        async def run(self, command):
            await wait_for_deadline()

        async def download(self, source, target):
            await wait_for_deadline()

        async def close(self):
            pass

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return CaptureMachine()

    task = arithmetic_task(environment_requirements=GRADER_ENVIRONMENT, output_paths=("/workspace/out",))
    with pytest.raises(asyncio.CancelledError):
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}]),
            {"local": Factory()},
            harnesses={"fixture": StepHarness(lambda _context: Episode())},
        ).run_task(rollout_spec(task, machine=machine_runtime(), harness="fixture", attempt_timeout=0.01))
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

    class Episode:
        def __init__(self, context):
            self.machine = context.machine

        async def prepare(self):
            return Prompt(({"role": "user", "content": "Run the task."},), {})

        async def advance(self, turn):
            result = await self.machine.run(Command(("whoami",)))
            assert (result.exit_code, result.stdout) == (0, b"learner")
            return Transition(done=True)

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
        ReplayModel([{"role": "assistant", "content": "12"}]),
        {"local": Factory()},
        harnesses={"fixture": StepHarness(Episode)},
    ).run_task(rollout_spec(task, machine=machine_runtime(user="learner"), harness="fixture"))
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
        await engine(ReplayModel([]), {"local": Factory()}).run_task(rollout_spec(task, machine=machine_runtime()))

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
        await engine(model, {"local": Factory()}).run_task(rollout_spec(task, machine=machine_runtime(user="learner")))
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
    spec = rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    if failure in {"download", "download_and_remove"}:
        with pytest.raises(RolloutInterrupted) as caught:
            await runner.run_task(spec)
        assert caught.value.operation == RolloutOperation.GRADE
        assert isinstance(caught.value.__cause__, ConnectionError)
        rollout, grade = caught.value.rollout, caught.value.grade
        assert (grade.status, grade.reward) == (Outcome.UNAVAILABLE, None)
    elif failure == "remove":
        result = await runner.run_task(spec)
        rollout, grade = result.rollout, result.grade
        assert (grade.status, grade.reward, grade.failure) == (Outcome.INFRA_ERROR, None, GradingFailure.EXECUTION)
        # The archive stays on the task machine, so the grader never starts.
        assert len(factory.machines) == 1
    else:
        result = await runner.run_task(spec)
        rollout, grade = result.rollout, result.grade
        assert (grade.status, grade.reward) == (Outcome.GRADED, 1.0)
    assert rollout.conversations[-1].response_token_ids == (20,)
    assert "cleanup_errors" not in grade.diagnostics
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
    record = await engine(model, {"local": Factory()}).run_task(
        rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    assert (record.grade.status, record.grade.reward) == (expected_status, expected_reward)
    if answer_type == AnswerType.NUMBER and status == "scored":
        candidate_record = await engine(ReplayModel([{"role": "assistant", "content": answer}])).run_task(
            rollout_spec(arithmetic_task())
        )
        assert (record.grade.status, record.grade.reward, record.grade.error) == (
            candidate_record.grade.status,
            candidate_record.grade.reward,
            candidate_record.grade.error,
        )
    if expected_status == Outcome.INFRA_ERROR:
        assert record.grade.failure == GradingFailure.INVALID_REWARD
    if answer_type == AnswerType.FILE:
        assert record.rollout.conversations[-1].loss_mask == (1, 0, 0, 1)
        assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    else:
        assert record.rollout.conversations[-1].loss_mask == (1,)
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@dataclass
class RecordingObserver:
    events: list[RolloutEvent] = field(default_factory=list)

    async def on_event(self, event: RolloutEvent) -> None:
        self.events.append(event)

    def payloads(self) -> list[RolloutEventPayload]:
        """The untimed payloads, after checking that every event belongs to one attempt of the arithmetic task."""
        assert {(event.run_id, event.task_id) for event in self.events} == {(self.events[0].run_id, "arithmetic")}
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
    observation = model.requests[1].messages[-1]
    return [
        AttemptStarted(),
        MachineReady(0.0),
        ConversationStarted(model.requests[0].messages, model.requests[0].options, 0.0),
        ModelTurnCompleted(0, model.requests[1].messages[-2], "stop", 2, 1, None, {}, 0.0),
        ToolCallCompleted("write", "shell", len(observation["content"].encode()), False, 0.0),
        TransitionCompleted(0, False, (observation,), 0.0),
    ]


async def test_observer_receives_ordered_events_for_a_multi_turn_shell_rollout():
    observer = RecordingObserver()
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": FixtureImageFactory()}, observer=observer).run_task(
        rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()), run_id="arithmetic/3"
    )
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert observer.payloads() == [
        *shell_turn_payloads(model),
        ModelTurnCompleted(1, record.rollout.conversations[-1].messages[-1], "stop", 5, 1, None, {}, 0.0),
        TransitionCompleted(1, True, (), 0.0),
        Graded(Outcome.GRADED, 1.0, None, None, 0.0),
        AttemptEnded("stop", None, None, 0.0),
    ]
    assert asdict(observer.events[-1]) == {
        "run_id": "arithmetic/3",
        "task_id": "arithmetic",
        "payload": {
            "stop_reason": "stop",
            "operation": None,
            "error": None,
            "duration": observer.events[-1].payload.duration,
            "type": RolloutEventType.ATTEMPT_ENDED,
        },
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
        await engine(model, {"local": FixtureImageFactory()}, observer=observer).run_task(
            rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
        )
    assert interrupted.value.operation == RolloutOperation.MODEL
    assert observer.payloads() == [
        *shell_turn_payloads(model),
        Graded(Outcome.GRADED, 1.0, None, None, 0.0),
        AttemptEnded(None, RolloutOperation.MODEL, repr(ConnectionError("inference server unavailable")), 0.0),
    ]


async def test_attempts_without_a_run_id_keep_their_events_apart():
    observer = RecordingObserver()
    runner = engine(ReplayModel([{"role": "assistant", "content": "12"}] * 2), observer=observer)
    results = [await runner.run_task(rollout_spec(arithmetic_task())) for _ in range(2)]
    run_ids = [event.run_id for event in observer.events]
    assert len(set(run_ids)) == 2
    assert [result.rollout.run_id for result in results] == list(dict.fromkeys(run_ids))


async def test_observer_failure_leaves_the_rollout_unchanged():
    class FailingObserver:
        async def on_event(self, event: RolloutEvent) -> None:
            if event.payload.type == RolloutEventType.MODEL_TURN_COMPLETED:
                raise RuntimeError("telemetry sink unavailable")

    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    record = await engine(model, {"local": FixtureImageFactory()}, observer=FailingObserver()).run_task(
        rollout_spec(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    )

    assert (record.grade.status, record.grade.reward, record.rollout.stop_reason) == (Outcome.GRADED, 1.0, "stop")
    # Three events precede the failing first model turn; the six after it are never delivered.
    assert (record.rollout.metrics["observer_error_count"], record.rollout.metrics["observer_undelivered_events"]) == (
        1.0,
        6.0,
    )


async def test_blocked_observer_does_not_hold_the_rollout():
    class BlockedObserver:
        async def on_event(self, event: RolloutEvent) -> None:
            await asyncio.Event().wait()

    model = ReplayModel([{"role": "assistant", "content": "12"}])
    record = await engine(model, observer=BlockedObserver()).run_task(
        rollout_spec(arithmetic_task(), attempt_timeout=5, cleanup_timeout=0.1)
    )

    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.rollout.metrics["observer_undelivered_events"] > 0


async def test_steps_keep_the_harness_data_of_each_transition():
    class Episode:
        def __init__(self):
            self.turns = 0

        async def prepare(self):
            return Prompt(({"role": "user", "content": "Return 12."},), {})

        async def advance(self, turn):
            self.turns += 1
            return Transition(
                done=self.turns == 2, observations=({"role": "user", "content": "Again."},), data=("reward", self.turns)
            )

        async def close(self):
            pass

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}] * 2),
        harnesses={"fixture": StepHarness(lambda _context: Episode())},
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert [step.transition.data for step in record.rollout.conversations[-1].steps] == [("reward", 1), ("reward", 2)]
    # The engine grades with the task's grader; the episode only handled the turns.
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_an_agent_with_two_open_requests_ends_only_its_own_attempt():
    # Callers such as SkyRL drop an interrupted attempt and stop on a token-contract error.
    async def agent(_context, complete):
        prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
        turn = await complete(prompt)
        follow_up = Prompt((*prompt.messages, turn.message, {"role": "user", "content": "Again."}), {})
        calls = [asyncio.create_task(complete(follow_up)) for _ in range(2)]
        await asyncio.wait(calls)
        for call in calls:
            call.result()

    class SlowModel(ReplayModel):
        async def complete(self, request):
            # The model yields once, so the second request comes while the first one is open.
            await asyncio.sleep(0)
            return await super().complete(request)

    model = SlowModel([{"role": "assistant", "content": "12"}] * 2)
    with pytest.raises(RolloutInterrupted) as interrupted:
        await engine(model, harnesses={"agent": AgentHarness(agent)}).run_task(
            rollout_spec(arithmetic_task(), harness="agent")
        )
    assert (interrupted.value.operation, type(interrupted.value.__cause__)) == (RolloutOperation.ADVANCE, RuntimeError)
    assert len(model.requests) == 2


async def test_a_request_that_the_agent_cancels_and_leaves_stays_in_the_record_and_fails_the_attempt():
    generating = asyncio.Event()

    async def agent(_context, complete):
        call = asyncio.create_task(complete(Prompt(({"role": "user", "content": "Return 12."},), {})))
        await generating.wait()
        # The client disconnects, and the agent exits without a retry.
        call.cancel()
        await asyncio.wait({call})

    async def model(_request):
        generating.set()
        await asyncio.Future()

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await RolloutEngine({}, {"agent": AgentHarness(agent)}).run_task(
            rollout_spec(arithmetic_task(), harness="agent"), model
        )
    # The attempt has no model turn, as for a harness that returns before its first request.
    assert caught.value.operation == RolloutOperation.PREPARE
    [conversation] = caught.value.rollout.conversations
    assert (conversation.messages, conversation.steps) == (({"role": "user", "content": "Return 12."},), ())
    assert caught.value.grade.status == Outcome.UNAVAILABLE


async def test_closing_answers_a_server_request_before_the_agent_drains_its_handlers():
    handlers = []

    async def agent(_context, complete):
        messages = ({"role": "user", "content": "Return 12."},)
        try:
            while True:
                handler = asyncio.create_task(complete(Prompt(messages, {})))
                handlers.append(handler)
                await asyncio.wait({handler})
                messages = (*messages, handler.result().message, {"role": "user", "content": "Continue."})
        finally:
            # A server that shuts down waits for its open handlers.
            await asyncio.wait(handlers)

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", max_turns=1, cleanup_timeout=5))

    assert isinstance(handlers[-1].exception(), RolloutEnded)
    assert "cleanup_errors" not in record.grade.diagnostics


async def test_a_cancelled_stop_propagates_and_lets_the_agent_finish_stopping():
    # The caller cancels the rollout again while the agent drains its handlers.
    agent_task = None
    entered = asyncio.Event()
    stopping = asyncio.Event()
    release = asyncio.Event()
    drained = False

    async def agent(_context, complete):
        nonlocal agent_task, drained
        agent_task = asyncio.current_task()
        try:
            await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        except asyncio.CancelledError:
            stopping.set()
            await release.wait()
            drained = True
            raise

    async def model(_request):
        entered.set()
        await asyncio.Future()

    pending = asyncio.create_task(
        harness_rollout(AgentHarness(agent), arithmetic_task(), model, limits=rollout_limits(cleanup_timeout=0.1))
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    pending.cancel()
    await asyncio.wait_for(stopping.wait(), timeout=5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()
    assert agent_task is not None
    with pytest.raises(asyncio.CancelledError):
        await agent_task
    assert drained


async def test_the_last_prompt_again_gets_the_same_turn_without_a_new_generation():
    prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
    received = []

    async def agent(_context, complete):
        received.append(await complete(prompt))
        received.append(await complete(prompt))

    model = ReplayModel([{"role": "assistant", "content": "12"}])
    record = await engine(model, harnesses={"agent": AgentHarness(agent)}).run_task(
        rollout_spec(arithmetic_task(), harness="agent")
    )
    assert len(model.requests) == 1
    assert received[0] is received[1]
    assert (record.rollout.conversations[-1].response_token_ids, record.grade.reward) == ((20,), 1.0)


async def test_an_ungraded_workspace_task_needs_no_task_machine():
    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Write the report."},), {}))

    task = arithmetic_task(answer_type=AnswerType.FILE, grader=NoGrader(reason="No evaluator"))
    record = await engine(
        ReplayModel([{"role": "assistant", "content": "Done."}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(task, harness="agent"))

    assert (record.grade.status, record.grade.error) == (Outcome.UNAVAILABLE, "No evaluator")


async def test_a_harness_runs_alone_on_a_local_machine():
    task = file_task()
    model = ReplayModel([shell_call("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    async with task_machine(task, FixtureImageFactory(), machine_runtime(), cleanup_timeout=5) as machine:
        rollout = await harness_rollout(
            ShellHarness(command_timeout=None), task, model.complete, machine=machine, limits=rollout_limits()
        )
        written = await machine.run(Command(("cat", "/workspace/answer")))
    assert written.stdout == b"12\n"
    [conversation] = rollout.conversations
    assert (conversation.response_token_ids, conversation.loss_mask) == ((20, 90, 91, 21), (1, 0, 0, 1))
    assert rollout.stop_reason == "stop"


async def test_a_harness_that_ignores_a_limit_runs_until_the_turn_deadline():
    class Stubborn:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            while True:
                try:
                    await session.reply(({"role": "user", "content": "Again."},))
                except LimitReached:
                    continue

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": Stubborn()}
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture", max_turns=1, total_turn_timeout=0.05))
    assert (record.rollout.stop_reason, record.grade.reward) == ("max_turns", 1.0)


class FinishingEpisode:
    async def prepare(self):
        return Prompt(({"role": "user", "content": "Return 12."},), {})

    async def advance(self, turn):
        return Transition(done=True)

    async def close(self):
        pass


async def test_a_failed_close_is_a_cleanup_error_of_a_graded_attempt():
    class FailingClose(FinishingEpisode):
        async def close(self):
            raise OSError("close failed")

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        harnesses={"fixture": StepHarness(lambda _context: FailingClose())},
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.grade.diagnostics["cleanup_errors"] == [{"operation": "episode_close", "exception_type": "OSError"}]
    assert record.rollout.conversations[-1].steps[-1].transition.done


async def test_an_agent_that_does_not_stop_is_a_cleanup_error_and_is_kept_until_it_ends(caplog):
    release = asyncio.Event()
    ended = asyncio.Event()

    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        try:
            await complete(Prompt(({"role": "user", "content": "Again."},), {}))
        finally:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            ended.set()
            raise ValueError("The agent failed after its cleanup deadline")

    runner = engine(ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)})
    try:
        record = await asyncio.wait_for(
            runner.run_task(rollout_spec(arithmetic_task(), harness="agent", max_turns=1, cleanup_timeout=0.1)),
            timeout=5,
        )
    finally:
        release.set()
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.grade.diagnostics["cleanup_errors"] == [{"operation": "agent_stop", "exception_type": "TimeoutError"}]
    await asyncio.wait_for(ended.wait(), timeout=5)
    await asyncio.sleep(0)
    # The cleanup logs the kept agent's failure. Without it, only asyncio would log the task that nobody awaited.
    assert any(
        record.name == "rolloutengine.cleanup" and record.exc_info and isinstance(record.exc_info[1], ValueError)
        for record in caplog.records
    )


async def test_recorded_step_data_keeps_the_tool_deadline():
    class SlowTool:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            session.record(metrics={"reward": 1.0})
            await asyncio.Future()

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": SlowTool()}).run_task(
            rollout_spec(arithmetic_task(), harness="fixture", tool_turn_timeout=0.05)
        )
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, TimeoutError)


@pytest.mark.parametrize("error", [RuntimeError, TimeoutError])
async def test_a_harness_failure_after_the_turn_deadline_is_not_a_normal_stop(error):
    class FailingOnCancel:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                # For example, the harness's own bounded cleanup timed out.
                raise error("The harness failed while it stopped") from None

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": FailingOnCancel()}
        ).run_task(rollout_spec(arithmetic_task(), harness="fixture", total_turn_timeout=0.05))
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, error)


async def test_observations_recorded_before_a_new_conversation_stay_in_the_earlier_one():
    note = {"role": "user", "content": "Noted."}

    class Restarting:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            session.record(observations=(note,))
            await session.start(Prompt(({"role": "user", "content": "Start again."},), {}))

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}] * 2), harnesses={"fixture": Restarting()}
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    earlier, _ = record.rollout.conversations
    assert earlier.steps[-1].transition.observations == (note,)


async def test_a_standalone_rollout_checks_the_task_before_the_model_runs():
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    with pytest.raises(NotImplementedError):
        await harness_rollout(
            AgentHarness(lambda _context, complete: complete(Prompt((), {}))),
            arithmetic_task(answer_format=AnswerCall()),
            model.complete,
            limits=rollout_limits(),
        )
    assert model.requests == []


async def test_a_judge_task_grades_on_the_host_without_a_verifier_machine(judge_server):
    factory = RecordingShellSimFactory()
    model = ReplayModel([{"role": "assistant", "content": "twelve"}])
    connection = JudgeConnection(judge_url(judge_server), "private-test-key")
    record = await engine(model, {"local": factory}).run_task(rollout_spec(judge_task(), judge_connection=connection))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert factory.machines == []
    [request] = judge_server.requests
    assert "Reference: 12\nCandidate: twelve" in request["messages"][-1]["content"]


async def test_a_judge_task_without_a_connection_uses_the_environment(judge_server, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, judge_url(judge_server))
    monkeypatch.setenv(API_KEY_ENV, "private-test-key")
    monkeypatch.setenv(MODEL_ENV, "environment-judge")
    judge_server.reply = "FAIL"
    model = ReplayModel([{"role": "assistant", "content": "13"}])
    record = await engine(model).run_task(rollout_spec(judge_task(model="")))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 0.0)
    [request] = judge_server.requests
    assert request["model"] == "environment-judge"


def test_a_judge_connection_never_reaches_the_serialized_spec():
    spec = rollout_spec(judge_task(), judge_connection=JudgeConnection("http://judge.invalid/v1", "private-test-key"))
    serialized = spec.model_dump_json()
    assert "private-test-key" not in serialized
    assert TaskRolloutSpec.model_validate_json(serialized).judge_connection is None
    pickled = pickle.dumps(spec)
    assert b"private-test-key" not in pickled
    restored = pickle.loads(pickled)
    assert (restored.judge_connection, restored.task, restored.limits) == (None, spec.task, spec.limits)
    # The spec in memory keeps its connection.
    assert spec.judge_connection is not None


async def test_a_harness_that_catches_its_tool_deadline_cannot_block_the_event_loop():
    class SwallowsDeadline:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
            while True:
                try:
                    await session.reply(({"role": "user", "content": "Again."},))
                except RolloutEnded:
                    continue

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": SwallowsDeadline()}
        ).run_task(rollout_spec(arithmetic_task(), harness="fixture", tool_turn_timeout=0.05, total_turn_timeout=0.2))
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, TimeoutError)


@pytest.mark.parametrize("asks_again", [False, True])
async def test_a_harness_that_catches_its_attempt_deadline_gets_no_more_turns(asks_again):
    class SwallowsDeadline:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
            while asks_again:
                try:
                    await session.reply(({"role": "user", "content": "Again."},))
                except RolloutEnded:
                    continue

    with pytest.raises(RolloutInterrupted) as caught:
        # Without a cancellation at each request after the attempt deadline, the loop above does not end.
        async with asyncio.timeout(5):
            await engine(
                ReplayModel([{"role": "assistant", "content": "12"}] * 2), harnesses={"fixture": SwallowsDeadline()}
            ).run_task(rollout_spec(arithmetic_task(), harness="fixture", attempt_timeout=0.05))
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ATTEMPT, TimeoutError)
    assert [len(conversation.steps) for conversation in caught.value.rollout.conversations] == [1]


async def test_a_turn_with_recorded_observations_takes_no_reply():
    class RecordsThenReplies:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            session.record(observations=({"role": "user", "content": "Raw result."},))
            await session.reply(({"role": "user", "content": "Summary."},))

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(
            ReplayModel([{"role": "assistant", "content": "12"}] * 2), harnesses={"fixture": RecordsThenReplies()}
        ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, RuntimeError)


async def test_an_agent_waits_while_the_engine_grades():
    returned = asyncio.Event()

    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        try:
            await complete(Prompt(({"role": "user", "content": "Again."},), {}))
        finally:
            returned.set()

    # The verifier machine is the only machine, so grading blocks while it starts.
    factory = BlockingFactory(blocked=0)
    task = arithmetic_task(grader=ScriptGrader(argv=("sh", "-c", "echo 1"), environment=GRADER_ENVIRONMENT))
    pending = asyncio.create_task(
        engine(
            ReplayModel([{"role": "assistant", "content": "12"}]),
            {"local": factory},
            harnesses={"agent": AgentHarness(agent)},
        ).run_task(rollout_spec(task, verifier_machine=machine_runtime(), harness="agent", max_turns=1))
    )
    try:
        await asyncio.wait_for(factory.entered.wait(), timeout=5)
        assert not returned.is_set()
    finally:
        factory.release.set()
    record = await pending
    assert returned.is_set()
    assert record.grade.reward == 1.0


async def test_a_harness_that_fails_after_a_model_failure_is_a_harness_failure():
    class GivesUp:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await session.reply(({"role": "user", "content": "Again."},))
            except RolloutEnded:
                raise RuntimeError("The harness gave up after the model failed") from None

    class FailingModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise ConnectionError("inference server unavailable")
            return await super().complete(request)

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await engine(FailingModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": GivesUp()}).run_task(
            rollout_spec(arithmetic_task(), harness="fixture")
        )
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, RuntimeError)
    assert caught.value.grade.status == Outcome.UNAVAILABLE


async def test_an_agent_that_fails_while_it_stops_is_a_cleanup_error():
    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))
        try:
            await complete(Prompt(({"role": "user", "content": "Again."},), {}))
        except (RolloutEnded, asyncio.CancelledError):
            raise OSError("The agent's server did not shut down") from None

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", max_turns=1))
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.grade.diagnostics["cleanup_errors"] == [{"operation": "agent_stop", "exception_type": "OSError"}]


async def test_the_end_of_the_harness_stops_a_model_call_and_drops_its_late_answer():
    stopped = asyncio.Event()
    late = asyncio.Event()
    answered = asyncio.Event()
    received = []

    async def agent(_context, complete):
        prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
        turn = await complete(prompt)
        follow_up = Prompt((*prompt.messages, turn.message, {"role": "user", "content": "Again."}), {})
        received.append(await complete(follow_up))

    class LateModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    # The transport ignores the cancellation, and it answers while the engine grades.
                    stopped.set()
                    await late.wait()
                answered.set()
            return await super().complete(request)

    model = LateModel([{"role": "assistant", "content": "12"}] * 2)
    factory = BlockingFactory(blocked=0)
    task = arithmetic_task(grader=ScriptGrader(argv=("sh", "-c", "echo 1"), environment=GRADER_ENVIRONMENT))
    spec = rollout_spec(task, verifier_machine=machine_runtime(), harness="agent", total_turn_timeout=0.05)
    pending = asyncio.create_task(
        engine(model, {"local": factory}, harnesses={"agent": AgentHarness(agent)}).run_task(spec)
    )
    try:
        await asyncio.wait_for(factory.entered.wait(), timeout=5)
        await asyncio.wait_for(stopped.wait(), timeout=5)
        late.set()
        await asyncio.wait_for(answered.wait(), timeout=5)
    finally:
        late.set()
        factory.release.set()
    record = await asyncio.wait_for(pending, timeout=5)

    assert received == []
    assert record.rollout.stop_reason == "total_turn_timeout"
    [conversation] = record.rollout.conversations
    assert (len(conversation.steps), conversation.response_token_ids) == (1, (20,))


async def test_an_agent_that_drains_its_handlers_stops_when_a_model_call_is_open_at_the_end():
    handlers = []

    async def agent(_context, complete):
        prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
        turn = await complete(prompt)
        follow_up = Prompt((*prompt.messages, turn.message, {"role": "user", "content": "Again."}), {})
        handlers.append(asyncio.create_task(complete(follow_up)))
        try:
            await asyncio.Future()
        finally:
            # A server that shuts down waits for its open handlers.
            await asyncio.wait(handlers)

    record = await engine(
        FirstTurnModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", total_turn_timeout=0.05, cleanup_timeout=1))
    assert (record.rollout.stop_reason, record.grade.reward) == ("total_turn_timeout", 1.0)
    assert "cleanup_errors" not in record.grade.diagnostics
    assert isinstance(handlers[0].exception(), RolloutEnded)


async def test_a_token_contract_error_in_a_handler_ends_the_attempt():
    handlers = []

    async def agent(_context, complete):
        handlers.append(asyncio.create_task(complete(Prompt(({"role": "user", "content": "Return 12."},), {}))))
        # The server keeps running after its handler failed.
        await asyncio.Future()

    async def model(_request):
        # The log probabilities do not align with the response tokens.
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), (-0.5, -0.5), "stop")

    with pytest.raises(RolloutContractError):
        await asyncio.wait_for(
            RolloutEngine({}, {"agent": AgentHarness(agent)}).run_task(
                rollout_spec(arithmetic_task(), harness="agent"), model
            ),
            timeout=5,
        )
    assert isinstance(handlers[0].exception(), RolloutEnded)


async def test_final_observations_replace_a_cancelled_reply():
    final = {"role": "user", "content": "Final result."}

    class CancelsItsReply:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            call = asyncio.create_task(session.reply(({"role": "user", "content": "Again."},)))
            await asyncio.sleep(0)
            call.cancel()
            await asyncio.wait({call})
            session.record(observations=(final,))

    record = await engine(
        FirstTurnModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": CancelsItsReply()}
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    [step] = record.rollout.conversations[-1].steps
    assert (step.transition.done, step.transition.observations) == (True, (final,))


async def test_a_harness_cannot_change_the_record_after_its_turn_deadline():
    class RecordsLate:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                with pytest.raises(RolloutEnded):
                    session.record(data={"reward": 1.0})
                raise

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": RecordsLate()}
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture", total_turn_timeout=0.05))
    assert record.rollout.stop_reason == "total_turn_timeout"
    assert record.rollout.conversations[-1].steps[-1].transition.data is None


async def test_an_agent_gets_no_turn_after_the_harness_ended():
    factory = BlockingFactory(blocked=0)
    received = []

    async def agent(_context, complete):
        prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
        received.append(await complete(prompt))
        # The agent sends its last prompt again while the engine grades.
        await factory.entered.wait()
        received.append(await complete(prompt))

    task = arithmetic_task(grader=ScriptGrader(argv=("sh", "-c", "echo 1"), environment=GRADER_ENVIRONMENT))
    spec = rollout_spec(task, verifier_machine=machine_runtime(), harness="agent", total_turn_timeout=0.05)
    runner = engine(
        ReplayModel([{"role": "assistant", "content": "12"}]),
        {"local": factory},
        harnesses={"agent": AgentHarness(agent)},
    )
    pending = asyncio.create_task(runner.run_task(spec))
    await asyncio.wait_for(factory.entered.wait(), timeout=5)
    factory.release.set()
    record = await pending

    assert len(received) == 1
    assert (record.rollout.stop_reason, record.grade.reward) == ("total_turn_timeout", 1.0)


async def test_an_agent_that_repeats_its_last_prompt_in_a_loop_cannot_block_a_deadline():
    async def agent(_context, complete):
        while True:
            await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}]), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", total_turn_timeout=0.05))
    assert (record.rollout.stop_reason, record.grade.reward) == ("total_turn_timeout", 1.0)


async def test_a_cancelled_reply_starts_a_new_tool_deadline():
    model = FirstTurnModel([{"role": "assistant", "content": "12"}])

    async def agent(_context, complete):
        prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
        turn = await complete(prompt)
        call = asyncio.create_task(
            complete(Prompt((*prompt.messages, turn.message, {"role": "user", "content": "Again."}), {}))
        )
        await model.blocked.wait()
        # The client disconnects, and the agent never sends a request again.
        call.cancel()
        await asyncio.Future()

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(model, harnesses={"agent": AgentHarness(agent)}).run_task(
            rollout_spec(arithmetic_task(), harness="agent", tool_turn_timeout=0.05)
        )
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, TimeoutError)


async def test_a_harness_records_step_data_only_between_model_requests():
    class RecordsDuringARequest:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            call = asyncio.create_task(session.reply(({"role": "user", "content": "Again."},)))
            await asyncio.sleep(0)
            try:
                session.record(observations=({"role": "user", "content": "Lost."},))
            finally:
                call.cancel()

    with pytest.raises(RolloutInterrupted) as caught:
        await engine(
            FirstTurnModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": RecordsDuringARequest()}
        ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.ADVANCE, RuntimeError)


@pytest.mark.parametrize("answers", [True, False])
async def test_a_transport_that_catches_the_cancellation_of_a_call_adds_no_turn(answers):
    generating = asyncio.Event()
    prompt = Prompt(({"role": "user", "content": "Return 12."},), {})
    requests = []

    async def agent(_context, complete):
        call = asyncio.create_task(complete(prompt))
        await generating.wait()
        call.cancel()
        await asyncio.wait({call})
        await complete(prompt)

    async def model(request):
        requests.append(request)
        if len(requests) == 1:
            generating.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if not answers:
                    raise GenerationLimitReached((10, 11)) from None
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20 + len(requests),), (-0.5,), "stop")

    record = await RolloutEngine({}, {"agent": AgentHarness(agent)}).run_task(
        rollout_spec(arithmetic_task(), harness="agent"), model
    )
    # The cancelled call adds nothing, so the same prompt again makes a new generation.
    assert len(requests) == 2
    [conversation] = record.rollout.conversations
    assert conversation.response_token_ids == (22,)


@pytest.mark.parametrize("answers", [True, False])
async def test_a_turn_after_the_model_turn_deadline_is_a_model_failure(answers):
    async def model(_request):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if not answers:
                raise GenerationLimitReached((10, 11)) from None
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), (-0.5,), "stop")

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await RolloutEngine({}, shell_harnesses()).run_task(
            rollout_spec(arithmetic_task(), model_turn_timeout=0.05), model
        )
    assert (caught.value.operation, type(caught.value.__cause__)) == (RolloutOperation.MODEL, TimeoutError)
    assert caught.value.rollout.conversations[-1].steps == ()


async def test_a_new_conversation_cut_off_by_the_end_of_the_harness_is_recorded_once():
    class StartsAgainInTheBackground:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            call = asyncio.create_task(session.start(Prompt(({"role": "user", "content": "Start again."},), {})))

            async def close():
                # The cut-off request ends in the cleanup, before the rollout is built.
                with pytest.raises(RolloutEnded):
                    await call

            context.cleanup.push("late_request", close)
            await asyncio.sleep(0)

    class LateModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    pass
            return await super().complete(request)

    model = LateModel([{"role": "assistant", "content": "12"}] * 2)
    rollout = await harness_rollout(
        StartsAgainInTheBackground(), arithmetic_task(), model.complete, limits=rollout_limits()
    )
    first, cut_off = rollout.conversations
    assert (len(first.steps), cut_off.steps) == (1, ())
    assert "cleanup_error_count" not in rollout.metrics


async def test_a_cleanup_action_starts_even_when_its_deadline_is_shorter_than_a_loop_turn():
    started = asyncio.Event()

    async def close():
        started.set()
        await asyncio.Future()

    with pytest.raises(TimeoutError):
        await finish_cleanup(close, timeout=1e-300)
    assert started.is_set()


async def test_a_task_machine_reports_its_close_failure_when_the_body_fails():
    class FailingClose:
        async def run(self, command):
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def close(self):
            raise OSError("close failed")

    class Factory(DockerMachineFactory):
        backend = Backend.DOCKER

        async def create(self, spec):
            return FailingClose()

    with pytest.raises(ValueError, match="body failed") as caught:
        task = arithmetic_task(environment_requirements=GRADER_ENVIRONMENT)
        async with task_machine(task, Factory(), machine_runtime(), cleanup_timeout=1):
            raise ValueError("body failed")
    assert caught.value.__notes__ == ["Machine cleanup failed: ['machine_close']"]


class StartsTwice:
    def check(self, task, limits):
        pass

    async def run(self, context, session):
        await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
        session.record(metrics={"score": 1.0})
        await session.start(Prompt(({"role": "user", "content": "Start again."},), {}))


async def test_a_model_failure_in_a_new_conversation_grades_the_earlier_one():
    class FailingModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise ConnectionError("inference server unavailable")
            return await super().complete(request)

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await engine(
            FailingModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": StartsTwice()}
        ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert caught.value.operation == RolloutOperation.MODEL
    assert (caught.value.grade.status, caught.value.grade.reward) == (Outcome.GRADED, 1.0)
    first, failed = caught.value.rollout.conversations
    assert (len(first.steps), failed.steps) == (1, ())
    assert caught.value.rollout.metrics == {"score": 1.0}


class RecordsAfterARefusal:
    """Records a final reward after the session refuses the second request."""

    def __init__(self, second: str):
        self.second = second

    def check(self, task, limits):
        pass

    async def run(self, context, session):
        await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
        session.record(data="turn", metrics={"score": 0.0})
        try:
            if self.second == "start":
                await session.start(Prompt(({"role": "user", "content": "Start again."},), {}))
            else:
                await session.reply(({"role": "user", "content": "Continue."},))
        except RolloutEnded:
            with pytest.raises(RuntimeError, match="no more observations"):
                session.record(observations=({"role": "user", "content": "Late."},))
            session.record(data="final", metrics={"score": 1.0})
            raise


@pytest.mark.parametrize(
    "second,refusal", [("reply", "limit"), ("reply", "model"), ("start", "model")], ids=["limit", "model", "new"]
)
async def test_a_harness_records_a_final_reward_after_the_session_refuses_a_request(second, refusal):
    class FailingModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise ConnectionError("inference server unavailable")
            return await super().complete(request)

    model = (ReplayModel if refusal == "limit" else FailingModel)([{"role": "assistant", "content": "12"}])
    spec = rollout_spec(arithmetic_task(), harness="fixture", **({"max_turns": 1} if refusal == "limit" else {}))
    runner = engine(model, harnesses={"fixture": RecordsAfterARefusal(second)})
    if refusal == "limit":
        rollout = (await runner.run_task(spec)).rollout
    else:
        with pytest.raises(GradedRolloutInterrupted) as caught:
            await runner.run_task(spec)
        assert caught.value.operation == RolloutOperation.MODEL
        rollout = caught.value.rollout
    # The reward goes to the step of the last turn, also when that turn is in an earlier conversation.
    [step] = rollout.conversations[0].steps
    assert (step.transition.data, step.transition.metrics) == ("final", {"score": 1.0})
    assert rollout.metrics["score"] == 1.0


async def test_recording_after_a_failed_first_request_raises_the_refusal_again():
    class RecordsAfterAFailedFirstRequest:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            try:
                await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            except RolloutEnded:
                # Without a turn there is no step for the reward, and the refusal stays the outcome.
                with pytest.raises(RolloutEnded):
                    session.record(data="final")
                raise

    class FailingModel(ReplayModel):
        async def complete(self, request):
            raise ConnectionError("inference server unavailable")

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await engine(FailingModel([]), harnesses={"fixture": RecordsAfterAFailedFirstRequest()}).run_task(
            rollout_spec(arithmetic_task(), harness="fixture")
        )
    assert caught.value.operation == RolloutOperation.MODEL
    assert isinstance(caught.value.__cause__, ConnectionError)


async def test_a_pinned_prompt_that_does_not_start_with_the_served_tokens_is_not_sent():
    recorded = []

    class PinsOtherTokens:
        def check(self, task, limits):
            pass

        async def run(self, context, session):
            await session.start(Prompt(({"role": "user", "content": "Return 12."},), {}))
            try:
                await session.reply(({"role": "user", "content": "Again."},), token_ids=(1, 2, 3))
            except RolloutEnded:
                # The refused request ended the turn's step, so the step still takes data.
                session.record(data="final")
                recorded.append(True)
                raise

    model = ReplayModel([{"role": "assistant", "content": "12"}, {"role": "assistant", "content": "12"}])
    with pytest.raises(RolloutContractError, match="must start with the tokens served so far"):
        await engine(model, harnesses={"fixture": PinsOtherTokens()}).run_task(
            rollout_spec(arithmetic_task(), harness="fixture")
        )
    assert (len(model.requests), recorded) == (1, [True])


async def test_a_final_call_keeps_the_observations_of_the_calls_before_it():
    task = file_task(final_tools=(FunctionDefinition(name="finish", parameters={"type": "object"}),))
    write = shell_call("echo 12 > /workspace/answer")["tool_calls"][0]
    finish = {"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}
    model = ReplayModel([{"role": "assistant", "content": "", "tool_calls": [write, finish]}])
    record = await engine(model, {"local": RecordingShellSimFactory()}).run_task(
        rollout_spec(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    )
    [step] = record.rollout.conversations[-1].steps
    [observation] = step.transition.observations
    assert (step.transition.done, observation["tool_call_id"]) == (True, "write")
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)


async def test_a_grading_failure_after_a_model_failure_keeps_the_model_failure():
    class FailingModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise ConnectionError("inference server unavailable")
            return await super().complete(request)

    command = f"mkdir -p /workspace/out && cd /workspace/out && touch $(seq {MAX_OUTPUT_FILES + 1})"
    spec = rollout_spec(
        report_task(output_paths=("/workspace/out",)), machine=machine_runtime(), verifier_machine=machine_runtime()
    )
    with pytest.raises(GradedRolloutInterrupted) as caught:
        await engine(FailingModel([shell_call(command)]), {"local": RecordingShellSimFactory()}).run_task(spec)
    assert caught.value.operation == RolloutOperation.MODEL
    assert isinstance(caught.value.__cause__, ConnectionError)
    [note] = caught.value.__cause__.__notes__
    assert "file or byte budget" in note
    assert caught.value.rollout.failure == RolloutFailure(
        "ConnectionError", {"operation": "model", "error": "inference server unavailable", "notes": [note]}
    )


async def test_a_generation_limit_in_a_new_conversation_grades_the_earlier_one():
    class LimitedModel(ReplayModel):
        async def complete(self, request):
            if self.requests:
                raise GenerationLimitReached((70, 71))
            return await super().complete(request)

    record = await engine(
        LimitedModel([{"role": "assistant", "content": "12"}]), harnesses={"fixture": StartsTwice()}
    ).run_task(rollout_spec(arithmetic_task(), harness="fixture"))
    assert (record.rollout.stop_reason, record.grade.status, record.grade.reward) == ("length", Outcome.GRADED, 1.0)
    first, unserved = record.rollout.conversations
    assert (len(first.steps), unserved.steps, unserved.prompt_token_ids) == (1, (), (70, 71))
    assert record.rollout.metrics == {"score": 1.0}


async def test_an_unlimited_agent_can_finish_beyond_the_old_turn_limit():
    async def agent(_context, complete):
        messages = ({"role": "user", "content": "Return 12."},)
        for _ in range(6):
            response = await complete(Prompt(messages, {}))
            messages = (*messages, response.message, {"role": "user", "content": "Continue."})

    record = await engine(
        ReplayModel([{"role": "assistant", "content": "12"}] * 6), harnesses={"agent": AgentHarness(agent)}
    ).run_task(rollout_spec(arithmetic_task(), harness="agent", max_turns=None))
    assert len(record.rollout.conversations[-1].steps) == 6
    assert record.rollout.stop_reason != "max_turns"
    assert record.grade.reward == 1.0
