# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage execution, grading, deadlines, and exact token evidence."""

import asyncio
import json
import tarfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    ExitReason,
    Machine,
    NetworkPolicy,
    Result,
    ShellSimBuiltins,
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
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    SessionGrader,
    ShellToolBinding,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellToolConfig
from verifyit.grade import grade as verifyit_grade
from verifyit.spec import FunctionCall, NumericSpec, PredictedActionSpec, StructuredExactSpec, parse_spec

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
from rolloutengine.task_session import WORKSPACE_INSTRUCTION

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE
)
TWELVE = NumericSpec("12", tolerance_abs=0, tolerance_rel=0)


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

    class Factory:
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

    backend = Backend.DOCKER

    async def create(self, spec):
        return await ShellSimMachineFactory().create(
            replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
        )


@dataclass
class RecordingShellSimFactory:
    backend: Backend = Backend.DOCKER

    machines: list[Machine] = field(default_factory=list)

    async def create(self, spec):
        machine = await FixtureImageFactory().create(spec)
        self.machines.append(machine)
        return machine


def arithmetic_task(**update) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "id": "arithmetic",
            "context": ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
            "environment_requirements": EnvironmentRequirements(),
            "answer_type": AnswerType.NUMBER,
            "answer_format": PlainText(),
            "grader": verifyit_package(TWELVE).grader,
            "source": Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            **update,
        }
    )


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


def engine(model, factories=None, *, sessions=None) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(model.complete, {} if factories is None else factories, sessions=sessions)


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
                reloaded.task, reloaded.runtime, reloaded.session, factories=runner.factories, sessions=runner.sessions
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

    class Factory:
        backend = Backend.DOCKER

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

    class Factory:
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
    script = (
        b"import json\n"
        b"messages = json.load(open('/tests/conversation.json'))\n"
        b"answer = open('/app/answer.txt').read()\n"
        b"question = messages[0]['content'] == 'What is six plus six?'\n"
        b"print(float(question and messages[-1]['content'] == answer == '12'))\n"
    )
    task = arithmetic_task(
        grader=ScriptGrader(argv=("python3", "/tests/grade.py"), environment=GRADER_ENVIRONMENT),
        resources=ResourceGroups(verifier=(inline_resource("grade.py", script),)),
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
            shell_call("test ! -f /tests/expected && cp input answer && echo tainted > baseline"),
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
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
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


async def test_reasoning_only_length_stop_reports_missing_answer():
    message = {"role": "assistant", "content": None, "reasoning": "Unfinished reasoning"}

    class TruncatedModel(ReplayModel):
        async def complete(self, request):
            return replace(await super().complete(request), stop_reason="length")

    record = await engine(TruncatedModel([message])).run(lowered(arithmetic_task()))

    assert record.stop_reason == "length"
    assert record.messages[-1] == message
    assert (record.grade.status, record.grade.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert record.response_token_ids == (20,)


async def test_reasoning_only_length_stop_grades_prior_file_submission():
    message = {"role": "assistant", "content": None, "reasoning": "Unfinished reasoning"}

    class TruncatedModel(ReplayModel):
        async def complete(self, request):
            turn = await super().complete(request)
            return replace(turn, stop_reason="length") if len(self.requests) == 2 else turn

    record = await engine(
        TruncatedModel([shell_call("echo 12 > /workspace/answer"), message]),
        {"local": FixtureImageFactory()},
    ).run(lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime()))

    assert record.stop_reason == "length"
    assert record.messages[-1] == message
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, 1.0)
    assert record.response_token_ids == (20, 90, 91, 21)


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

    runner = ShellboxRolloutEngine(complete, {"local": FixtureImageFactory()})
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

    class Factory:
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

    class Factory:
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

    class Factory:
        backend = Backend.DOCKER

        async def create(self, spec):
            return Machine()

    class Session:
        def __init__(self, lowered, machine):
            self.machine = machine

        async def prepare(self):
            return SessionStart(({"role": "user", "content": "Run the task."},), {})

        async def advance(self, turn):
            result = await self.machine.run(Command(("whoami",)))
            assert (result.exit_code, result.stdout) == (0, b"learner")
            return Transition(done=True)

        async def grade(self, messages):
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

    class Factory:
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

    class Factory:
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

    class Factory:
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
