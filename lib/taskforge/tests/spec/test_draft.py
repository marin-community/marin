# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from rolloutengine.machines import _machine_spec
from rolloutengine.models import ModelRequest, ModelTurn, TaskRolloutSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    ExecutionRequirements,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
)
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerFormat,
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    FinalAction,
    FunctionDefinition,
    JsonValueAnswer,
    NoGrader,
    PlainText,
    ResourceGroups,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    VerifierArtifact,
    VerifierCommand,
    VerifyitGrader,
)
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import (
    ExactSpec,
    FunctionCall,
    MathSpec,
    McqSpec,
    NumericSpec,
    PredictedActionSpec,
    PytestSpec,
    StructuredExactSpec,
)

from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec.draft import (
    ANSWER_PATH,
    FILESYSTEM_CAPABILITY,
    GRADER_VERDICT_PATH,
    LOCAL_COPY_ON_WRITE,
    LOCAL_CPYTHON,
    LOCAL_READ_ONLY,
    LOCAL_UNWRITABLE_ROOTS,
    PYTHON_GRADER_ARGV,
    SHELL_CAPABILITY,
    MachineSettings,
    Resources,
    answer_grader,
    assemble,
    file,
    grader_environment,
    lower,
    machine,
    machine_backend,
    python_grader,
    requirements,
    reward_file,
    rollout_engine,
    rollout_limits,
    script_grader,
)
from taskforge.spec.verdict import GraderCheck, grader_checks, verdict_json
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
PLAIN = PlainText()
FACTORIES = {"shellsim": PortlessShellSimFactory(), "docker": FixtureImageFactory(Backend.DOCKER)}
COMMAND_TIMEOUT = 10
LIMITS = rollout_limits(
    max_turns=4,
    model_turn_timeout=None,
    tool_turn_timeout=20,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=30,
    cleanup_timeout=10,
)
SHELLSIM = machine(startup_timeout=30)
VERIFIER = machine(startup_timeout=30)
IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
BASE = grader_environment(None)
# sh programs, run as /tests/grade.py by ``shell_backed`` packages: a bare ShellSim machine has no python3.
SUM_GRADER = """expected=$(jq -r .expected /tests/config.json)
passed=false
if test -f /workspace/answer; then
    got=$(cat /workspace/answer)
    echo "got $got"
    if test "$got" = "$expected"; then passed=true; fi
fi
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "answer", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""
TEXT_GRADER = """passed=false
if test "$(cat /app/answer.txt)" = twelve; then passed=true; fi
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "answer", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""
KEY_GRADER = """passed=false
if test "$(cat /app/answer.txt)" = "$(cat /tests/key/answer.txt)"; then passed=true; fi
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "answer", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        continuation = request.continuation
        prompt = (10, 11) if continuation is None else (*continuation.served_token_ids, 90, 91)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


@dataclass(frozen=True)
class RecordingFactory:
    """A container factory that is registered for lowering but never asked for a machine."""

    backend: Backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """Accept every environment: these tests check lowering, not a backend's support."""

    async def create(self, spec: MachineSpec):
        raise AssertionError(f"Unexpected machine for {spec}")


@dataclass(frozen=True)
class LocalRecordingFactory(RecordingFactory):
    """A local factory that is registered for lowering and states its mode for system directories."""

    system_directories: str


def shell_message(call_id: str, command: str) -> dict:
    arguments = json.dumps({"command": command})
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "shell", "arguments": arguments}}],
    }


async def run(lowered: TaskRolloutSpec, messages: list[dict]):
    engine = rollout_engine(FACTORIES, command_timeout=COMMAND_TIMEOUT)
    restored = TaskRolloutSpec.model_validate_json(lowered.model_dump_json())
    return await engine.run_task(restored, ReplayModel(messages).complete)


def lower_here(
    task: TaskSpec,
    task_machine: MachineSettings | None = None,
    verifier_machine: MachineSettings | None = None,
    host: MachineHost = MachineHost.LAPTOP,
) -> TaskRolloutSpec:
    return lower(
        task,
        host=host,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=LIMITS,
        factories=FACTORIES,
    )


def sum_grader() -> GraderPackage:
    return python_grader(SUM_GRADER, {"expected": "12"}, environment=BASE, answer_path=None, timeout=20)


def shell_backed(package: GraderPackage) -> GraderPackage:
    """``package`` with ``sh`` in place of ``python3``, so its ``/tests/grade.py`` runs as a sh program.

    A bare ShellSim machine has no python3 (shellsim 0.1.36), and Marin CI leaves interpreter coverage to
    ShellSim. A test of a shell-backed package checks what ``python_grader`` packages and how its verdict is
    read: the program and private config under ``/tests``, its further private files, the captured output and
    the extracted answer it reads, and the ``GRADER_VERDICT_PATH`` reward. The program's language is not tested.
    """
    assert isinstance(package.grader, ScriptGrader) and package.grader.argv == PYTHON_GRADER_ARGV
    return replace(package, grader=package.grader.model_copy(update={"argv": ("sh", *PYTHON_GRADER_ARGV[1:])}))


def file_task(grader: GraderPackage | None = None) -> TaskSpec:
    return assemble(
        "sum-file",
        "Write the sum of the numbers in /workspace/numbers.txt to /workspace/answer.",
        AnswerType.FILE,
        PLAIN,
        grader or sum_grader(),
        SOURCE,
        environment=requirements(image=None),
        files=(file("workspace/numbers.txt", "5\n7\n"),),
        output_paths=("/workspace/answer",),
    )


def answer_task(
    grader: GraderPackage, answer_type: AnswerType = AnswerType.TEXT, answer_format: AnswerFormat = PLAIN, **fields
) -> TaskSpec:
    return assemble(
        "answer", "What is six plus six?", answer_type, answer_format, grader, SOURCE, environment=None, **fields
    )


def docker_task(grader: GraderPackage) -> TaskSpec:
    return assemble(
        "docker",
        "Fix the bug in /app.",
        AnswerType.WORKSPACE_STATE,
        PLAIN,
        grader,
        SOURCE,
        environment=requirements(image=IMAGE, setup=("pip install pytest",), workdir="/app", env={"MODE": "test"}),
        files=(file("app/run.sh", "#!/bin/sh\npython3 main.py\n", mode=0o755),),
        output_paths=("/app/main.py",),
        tags=("taskforge",),
    )


def docker_grader(**fields) -> GraderPackage:
    return script_grader(
        ("python3", "/tests/grade.py"),
        reward_file("/logs/verifier/reward.json", RewardFileFormat.JSON, key="score", pass_above=0.5),
        environment=grader_environment(IMAGE, env=fields.pop("env", None)),
        answer_path=None,
        timeout=60,
        files=(file("grade.py", "print(1)\n"),),
        **fields,
    )


def test_file_lands_relative_to_the_machine_root_with_its_mode():
    resource = file("app/run.sh", "#!/bin/sh\n", mode=0o755)

    assert (resource.path, resource.mode, resource_bytes(resource)) == ("app/run.sh", "755", b"#!/bin/sh\n")
    with pytest.raises(ValueError):
        file("/workspace/answer", "12")


@pytest.mark.parametrize("answer,expected", [("12", 1.0), ("13", 0.0)])
async def test_python_grader_grades_the_captured_output_of_a_shellsim_run_on_a_verifier_machine(answer, expected):
    lowered = lower_here(file_task(shell_backed(sum_grader())), SHELLSIM, VERIFIER)

    result = await run(
        lowered, [shell_message("c1", f"echo {answer} > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert "config.json" not in json.dumps([chat.messages for chat in result.rollout.conversations])


@pytest.mark.parametrize("answer,expected", [("twelve", 1.0), ("eleven", 0.0)])
async def test_python_grader_reads_the_extracted_answer_of_a_task_without_a_machine(answer, expected):
    grader = shell_backed(python_grader(TEXT_GRADER, {}, environment=BASE, answer_path=ANSWER_PATH, timeout=20))
    lowered = lower_here(answer_task(grader), verifier_machine=VERIFIER)

    result = await run(lowered, [{"role": "assistant", "content": answer}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert lowered.verifier_machine is not None and lowered.verifier_machine.backend == "docker"


@pytest.mark.parametrize("answer,expected", [("twelve", 1.0), ("eleven", 0.0)])
async def test_python_grader_reads_its_private_files_from_the_tests_directory(answer, expected):
    grader = shell_backed(
        python_grader(
            KEY_GRADER,
            {},
            environment=BASE,
            answer_path=ANSWER_PATH,
            timeout=20,
            files=(file("key/answer.txt", "twelve\n"),),
        )
    )
    lowered = lower_here(answer_task(grader), verifier_machine=VERIFIER)

    result = await run(lowered, [{"role": "assistant", "content": answer}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)


async def test_python_grader_verdict_carries_its_checks_into_the_grade():
    checks = (GraderCheck("total", True, 1.0), GraderCheck("units", False, 1.0))
    script = f"printf '%s' {shlex.quote(verdict_json(0.5, checks))} > {GRADER_VERDICT_PATH}\n"
    grader = shell_backed(python_grader(script, {}, environment=BASE, answer_path=ANSWER_PATH, timeout=20))
    lowered = lower_here(answer_task(grader), verifier_machine=VERIFIER)

    result = await run(lowered, [{"role": "assistant", "content": "twelve"}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 0.5)
    assert grader_checks(result.grade) == checks


def test_python_grader_refuses_a_file_that_shadows_its_program_or_config():
    with pytest.raises(ValueError, match="shadow"):
        python_grader(
            TEXT_GRADER, {}, environment=BASE, answer_path=ANSWER_PATH, timeout=20, files=(file("config.json", "{}"),)
        )


def test_grader_environment_is_the_task_image_or_the_grader_base():
    assert grader_environment(None) == EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=GRADER_BASE_IMAGE
    )
    assert grader_environment(IMAGE, env={"A": "1"}) == EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE, environment_variables={"A": "1"}
    )


@pytest.mark.parametrize(
    "spec,right,wrong",
    [
        (ExactSpec(expected=("twelve",)), "Twelve", "eleven"),
        (NumericSpec(expected="12", tolerance_abs=0.0, tolerance_rel=0.0), "12", "13"),
        (McqSpec(expected="B"), "B", "C"),
        (MathSpec(expected="12"), "12", "13"),
    ],
)
async def test_answer_grader_grades_the_final_reply_in_process(spec, right, wrong):
    lowered = lower_here(answer_task(answer_grader(spec), system="Answer briefly."))

    rewards = [(await run(lowered, [{"role": "assistant", "content": text}])).grade.reward for text in (right, wrong)]

    assert rewards == [1.0, 0.0]
    assert (lowered.task_machine, lowered.verifier_machine) == (None, None)
    assert lowered.task.context.events[0].content == "Answer briefly."
    assert lowered.task.answer_format == PLAIN


async def test_predicted_action_grader_grades_a_native_action():
    lookup = FunctionDefinition(name="lookup", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
    spec = PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"city": "Paris"}),))
    task = answer_task(answer_grader(spec), AnswerType.NATIVE_ACTION, FinalAction(), final_tools=(lookup,))
    lowered = lower_here(task)

    def call(city: str) -> dict:
        arguments = json.dumps({"city": city})
        return {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": arguments}}],
        }

    rewards = [(await run(lowered, [call(city)])).grade.reward for city in ("Paris", "Rome")]

    assert rewards == [1.0, 0.0]


async def test_structured_exact_grader_grades_a_json_answer():
    grader = answer_grader(StructuredExactSpec(expected={"total": 12}))
    lowered = lower_here(answer_task(grader, AnswerType.JSON, JsonValueAnswer()))

    rewards = [
        (await run(lowered, [{"role": "assistant", "content": text}])).grade.reward
        for text in ('{"total": 12}', '{"total": 13}')
    ]

    assert rewards == [1.0, 0.0]


def test_answer_grader_needs_an_environment_for_a_mode_that_does_not_grade_in_process():
    with pytest.raises(ValueError, match="does not grade in process"):
        answer_grader(PytestSpec(paths=("test_answer.py",)))

    package = answer_grader(PytestSpec(paths=("test_answer.py",)), environment=grader_environment(IMAGE))

    assert isinstance(package.grader, VerifyitGrader) and package.grader.environment is not None


def test_answer_grader_refuses_the_grader_base_image_which_has_no_verifyit():
    with pytest.raises(ValueError, match="no verifyit"):
        answer_grader(PytestSpec(paths=("test_answer.py",)), environment=grader_environment(None))


@pytest.mark.parametrize("host,backend", [(MachineHost.LAPTOP, Backend.DOCKER), (MachineHost.IRIS, Backend.GVISOR)])
def test_docker_task_lowers_both_machines_onto_the_hosts_container_backend(host, backend):
    grader = docker_grader(
        collect=(VerifierCommand(argv=("cp", "-r", "/app", "/tmp/app")),),
        artifacts=(VerifierArtifact(source="/app", target="/submission", kind=ArtifactKind.DIRECTORY),),
        env={"STRICT": "1"},
    )
    task = docker_task(grader)
    task_machine = machine(
        startup_timeout=300,
        network=True,
        resources=Resources(memory_mb=4096, cpus=4, storage_mb=10240),
        user="root",
        cleanup_timeout=20,
    )
    verifier_machine = machine(startup_timeout=120, user="0")

    lowered = lower(
        task,
        host=host,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=LIMITS,
        factories={backend.value: RecordingFactory(backend)},
    )

    assert lowered.task_machine is not None and lowered.verifier_machine is not None
    assert (lowered.task_machine.backend, lowered.verifier_machine.backend) == (backend.value, backend.value)
    assert (lowered.task_machine.network, lowered.task_machine.cpus, lowered.task_machine.memory_mb) == (
        NetworkPolicy.ALLOW,
        4,
        4096,
    )
    assert (lowered.task_machine.storage_mb, lowered.task_machine.user, lowered.task_machine.cleanup_timeout) == (
        10240,
        "root",
        20,
    )
    assert (lowered.verifier_machine.network, lowered.verifier_machine.user) == (NetworkPolicy.DENY, "0")
    assert TaskRolloutSpec.model_validate_json(lowered.model_dump_json()) == lowered


def test_script_grader_keeps_its_files_private_and_its_image_on_the_verifier():
    task = docker_task(docker_grader(env={"STRICT": "1"}))

    assert isinstance(task.grader, ScriptGrader)
    assert (task.grader.argv, task.grader.answer_path, task.grader.timeout) == (("python3", "/tests/grade.py"), None, 60)
    assert task.grader.environment == EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE, environment_variables={"STRICT": "1"}
    )
    assert [item.path for item in task.resources.verifier] == ["grade.py"]
    assert [item.path for item in task.resources.worker] == ["app/run.sh"]
    assert task.resources.all == ()
    assert task.environment_requirements == EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=IMAGE,
        working_directory="/app",
        setup_commands=("pip install pytest",),
        environment_variables={"MODE": "test"},
    )


def test_python_grader_writes_its_program_and_config_under_tests():
    grader = python_grader(SUM_GRADER, {"expected": "12"}, environment=BASE, answer_path=None, timeout=20)

    assert isinstance(grader.grader, ScriptGrader)
    assert grader.grader.argv == ("python3", "/tests/grade.py")
    assert isinstance(grader.grader.reward, FileReward)
    assert [(f.path, f.format, f.key) for f in grader.grader.reward.files] == [
        (GRADER_VERDICT_PATH, RewardFileFormat.JSON, "reward")
    ]
    assert {item.path: json.loads(resource_bytes(item)) for item in grader.resources if item.path == "config.json"} == {
        "config.json": {"expected": "12"}
    }


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"files": (file("workspace/a", "1"),)}, "require a task machine"),
        ({"output_paths": ("/workspace/answer",)}, "require a task machine"),
        ({"environment": EnvironmentRequirements(), "files": (file("workspace/a", "1"),)}, "require a task machine"),
        ({"environment": EnvironmentRequirements(capabilities=(FILESYSTEM_CAPABILITY,))}, "'shell' capability"),
    ],
)
def test_assemble_rejects_machine_inputs_without_a_shell_task_machine(fields: dict[str, Any], message):
    """``NO_MACHINE`` (an empty requirements) means no task machine, exactly as ``None`` does."""
    grader = answer_grader(ExactSpec(expected=("12",)))
    with pytest.raises(ValueError, match=message):
        arguments: dict[str, Any] = {"environment": None, **fields}
        assemble("bad", "Do it.", AnswerType.TEXT, PLAIN, grader, SOURCE, **arguments)


EXACT = answer_grader(ExactSpec(expected=("12",)))
SHELL = script_grader(("true",), ExitCodeReward(), environment=BASE, answer_path=None, timeout=10)
TEXT_SHELL = script_grader(("true",), ExitCodeReward(), environment=BASE, answer_path=ANSWER_PATH, timeout=10)


@pytest.mark.parametrize(
    "answer_type,answer_format,grader,output_paths,message",
    [
        (AnswerType.FILE, PLAIN, SHELL, (), "needs output paths or grader artifacts"),
        (AnswerType.WORKSPACE_STATE, PLAIN, SHELL, (), "needs output paths or grader artifacts"),
        (AnswerType.FILE, PLAIN, EXACT, ("/workspace/answer",), "requires a grading environment"),
        (AnswerType.FILE, PLAIN, TEXT_SHELL, ("/workspace/answer",), "no extracted answer to write"),
        (AnswerType.JSON, PLAIN, EXACT, (), "cannot carry a json answer"),
        (AnswerType.NATIVE_ACTION, FinalAction(), EXACT, (), "not accepted by the selected verifier"),
        (
            AnswerType.TEXT,
            PLAIN,
            answer_grader(StructuredExactSpec(expected={"t": 1})),
            (),
            "not accepted by the selected verifier",
        ),
    ],
)
def test_assemble_rejects_graders_that_cannot_see_the_answer(answer_type, answer_format, grader, output_paths, message):
    final_tools = (FunctionDefinition(name="f", parameters={}),) if answer_type == AnswerType.NATIVE_ACTION else ()
    with pytest.raises(ValueError, match=message):
        assemble(
            "bad",
            "Do it.",
            answer_type,
            answer_format,
            grader,
            SOURCE,
            environment=requirements(image=None),
            output_paths=output_paths,
            final_tools=final_tools,
        )


def test_assemble_keeps_a_task_without_a_grader():
    task = answer_task(GraderPackage(NoGrader(reason="the source has no checkable answer")))

    assert task.grader == NoGrader(reason="the source has no checkable answer")
    assert lower_here(task).verifier_machine is None


@pytest.mark.parametrize(
    "path,message",
    [
        ("workspace/answer", "absolute"),
        ("/workspace/../tests/x", "normalized"),
        ("/tests/answer", "private mounts"),
        ("/logs/verifier/reward.json", "private mounts"),
    ],
)
def test_assemble_rejects_output_paths_outside_the_agents_reach(path, message):
    with pytest.raises(ValueError, match=message):
        assemble(
            "bad",
            "Do it.",
            AnswerType.FILE,
            PLAIN,
            sum_grader(),
            SOURCE,
            environment=requirements(image=None),
            output_paths=(path,),
        )


def test_assemble_rejects_private_grader_content_shipped_to_the_agent():
    copy = file("workspace/notes.py", SUM_GRADER)
    with pytest.raises(ValueError, match=r"agent-visible: \['grade.py'\]"):
        assemble(
            "leak",
            "Write the answer.",
            AnswerType.FILE,
            PLAIN,
            sum_grader(),
            SOURCE,
            environment=requirements(image=None),
            files=(copy,),
            output_paths=("/workspace/answer",),
        )


@pytest.mark.parametrize(
    "task_machine,verifier_machine,message",
    [
        (None, VERIFIER, "task machine is given exactly"),
        (SHELLSIM, None, "verifier machine is given exactly"),
    ],
)
def test_lower_takes_exactly_the_machines_the_task_needs(task_machine, verifier_machine, message):
    with pytest.raises(ValueError, match=message):
        lower_here(file_task(), task_machine, verifier_machine)
    with pytest.raises(ValueError, match="verifier machine is given exactly"):
        lower_here(answer_task(EXACT), verifier_machine=VERIFIER)


def test_lower_propagates_rolloutengines_rejections():
    with pytest.raises(ValueError, match="Unknown Shellbox factory: 'docker'"):
        lower(
            file_task(),
            host=MachineHost.LAPTOP,
            task_machine=SHELLSIM,
            verifier_machine=VERIFIER,
            limits=LIMITS,
            factories={"shellsim": ShellSimMachineFactory()},
        )
    # The command timeout belongs to the engine's shell harness, so the engine that runs a spec rejects it.
    lowered = lower_here(file_task(), SHELLSIM, VERIFIER)
    with pytest.raises(ValueError, match="command timeout must be less"):
        rollout_engine(FACTORIES, command_timeout=LIMITS.tool_turn_timeout).check(lowered)


def test_lower_puts_a_shellsim_task_on_shellsim_and_its_grader_on_the_hosts_container_backend():
    lowered = lower(
        file_task(),
        host=MachineHost.IRIS,
        task_machine=SHELLSIM,
        verifier_machine=VERIFIER,
        limits=LIMITS,
        factories={"shellsim": ShellSimMachineFactory(), "gvisor": RecordingFactory(Backend.GVISOR)},
    )

    assert lowered.task_machine is not None and lowered.verifier_machine is not None
    assert (lowered.task_machine.backend, lowered.verifier_machine.backend) == ("shellsim", "gvisor")


def test_a_shellsim_task_machine_asks_for_the_catalogs_cpython():
    lowered = lower(
        file_task(),
        host=MachineHost.LAPTOP,
        task_machine=SHELLSIM,
        verifier_machine=VERIFIER,
        limits=LIMITS,
        factories=FACTORIES,
    )
    environment = lowered.task.environment_requirements
    assert lowered.task_machine is not None

    # ShellSim installs the catalog's CPython for any set ``pypi_packages``, here none beyond it.
    assert _machine_spec(environment, lowered.task_machine).source == PackageSpec(pypi_packages=())
    assert requirements(image=IMAGE).package_spec is None


def test_a_lock_only_grader_lowers_onto_the_local_backend():
    locked = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="gs://bucket/env/uv.lock"
    )
    grader = python_grader(TEXT_GRADER, {}, environment=locked, answer_path=ANSWER_PATH, timeout=20)

    assert machine_backend(locked, MachineHost.IRIS) == Backend.LOCAL
    # RolloutEngine accepts any factory whose backend is local, so a wrapped local factory serves the grader.
    lowered = lower(
        answer_task(grader),
        host=MachineHost.IRIS,
        task_machine=None,
        verifier_machine=VERIFIER,
        limits=LIMITS,
        factories={"local": RecordingFactory(Backend.LOCAL)},
    )

    assert lowered.verifier_machine is not None and lowered.verifier_machine.backend == "local"


def cpython_task(grader: GraderPackage | None = None) -> TaskSpec:
    return assemble(
        "sum-cpython",
        "Write the sum of the numbers in /workspace/numbers.txt to /workspace/answer.",
        AnswerType.FILE,
        PLAIN,
        grader or sum_grader(),
        SOURCE,
        environment=requirements(image=GRADER_BASE_IMAGE),
        files=(file("workspace/numbers.txt", "5\n7\n"),),
        output_paths=("/workspace/answer",),
    )


def lower_on_iris(task: TaskSpec, backends: tuple[Backend, ...]) -> TaskRolloutSpec:
    return lower(
        task,
        host=MachineHost.IRIS,
        task_machine=SHELLSIM,
        verifier_machine=VERIFIER,
        limits=LIMITS,
        factories={backend.value: RecordingFactory(backend) for backend in backends},
    )


def machine_sources(lowered: TaskRolloutSpec) -> tuple[tuple[str, object], tuple[str, object]]:
    """Each machine's backend and the ``MachineSpec`` source RolloutEngine creates it from."""
    task = lowered.task
    assert lowered.task_machine is not None and lowered.verifier_machine is not None
    assert isinstance(task.grader, ScriptGrader)
    return (
        (lowered.task_machine.backend, _machine_spec(task.environment_requirements, lowered.task_machine).source),
        (lowered.verifier_machine.backend, _machine_spec(task.grader.environment, lowered.verifier_machine).source),
    )


def test_grader_base_machines_run_a_local_cpython_when_the_host_has_the_local_backend():
    lowered = lower_on_iris(cpython_task(), (Backend.GVISOR, Backend.LOCAL))

    assert machine_sources(lowered) == (("local", LOCAL_CPYTHON), ("local", LOCAL_CPYTHON))
    assert TaskRolloutSpec.model_validate_json(lowered.model_dump_json()) == lowered


def test_grader_base_machines_stay_on_the_container_backend_without_the_local_backend():
    lowered = lower_on_iris(cpython_task(), (Backend.GVISOR,))

    image = RegistryImage(GRADER_BASE_IMAGE)
    assert machine_sources(lowered) == (("gvisor", image), ("gvisor", image))
    assert lowered.task == cpython_task()


def test_a_task_image_stays_on_the_container_backend_beside_the_local_backend():
    lowered = lower_on_iris(docker_task(docker_grader()), (Backend.GVISOR, Backend.LOCAL))

    assert machine_sources(lowered) == (("gvisor", RegistryImage(IMAGE)), ("gvisor", RegistryImage(IMAGE)))


def package_task() -> TaskSpec:
    return cpython_task().model_copy(
        update={"environment_requirements": requirements(image=None, packages=("numpy", "pandas"), setup=("true",))}
    )


def test_a_package_machine_installs_its_packages_on_the_local_backend():
    lowered = lower_on_iris(package_task(), (Backend.GVISOR, Backend.LOCAL))

    packages = PackageSpec(pypi_packages=("numpy", "pandas"))
    assert machine_sources(lowered) == (("local", packages), ("local", LOCAL_CPYTHON))
    assert lowered.task.environment_requirements.setup_commands == ("true",)


def test_a_package_machine_is_refused_on_a_host_without_the_local_backend():
    with pytest.raises(ValueError, match="Unknown Shellbox factory: 'local'"):
        lower_on_iris(package_task(), (Backend.GVISOR,))


def test_a_package_machine_takes_no_image_and_one_requirement_per_package():
    with pytest.raises(ValueError, match="not both"):
        requirements(image=IMAGE, packages=("numpy",))
    with pytest.raises(ValueError, match="without pip options"):
        requirements(image=None, packages=("numpy pandas\nscipy",))


def opt_task() -> TaskSpec:
    """``cpython_task`` shipping a file under /opt, a system directory of the local backend."""
    return cpython_task().model_copy(
        update={"resources": ResourceGroups(worker=(file("opt/audit/README.md", "audit"),))}
    )


def lower_beside_local(task: TaskSpec, local: RecordingFactory) -> TaskRolloutSpec:
    return lower(
        task,
        host=MachineHost.IRIS,
        task_machine=SHELLSIM,
        verifier_machine=VERIFIER,
        limits=LIMITS,
        factories={Backend.GVISOR.value: RecordingFactory(Backend.GVISOR), Backend.LOCAL.value: local},
    )


def test_a_copy_on_write_local_factory_keeps_a_task_writing_under_opt_on_the_local_backend():
    lowered = lower_beside_local(opt_task(), LocalRecordingFactory(Backend.LOCAL, LOCAL_COPY_ON_WRITE))

    assert machine_sources(lowered) == (("local", LOCAL_CPYTHON), ("local", LOCAL_CPYTHON))


@pytest.mark.parametrize(
    "local",
    [LocalRecordingFactory(Backend.LOCAL, LOCAL_READ_ONLY), RecordingFactory(Backend.LOCAL)],
    ids=["read-only", "mode-unstated"],
)
def test_a_read_only_local_factory_moves_a_task_writing_under_opt_to_the_container_backend(local: RecordingFactory):
    """Files the task ships under /opt cannot land on a read-only local backend, so the task machine stays a
    container."""
    lowered = lower_beside_local(opt_task(), local)
    plain = lower_beside_local(cpython_task(), local)

    assert machine_sources(lowered) == (("gvisor", RegistryImage(GRADER_BASE_IMAGE)), ("local", LOCAL_CPYTHON))
    assert machine_sources(plain)[0] == ("local", LOCAL_CPYTHON)


def test_the_local_modes_and_unwritable_roots_match_shellbox():
    local = pytest.importorskip("shellbox.backends.local.machine", reason="the local backend imports on Linux only")
    modes = local.SystemDirectories
    overlaid = tuple(root for root in local.SYSTEM_DIRECTORIES if root not in local.OVERLAID_DIRECTORIES)

    assert (LOCAL_READ_ONLY, LOCAL_COPY_ON_WRITE) == (modes.READ_ONLY, modes.COPY_ON_WRITE)
    assert set(LOCAL_UNWRITABLE_ROOTS) == set(modes)
    assert LOCAL_UNWRITABLE_ROOTS[modes.READ_ONLY] == local.SYSTEM_DIRECTORIES
    assert LOCAL_UNWRITABLE_ROOTS[modes.COPY_ON_WRITE] == overlaid
