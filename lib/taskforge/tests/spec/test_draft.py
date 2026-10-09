# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn, SuppliedState
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineSpec, NetworkPolicy
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerFormat,
    AnswerType,
    ArtifactKind,
    EnvironmentRequirements,
    ExitCodeReward,
    FinalAction,
    FunctionDefinition,
    JsonValueAnswer,
    NoGrader,
    PlainText,
    RewardFileFormat,
    ScriptGrader,
    Source,
    StdoutReward,
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
    script_grader,
    session,
)
from tests.sandbox.fixture_images import FixtureImageFactory

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
PLAIN = PlainText()
FACTORIES = {"shellsim": ShellSimMachineFactory(), "docker": FixtureImageFactory(Backend.DOCKER)}
SESSION = session(
    max_turns=4,
    model_turn_timeout=None,
    command_timeout=10,
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
SUM_GRADER = """import json, pathlib
expected = json.loads(pathlib.Path("/tests/config.json").read_text())["expected"]
answer = pathlib.Path("/workspace/answer")
got = answer.read_text().strip() if answer.is_file() else None
print(f"got {got!r}")
print(float(got == expected))
"""
TEXT_GRADER = """import pathlib
print(float(pathlib.Path("/app/answer.txt").read_text().strip() == "twelve"))
"""
KEY_GRADER = """import pathlib
got = pathlib.Path("/app/answer.txt").read_text().strip()
print(float(got == pathlib.Path("/tests/key/answer.txt").read_text().strip()))
"""


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


@dataclass(frozen=True)
class RecordingFactory:
    """A container factory that is registered for lowering but never asked for a machine."""

    backend: Backend

    async def create(self, spec: MachineSpec):
        raise AssertionError(f"Unexpected machine for {spec}")


def shell_message(call_id: str, command: str) -> dict:
    arguments = json.dumps({"command": command})
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "shell", "arguments": arguments}}],
    }


async def run(lowered: LoweredTaskSpec, messages: list[dict]):
    engine = ShellboxRolloutEngine(ReplayModel(messages).complete, FACTORIES)
    restored = LoweredTaskSpec.model_validate_json(lowered.model_dump_json())
    return await engine.run(restored)


def lower_here(
    task: TaskSpec,
    task_machine: MachineSettings | None = None,
    verifier_machine: MachineSettings | None = None,
    host: MachineHost = MachineHost.LAPTOP,
) -> LoweredTaskSpec:
    return lower(
        task,
        host=host,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        session=SESSION,
        factories=FACTORIES,
    )


def sum_grader() -> GraderPackage:
    return python_grader(SUM_GRADER, {"expected": "12"}, environment=BASE, answer_path=None, timeout=20)


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
    lowered = lower_here(file_task(), SHELLSIM, VERIFIER)

    result = await run(
        lowered, [shell_message("c1", f"echo {answer} > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert "config.json" not in json.dumps(result.messages)


@pytest.mark.parametrize("answer,expected", [("12\n", 1.0), ("13\n", 0.0)])
async def test_python_grader_grades_a_supplied_workspace(answer, expected):
    lowered = lower_here(file_task(), SHELLSIM, VERIFIER)
    engine = ShellboxRolloutEngine(ReplayModel([]).complete, FACTORIES)
    messages = ({"role": "user", "content": "q"}, {"role": "assistant", "content": "done"})

    grade = await engine.grade_state(lowered, SuppliedState(messages, resources=(file("workspace/answer", answer),)))

    assert (grade.status, grade.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize("answer,expected", [("twelve", 1.0), ("eleven", 0.0)])
async def test_python_grader_reads_the_extracted_answer_of_a_task_without_a_machine(answer, expected):
    grader = python_grader(TEXT_GRADER, {}, environment=BASE, answer_path=ANSWER_PATH, timeout=20)
    lowered = lower_here(answer_task(grader), verifier_machine=VERIFIER)

    result = await run(lowered, [{"role": "assistant", "content": answer}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert lowered.runtime.verifier_machine is not None and lowered.runtime.verifier_machine.backend == "docker"


@pytest.mark.parametrize("answer,expected", [("twelve", 1.0), ("eleven", 0.0)])
async def test_python_grader_reads_its_private_files_from_the_tests_directory(answer, expected):
    grader = python_grader(
        KEY_GRADER,
        {},
        environment=BASE,
        answer_path=ANSWER_PATH,
        timeout=20,
        files=(file("key/answer.txt", "twelve\n"),),
    )
    lowered = lower_here(answer_task(grader), verifier_machine=VERIFIER)

    result = await run(lowered, [{"role": "assistant", "content": answer}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)


def test_python_grader_refuses_a_file_that_shadows_its_program_or_config():
    with pytest.raises(ValueError, match="shadow"):
        python_grader(
            TEXT_GRADER, {}, environment=BASE, answer_path=ANSWER_PATH, timeout=20, files=(file("config.json", "{}"),)
        )


def test_grader_environment_is_the_task_image_or_the_grader_base():
    assert grader_environment(None) == EnvironmentRequirements(docker_image=GRADER_BASE_IMAGE)
    assert grader_environment(IMAGE, env={"A": "1"}) == EnvironmentRequirements(
        docker_image=IMAGE, environment_variables={"A": "1"}
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
    assert (lowered.runtime.task_machine, lowered.runtime.verifier_machine) == (None, None)
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
        session=SESSION,
        factories={backend.value: RecordingFactory(backend)},
    )

    runtime = lowered.runtime
    assert runtime.task_machine is not None and runtime.verifier_machine is not None
    assert (runtime.task_machine.backend, runtime.verifier_machine.backend) == (backend.value, backend.value)
    assert (runtime.task_machine.network, runtime.task_machine.cpus, runtime.task_machine.memory_mb) == (
        NetworkPolicy.ALLOW,
        4,
        4096,
    )
    assert (runtime.task_machine.storage_mb, runtime.task_machine.user, runtime.task_machine.cleanup_timeout) == (
        10240,
        "root",
        20,
    )
    assert (runtime.verifier_machine.network, runtime.verifier_machine.user) == (NetworkPolicy.DENY, "0")
    assert LoweredTaskSpec.model_validate_json(lowered.model_dump_json()) == lowered


def test_script_grader_keeps_its_files_private_and_its_image_on_the_verifier():
    task = docker_task(docker_grader(env={"STRICT": "1"}))

    assert isinstance(task.grader, ScriptGrader)
    assert (task.grader.argv, task.grader.answer_path, task.grader.timeout) == (("python3", "/tests/grade.py"), None, 60)
    assert task.grader.environment == EnvironmentRequirements(docker_image=IMAGE, environment_variables={"STRICT": "1"})
    assert [item.path for item in task.resources.verifier] == ["grade.py"]
    assert [item.path for item in task.resources.worker] == ["app/run.sh"]
    assert task.resources.all == ()
    assert task.environment_requirements == EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        docker_image=IMAGE,
        working_directory="/app",
        setup_commands=("pip install pytest",),
        environment_variables={"MODE": "test"},
    )


def test_python_grader_writes_its_program_and_config_under_tests():
    grader = python_grader(SUM_GRADER, {"expected": "12"}, environment=BASE, answer_path=None, timeout=20)

    assert isinstance(grader.grader, ScriptGrader)
    assert (grader.grader.argv, grader.grader.reward) == (("python3", "/tests/grade.py"), StdoutReward())
    assert {item.path: json.loads(resource_bytes(item)) for item in grader.resources if item.path == "config.json"} == {
        "config.json": {"expected": "12"}
    }


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"files": (file("workspace/a", "1"),)}, "require a task machine"),
        ({"output_paths": ("/workspace/answer",)}, "require a task machine"),
        ({"environment": EnvironmentRequirements()}, "'shell' capability"),
    ],
)
def test_assemble_rejects_machine_inputs_without_a_shell_task_machine(fields, message):
    grader = answer_grader(ExactSpec(expected=("12",)))
    with pytest.raises(ValueError, match=message):
        assemble("bad", "Do it.", AnswerType.TEXT, PLAIN, grader, SOURCE, **{"environment": None, **fields})


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
    assert lower_here(task).runtime.verifier_machine is None


@pytest.mark.parametrize(
    "path,message",
    [
        ("workspace/answer", "absolute"),
        ("/workspace/../tests/x", "absolute and normalized"),
        ("/tests/answer", "outside /tests"),
        ("/logs/verifier/reward.json", "outside /tests"),
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
            session=SESSION,
            factories={"shellsim": ShellSimMachineFactory()},
        )
    slow_commands = session(
        max_turns=4,
        model_turn_timeout=None,
        command_timeout=20,
        tool_turn_timeout=20,
        total_turn_timeout=None,
        attempt_timeout=None,
        verifier_timeout=30,
        cleanup_timeout=10,
    )
    with pytest.raises(ValueError, match="command timeout must be less"):
        lower(
            file_task(),
            host=MachineHost.LAPTOP,
            task_machine=SHELLSIM,
            verifier_machine=VERIFIER,
            session=slow_commands,
            factories=FACTORIES,
        )


def test_lower_puts_a_shellsim_task_on_shellsim_and_its_grader_on_the_hosts_container_backend():
    lowered = lower(
        file_task(),
        host=MachineHost.IRIS,
        task_machine=SHELLSIM,
        verifier_machine=VERIFIER,
        session=SESSION,
        factories={"shellsim": ShellSimMachineFactory(), "gvisor": RecordingFactory(Backend.GVISOR)},
    )

    assert lowered.runtime.task_machine is not None and lowered.runtime.verifier_machine is not None
    assert (lowered.runtime.task_machine.backend, lowered.runtime.verifier_machine.backend) == ("shellsim", "gvisor")


def test_a_lock_only_grader_lowers_onto_the_local_backend():
    locked = EnvironmentRequirements(compatible_backends=(Backend.LOCAL,), packages_lock="gs://bucket/env/uv.lock")
    grader = python_grader(TEXT_GRADER, {}, environment=locked, answer_path=ANSWER_PATH, timeout=20)

    assert machine_backend(locked, MachineHost.IRIS) == Backend.LOCAL
    with pytest.raises(ValueError, match="Lock-only graders require a LocalMachineFactory"):
        lower(
            answer_task(grader),
            host=MachineHost.IRIS,
            task_machine=None,
            verifier_machine=VERIFIER,
            session=SESSION,
            factories={"local": RecordingFactory(Backend.LOCAL)},
        )
