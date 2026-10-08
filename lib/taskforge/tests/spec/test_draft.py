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
from taskcompendium.models import AnswerType, EnvironmentRequirements, FunctionDefinition, Source, TaskSpec, VerifierSpec
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.shell_verifier import (
    ArtifactKind,
    ExitCodeReward,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.submission import FinalAction, JsonValueAnswer, PlainText, SubmissionConvention
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
from taskforge.spec.draft import (
    FILESYSTEM_CAPABILITY,
    SHELL_CAPABILITY,
    MachineSettings,
    Resources,
    answer_verifier,
    assemble,
    file,
    lower,
    machine,
    requirements,
    reward_file,
    script_verifier,
    session,
    shell_verifier,
)

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
PLAIN = PlainText(id="plain")
FACTORIES = {"shellsim": ShellSimMachineFactory()}
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
IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
SUM_GRADER = """import json, os, pathlib
workspace = pathlib.Path(os.environ["VERIFYIT_WORKSPACE"])
expected = json.loads(pathlib.Path(os.environ["VERIFYIT_TESTS_DIR"], "config.json").read_text())["expected"]
answer = workspace / "captured/workspace/answer"
got = answer.read_text().strip() if answer.is_file() else None
verdict = {"status": "scored", "reward": float(got == expected), "detail": {"got": got}}
pathlib.Path(os.environ["VERIFYIT_LOGS_DIR"], "verdict.json").write_text(json.dumps(verdict))
"""
TEXT_GRADER = """import json, os, pathlib
got = pathlib.Path(os.environ["VERIFYIT_WORKSPACE"], "answer.txt").read_text().strip()
verdict = {"status": "scored", "reward": float(got == "twelve"), "detail": {}}
pathlib.Path(os.environ["VERIFYIT_LOGS_DIR"], "verdict.json").write_text(json.dumps(verdict))
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


async def run(lowered: LoweredTaskSpec, messages: list[dict], convention: SubmissionConvention = PLAIN):
    engine = ShellboxRolloutEngine(ReplayModel(messages).complete, FACTORIES, convention=convention)
    restored = LoweredTaskSpec.model_validate_json(lowered.model_dump_json())
    return await engine.run(restored)


def lower_here(task: TaskSpec, task_machine: MachineSettings | None = None) -> LoweredTaskSpec:
    return lower(
        task,
        host=MachineHost.LAPTOP,
        task_machine=task_machine,
        verifier_machine=None,
        session=SESSION,
        factories=FACTORIES,
    )


def file_task(grader: GraderPackage | None = None) -> TaskSpec:
    return assemble(
        "sum-file",
        "Write the sum of the numbers in /workspace/numbers.txt to /workspace/answer.",
        AnswerType.FILE,
        grader or script_verifier(SUM_GRADER, {"expected": "12"}, timeout=20),
        SOURCE,
        environment=requirements(image=None),
        files=(file("workspace/numbers.txt", "5\n7\n"),),
        output_paths=("/workspace/answer",),
    )


def answer_task(grader: GraderPackage, answer_type: AnswerType = AnswerType.TEXT, **fields) -> TaskSpec:
    return assemble("answer", "What is six plus six?", answer_type, grader, SOURCE, environment=None, **fields)


def docker_task(grader: GraderPackage) -> TaskSpec:
    return assemble(
        "docker",
        "Fix the bug in /app.",
        AnswerType.WORKSPACE_STATE,
        grader,
        SOURCE,
        environment=requirements(image=IMAGE, setup=("pip install pytest",), workdir="/app", env={"MODE": "test"}),
        files=(file("app/run.sh", "#!/bin/sh\npython3 main.py\n", mode=0o755),),
        output_paths=("/app",),
        tags=("taskforge",),
    )


def docker_grader(**fields) -> GraderPackage:
    return shell_verifier(
        ("python3", "/tests/grade.py"),
        reward_file("/logs/verifier/reward.json", RewardFileFormat.JSON, key="score", pass_above=0.5),
        image=IMAGE,
        files=(file("grade.py", "print(1)\n"),),
        **fields,
    )


def test_file_lands_relative_to_the_machine_root_with_its_mode():
    resource = file("app/run.sh", "#!/bin/sh\n", mode=0o755)

    assert (resource.path, resource.mode, resource_bytes(resource)) == ("app/run.sh", "755", b"#!/bin/sh\n")
    with pytest.raises(ValueError):
        file("/workspace/answer", "12")


@pytest.mark.parametrize("answer,expected", [("12", 1.0), ("13", 0.0)])
async def test_script_verifier_grades_the_captured_output_of_a_shellsim_run(answer, expected):
    lowered = lower_here(file_task(), SHELLSIM)

    result = await run(
        lowered, [shell_message("c1", f"echo {answer} > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert "config.json" not in json.dumps(result.messages)


@pytest.mark.parametrize("answer,expected", [("12\n", 1.0), ("13\n", 0.0)])
async def test_script_verifier_grades_a_supplied_workspace(answer, expected):
    task = file_task()
    lowered = lower_here(task, SHELLSIM)
    engine = ShellboxRolloutEngine(ReplayModel([]).complete, FACTORIES, convention=PLAIN)
    messages = ({"role": "user", "content": "q"}, {"role": "assistant", "content": "done"})

    grade = await engine.grade_state(lowered, SuppliedState(messages, resources=(file("workspace/answer", answer),)))

    assert (grade.status, grade.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize("answer,expected", [("twelve", 1.0), ("eleven", 0.0)])
async def test_script_verifier_grades_the_text_answer_of_a_task_without_a_machine(answer, expected):
    lowered = lower_here(answer_task(script_verifier(TEXT_GRADER, {}, timeout=20)))

    result = await run(lowered, [{"role": "assistant", "content": answer}])

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize(
    "spec,right,wrong",
    [
        (ExactSpec(expected=("twelve",)), "Twelve", "eleven"),
        (NumericSpec(expected="12", tolerance_abs=0.0, tolerance_rel=0.0), "12", "13"),
        (McqSpec(expected="B"), "B", "C"),
    ],
)
async def test_answer_verifier_grades_the_final_reply_after_a_json_round_trip(spec, right, wrong):
    lowered = lower_here(answer_task(answer_verifier(spec), system="Answer briefly."))

    rewards = [(await run(lowered, [{"role": "assistant", "content": text}])).grade.reward for text in (right, wrong)]

    assert rewards == [1.0, 0.0]
    assert lowered.runtime.task_machine is None
    assert lowered.task.context.events[0].content == "Answer briefly."


async def test_predicted_action_verifier_grades_a_native_action():
    lookup = FunctionDefinition(name="lookup", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
    spec = PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"city": "Paris"}),))
    lowered = lower_here(answer_task(answer_verifier(spec), AnswerType.NATIVE_ACTION, final_tools=(lookup,)))

    def call(city: str) -> dict:
        arguments = json.dumps({"city": city})
        return {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": arguments}}],
        }

    convention = FinalAction(id="action")
    rewards = [(await run(lowered, [call(city)], convention)).grade.reward for city in ("Paris", "Rome")]

    assert rewards == [1.0, 0.0]


async def test_structured_exact_verifier_grades_a_json_answer():
    lowered = lower_here(answer_task(answer_verifier(StructuredExactSpec(expected={"total": 12})), AnswerType.JSON))
    convention = JsonValueAnswer(id="json")

    rewards = [
        (await run(lowered, [{"role": "assistant", "content": text}], convention)).grade.reward
        for text in ('{"total": 12}', '{"total": 13}')
    ]

    assert rewards == [1.0, 0.0]


@pytest.mark.parametrize("spec", [MathSpec(expected="12"), PytestSpec(paths=("test_answer.py",))])
def test_answer_verifier_accepts_only_candidate_modes(spec):
    with pytest.raises(ValueError, match="not a candidate-mode"):
        answer_verifier(spec)


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


def test_shell_verifier_keeps_its_files_private_and_its_image_on_the_verifier():
    task = docker_task(docker_grader(env={"STRICT": "1"}))

    parameters = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)

    assert parameters.argv == ("python3", "/tests/grade.py")
    assert task.verifier.environment_requirements == EnvironmentRequirements(
        docker_image=IMAGE, environment_variables={"STRICT": "1"}
    )
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


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"files": (file("workspace/a", "1"),)}, "require a task machine"),
        ({"output_paths": ("/workspace/answer",)}, "require a task machine"),
        ({"environment": EnvironmentRequirements()}, "'shell' capability"),
    ],
)
def test_assemble_rejects_machine_inputs_without_a_shell_task_machine(fields, message):
    grader = answer_verifier(ExactSpec(expected=("12",)))
    with pytest.raises(ValueError, match=message):
        assemble("bad", "Do it.", AnswerType.TEXT, grader, SOURCE, **{"environment": None, **fields})


SCRIPT = script_verifier(SUM_GRADER, {"expected": "12"}, timeout=20)
EXACT = answer_verifier(ExactSpec(expected=("12",)))
SHELL = shell_verifier(("true",), ExitCodeReward(), image=IMAGE)


@pytest.mark.parametrize(
    "answer_type,grader,image,output_paths,message",
    [
        (AnswerType.FILE, SHELL, None, ("/workspace/answer",), "image-backed task machine"),
        (AnswerType.STATE, SCRIPT, None, (), "cannot grade a 'state' answer"),
        (AnswerType.JSON, SCRIPT, None, (), "cannot grade a 'json' answer"),
        (AnswerType.FILE, SCRIPT, None, (), "needs output paths"),
        (AnswerType.WORKSPACE_STATE, SCRIPT, IMAGE, (), "needs output paths"),
        (AnswerType.FILE, EXACT, None, ("/workspace/answer",), "requires a script or shell verifier"),
        (AnswerType.NATIVE_ACTION, EXACT, None, (), "json or number or text answer"),
        (AnswerType.TEXT, answer_verifier(StructuredExactSpec(expected={"t": 1})), None, (), "json answer"),
        (AnswerType.JSON, answer_verifier(NumericSpec("12", 0.0, 0.0)), None, (), "number or text answer"),
        (
            AnswerType.TEXT,
            GraderPackage(VerifierSpec(kind="math", parameters_json='{"expected": "12"}')),
            None,
            (),
            "does not grade 'math'",
        ),
    ],
)
def test_assemble_rejects_graders_that_cannot_see_the_answer(answer_type, grader, image, output_paths, message):
    final_tools = (FunctionDefinition(name="f", parameters={}),) if answer_type == AnswerType.NATIVE_ACTION else ()
    with pytest.raises(ValueError, match=message):
        assemble(
            "bad",
            "Do it.",
            answer_type,
            grader,
            SOURCE,
            environment=requirements(image=image),
            output_paths=output_paths,
            final_tools=final_tools,
        )


@pytest.mark.parametrize(
    "grader",
    [
        GraderPackage(VerifierSpec(kind="exact", parameters_json='{"expected": 12}')),
        GraderPackage(VerifierSpec(kind="shell", parameters_json='{"argv": []}')),
        GraderPackage(VerifierSpec(kind="skipped", parameters_json="{}")),
    ],
)
def test_assemble_rejects_invalid_grader_parameters(grader):
    with pytest.raises(ValueError):
        assemble("bad", "Do it.", AnswerType.TEXT, grader, SOURCE, environment=requirements(image=IMAGE))


@pytest.mark.parametrize(
    "path,message",
    [
        ("workspace/answer", "absolute"),
        ("/workspace/../tests/x", "absolute and normalized"),
        ("/tests/answer", "private grading files"),
        ("/logs/verifier/reward.json", "private grading files"),
    ],
)
def test_assemble_rejects_output_paths_outside_the_agents_reach(path, message):
    with pytest.raises(ValueError, match=message):
        assemble(
            "bad", "Do it.", AnswerType.FILE, SCRIPT, SOURCE, environment=requirements(image=None), output_paths=(path,)
        )


def test_assemble_rejects_private_grader_content_shipped_to_the_agent():
    copy = file("workspace/notes.py", SUM_GRADER)
    with pytest.raises(ValueError, match=r"agent-visible: \['grader.py'\]"):
        assemble(
            "leak",
            "Write the answer.",
            AnswerType.FILE,
            SCRIPT,
            SOURCE,
            environment=requirements(image=None),
            files=(copy,),
            output_paths=("/workspace/answer",),
        )


@pytest.mark.parametrize(
    "task_machine,verifier_machine,message",
    [
        (None, None, "task machine is given exactly"),
        (SHELLSIM, SHELLSIM, "verifier machine is given exactly"),
    ],
)
def test_lower_takes_exactly_the_machines_the_task_needs(task_machine, verifier_machine, message):
    with pytest.raises(ValueError, match=message):
        lower(
            file_task(),
            host=MachineHost.LAPTOP,
            task_machine=task_machine,
            verifier_machine=verifier_machine,
            session=SESSION,
            factories=FACTORIES,
        )
    with pytest.raises(ValueError, match="task machine is given exactly"):
        lower_here(answer_task(EXACT), SHELLSIM)


def test_lower_requires_a_verifier_machine_for_a_shell_grader():
    with pytest.raises(ValueError, match="verifier machine is given exactly"):
        lower(
            docker_task(docker_grader()),
            host=MachineHost.LAPTOP,
            task_machine=SHELLSIM,
            verifier_machine=None,
            session=SESSION,
            factories={"docker": RecordingFactory(Backend.DOCKER)},
        )


def test_lower_propagates_rolloutengines_rejections():
    with pytest.raises(ValueError, match="Unknown Shellbox factory: 'docker'"):
        lower(
            docker_task(docker_grader()),
            host=MachineHost.LAPTOP,
            task_machine=SHELLSIM,
            verifier_machine=SHELLSIM,
            session=SESSION,
            factories=FACTORIES,
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
            verifier_machine=None,
            session=slow_commands,
            factories=FACTORIES,
        )


def test_lower_picks_shellsim_for_a_task_without_an_image_on_every_host():
    lowered = lower(
        file_task(),
        host=MachineHost.IRIS,
        task_machine=SHELLSIM,
        verifier_machine=None,
        session=SESSION,
        factories=FACTORIES,
    )

    assert lowered.runtime.task_machine is not None
    assert (lowered.runtime.task_machine.backend, lowered.runtime.verifier_machine) == ("shellsim", None)
