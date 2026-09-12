# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Docker trials using real source graders and unprivileged candidates."""

import json
import shlex
import subprocess

import msgspec
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.harbor.container import grade_in_container
from taskcompendium.harbor.runner import run_trial
from taskcompendium.lowering import export_task
from taskcompendium.models import (
    AnswerRequirements,
    ChatWithTools,
    ContainerRuntime,
    DockerEnvironment,
    Embedded,
    ExecutionConfig,
    FinalState,
    ImageOverlay,
    Outcome,
    Protocol,
    Resource,
    ResourceRole,
    Source,
    TaskMetadata,
    TaskSpecification,
    VerifierSpec,
)

pytestmark = pytest.mark.docker


def _spec(image, mode=Mode.STDIO, parameters=None, resources=()):
    return TaskSpecification(
        id="code/sum",
        instructions="Write main.py to read two integers and print their sum.",
        environment=DockerEnvironment(image),
        resources=resources
        or (
            Resource("cases/input_1.txt", (ResourceRole.VERIFIER,), Embedded(b"2 3\n")),
            Resource("cases/output_1.txt", (ResourceRole.VERIFIER,), Embedded(b"5\n")),
            Resource("cases/input_2.txt", (ResourceRole.VERIFIER,), Embedded(b"-1 8\n")),
            Resource("cases/output_2.txt", (ResourceRole.VERIFIER,), Embedded(b"7\n")),
        ),
        verifier=VerifierSpec(mode, parameters or {"command": "python3 main.py"}),
        verifier_runtime=ContainerRuntime(image),
        metadata=TaskMetadata(Source("test", "1", "0", "1")),
        answer_requirements=AnswerRequirements("final_state"),
    )


@pytest.mark.parametrize("source,reward", [("print(sum(map(int,input().split())))", 1.0), ("print(99)", 0.0)])
async def test_harbor_docker_trial_grades_code_and_ignores_agent_reward(tmp_path, runtime_image, source, reward):
    spec = _spec(runtime_image)
    protocol = Protocol("code", ChatWithTools(), FinalState(("main.py", "reward.txt")))
    task = export_task(
        spec,
        protocol,
        ExecutionConfig("replay", spec.environment),
        tmp_path / "task",
        agent_kwargs={
            "commands": [f"printf '%s' {shlex.quote(source)} > main.py", "echo 1 > reward.txt"],
        },
    )
    execution = json.loads((task / "execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", "code")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": reward}


def test_container_candidate_cannot_read_gold_or_forge_supervisor_result(tmp_path, runtime_image):
    source = """import pathlib
for path in ['/input/specification.json', '/result/result.json']:
    try:
        pathlib.Path(path).write_text('{"status":"graded","reward":1,"detail":{}}')
    except OSError as error:
        assert error.errno in (13, 30)
    else:
        raise AssertionError('candidate escaped isolation')
try:
    pathlib.Path('/input/specification.json').read_text()
except PermissionError:
    pass
else:
    raise AssertionError('candidate read gold')
print(sum(map(int,input().split())))
"""
    (tmp_path / "main.py").write_text(source)
    result = grade_in_container(
        _spec(runtime_image), Protocol("code", ChatWithTools(), FinalState(("main.py",))), None, tmp_path
    )
    assert result.status == Outcome.GRADED
    assert result.reward == 1.0


@pytest.mark.parametrize("add,reward", [("+", 1.0), ("-", 0.0)])
def test_container_compiled_program_uses_original_stdio_grader(tmp_path, runtime_image, add, reward):
    (tmp_path / "main.cpp").write_text(
        "#include <iostream>\nint main(){int a,b;std::cin>>a>>b;std::cout<<(a" + add + "b);}"
    )
    spec = _spec(runtime_image, parameters={"command": "./main", "build": "g++ -o main main.cpp"})
    result = grade_in_container(spec, Protocol("cpp", ChatWithTools(), FinalState(("main.cpp",))), None, tmp_path)
    assert result.status == Outcome.GRADED
    assert result.reward == reward


@pytest.mark.parametrize("value,reward", [(5, 1.0), (99, 0.0)])
def test_container_pytest_uses_hidden_original_tests(tmp_path, runtime_image, value, reward):
    (tmp_path / "solution.py").write_text(f"def add(a,b): return {value}\n")
    resources = (
        Resource(
            "tests/test_solution.py",
            (ResourceRole.VERIFIER,),
            Embedded(b"from solution import add\ndef test_add(): assert add(2,3) == 5\n"),
        ),
    )
    spec = _spec(
        runtime_image,
        Mode.PYTEST,
        {
            "paths": ["tests/test_solution.py"],
            "restore": ["tests/test_solution.py"],
            "must_pass": ["tests/test_solution.py::test_add"],
        },
        resources,
    )
    result = grade_in_container(spec, Protocol("pytest", ChatWithTools(), FinalState(("solution.py",))), None, tmp_path)
    assert result.status == Outcome.GRADED
    assert result.reward == reward


@pytest.mark.parametrize("value,reward", [("good", 1.0), ("bad", 0.0)])
def test_container_script_preserves_original_reward_contract(tmp_path, runtime_image, value, reward):
    (tmp_path / "answer.txt").write_text(value)
    resources = (
        Resource(
            "check.sh",
            (ResourceRole.VERIFIER,),
            Embedded(b'if [ "$(cat answer.txt)" = good ]; then echo 1; else echo 0; fi\n'),
        ),
    )
    spec = _spec(runtime_image, Mode.SCRIPT, {"path": "check.sh"}, resources)
    result = grade_in_container(spec, Protocol("script", ChatWithTools(), FinalState(("answer.txt",))), None, tmp_path)
    assert result.status == Outcome.GRADED
    assert result.reward == reward


@pytest.fixture(scope="module")
def overlay_image(runtime_image, tmp_path_factory):
    directory = tmp_path_factory.mktemp("overlay-image")
    subprocess.run(["docker", "tag", runtime_image, "taskcompendium-overlay-base:validation"], check=True)
    (directory / "Dockerfile").write_text(
        "FROM taskcompendium-overlay-base:validation\n"
        "RUN mkdir -p /app/.venv/bin && "
        "ln -s /usr/local/bin/python3 /app/.venv/bin/python && "
        "echo original > /app/.venv/dependency && echo stale > /app/deleted.txt\n"
    )
    subprocess.run(["docker", "build", "-q", "-t", "taskcompendium-overlay:validation", str(directory)], check=True)
    return subprocess.check_output(
        ["docker", "image", "inspect", "taskcompendium-overlay:validation", "--format", "{{.Id}}"], text=True
    ).strip()


async def test_overlay_trial_excludes_dependencies_and_replaces_deleted_files(tmp_path, overlay_image):
    spec = msgspec.structs.replace(
        _spec(overlay_image, parameters={"command": ".venv/bin/python main.py"}),
        verifier_runtime=ContainerRuntime(overlay_image, workspace=ImageOverlay((".venv",))),
    )
    source = """import os, pathlib
assert os.getuid() == 65534
assert pathlib.Path('.venv/bin/python').is_symlink()
assert pathlib.Path('.venv/dependency').read_text().strip() == 'original'
assert not pathlib.Path('deleted.txt').exists()
for path in ['/input/specification.json', '/result/result.json']:
    try:
        pathlib.Path(path).write_text('forged')
    except PermissionError:
        pass
    else:
        raise AssertionError('candidate escaped isolation')
print(sum(map(int,input().split())))
"""
    task = export_task(
        spec,
        Protocol("overlay", ChatWithTools(), FinalState((".",), excluded_paths=(".venv",))),
        ExecutionConfig("replay", spec.environment),
        tmp_path / "task",
        agent_kwargs={
            "commands": [
                "rm deleted.txt; echo tampered > .venv/dependency",
                f"printf '%s' {shlex.quote(source)} > main.py",
            ]
        },
    )
    result = await run_trial(task, json.loads((task / "execution.json").read_text()), tmp_path / "trials", "overlay")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}


def test_overlay_rejects_submitted_dependency_directory(tmp_path, overlay_image):
    (tmp_path / ".venv").mkdir()
    spec = msgspec.structs.replace(
        _spec(overlay_image), verifier_runtime=ContainerRuntime(overlay_image, workspace=ImageOverlay((".venv",)))
    )
    result = grade_in_container(
        spec, Protocol("overlay", ChatWithTools(), FinalState((".",), excluded_paths=(".venv",))), None, tmp_path
    )
    assert result.status == Outcome.INFRA_ERROR
    assert result.reward is None


async def test_overlay_trial_rejects_submitted_symlink(tmp_path, overlay_image):
    spec = msgspec.structs.replace(
        _spec(overlay_image), verifier_runtime=ContainerRuntime(overlay_image, workspace=ImageOverlay((".venv",)))
    )
    task = export_task(
        spec,
        Protocol("overlay", ChatWithTools(), FinalState((".",), excluded_paths=(".venv",))),
        ExecutionConfig("replay", spec.environment),
        tmp_path / "task",
        agent_kwargs={"commands": ["ln -s .venv/dependency main.py"]},
    )
    result = await run_trial(task, json.loads((task / "execution.json").read_text()), tmp_path / "trials", "symlink")
    assert result.exception_info is not None
    assert result.verifier_result is None
