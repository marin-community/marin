# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository conversion preserves the hidden grader and defers image builds."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from taskcompendium.models import AnswerType, ArtifactKind, ScriptGrader, TaskSpec
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask
from taskcompendium.runtime.resources import resource_bytes
from verifyit.modes import grade_pytest
from verifyit.spec import PytestSpec, parse_spec

from experiments.post_training.task_curation.datasets.tasktrove.repositories import sources
from experiments.post_training.task_curation.tests.conversion import convert_row, tasktrove_row


@dataclass(frozen=True)
class RepositoryFixture:
    task: TaskSpec
    workspace: Path
    tests_dir: Path
    spec: PytestSpec
    trusted_tests: str


def swesmith_pipeline():
    pipeline = next(source.pipeline for source in sources() if source.name == "tasktrove-swesmith")
    assert pipeline is not None
    return pipeline


def converted(blob: bytes) -> NormalizedTask:
    result = convert_row(swesmith_pipeline(), {"path": "swesmith-fixture", "task_binary": blob})
    assert isinstance(result, NormalizedTask)
    return result


def verifier_spec(task: TaskSpec) -> PytestSpec:
    resource = next(resource for resource in task.resources.verifier if resource.path == "taskcompendium-verifier.toml")
    spec = parse_spec(resource_bytes(resource).decode())
    assert isinstance(spec, PytestSpec)
    return spec


def test_swesmith_keeps_build_and_private_roles(repository_fixture):
    task = TaskSpec.model_validate_json(repository_fixture.task.model_dump_json())
    assert isinstance(task.grader, ScriptGrader)
    assert task.answer_type == AnswerType.WORKSPACE_STATE
    assert [(item.source, item.target, item.kind) for item in task.grader.artifacts] == [
        ("/testbed", "/testbed", ArtifactKind.DIRECTORY)
    ]
    assert task.grader.answer_path is None
    public = {item.path: resource_bytes(item) for item in (*task.resources.all, *task.resources.worker)}
    private = {item.path: resource_bytes(item) for item in task.resources.verifier}
    oracle = {item.path: resource_bytes(item) for item in task.resources.oracle}
    assert public == {"setup_files/requirements.txt": b"pytest\n"}
    assert "config.json" not in private and "test_state.py" not in private
    assert json.loads(oracle["source_archive/tests/config.json"])["patch"] == "oracle-only"
    assert oracle["solution/solve.sh"] == b"oracle-only"
    assert private["trusted_test_paths.txt"] == b"tests/test_calc.py\n"
    spec = verifier_spec(task)
    assert spec.must_pass == ("tests/test_calc.py::test_add",)
    assert spec.protected_paths_files == ("trusted_test_paths.txt",)
    assert "git_repository" in task.environment_requirements.capabilities
    context = task.environment_requirements.docker_build
    assert context is not None
    build = {item.path: resource_bytes(item) for item in context.files}
    assert build["helper.sh"] == b"#!/bin/sh\nexit 0\n"
    assert "taskcompendium-repository-setup.sh" not in build
    grader_context = task.grader.environment.docker_build
    assert grader_context is not None
    grader_build = {item.path: resource_bytes(item) for item in grader_context.files}
    assert grader_build["taskcompendium-public/setup_files/requirements.txt"] == public["setup_files/requirements.txt"]
    assert (
        grader_build["taskcompendium-repository-setup.sh"]
        == oracle["instruction.md"].split(b"```bash\n", 1)[1].split(b"\n```", 1)[0]
    )
    assert not any(path.startswith(("tests/", "solution/")) for path in build)
    assert task.context.events[0].content.encode() == oracle["instruction.md"]


@pytest.mark.parametrize(
    "problem, reason", [("doctest", "unsupported_variant"), ("missing_ref", "missing_repository_ref")]
)
def test_unusable_repository_grading_contract_is_rejected(problem, reason):
    files = source_files("abcdef0")
    config = json.loads(files["tests/config.json"])
    if problem == "doctest":
        config["FAIL_TO_PASS"] = ["docs/example.rst::example"]
    else:
        files["instruction.md"] = b"Fix the spin bug."
    files["tests/config.json"] = json.dumps(config).encode()
    result = convert_row(swesmith_pipeline(), tasktrove_row(files))
    assert isinstance(result, ImportRejection)
    assert result.reason == reason


def git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()


def source_files(commit: str) -> dict[str, bytes]:
    return {
        "instruction.md": (
            (
                "## Environment Setup (complete these steps first)\n\n```bash\n"
                f"cd /testbed\ngit checkout {commit}\n```\nFix add.\n"
            ).encode()
        ),
        "environment/Dockerfile": b"FROM python:3.10-bookworm\nRUN pip install --upgrade pip uv pytest\n",
        "environment/helper.sh": b"#!/bin/sh\nexit 0\n",
        "setup_files/requirements.txt": b"pytest\n",
        "tests/config.json": (
            json.dumps(
                {
                    "repo": "fixture/calc",
                    "FAIL_TO_PASS": ["tests/test_calc.py::test_add"],
                    "PASS_TO_PASS": [],
                    "patch": "oracle-only",
                }
            ).encode()
        ),
        "tests/test.sh": f"install_trusted_test_paths.sh /testbed {commit} /tests/trusted_test_paths.txt\n".encode(),
        "tests/trusted_test_paths.txt": b"tests/test_calc.py\n",
        "solution/solve.sh": b"oracle-only",
    }


@pytest.fixture
def repository_fixture(tmp_path):
    """A controlled local repository; no source archive shell script is executed."""
    workspace = tmp_path / "repository"
    workspace.mkdir()
    (workspace / "tests").mkdir()
    (workspace / "calc.py").write_text("def add(a, b): return a - b\n")
    (workspace / "conftest.py").write_text("# Trusted pytest configuration.\n")
    trusted_tests = "from calc import add\ndef test_add(): assert add(2, 3) == 5\n"
    (workspace / "tests/test_calc.py").write_text(trusted_tests)
    git(workspace, "init", "-q")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "add", ".")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
    commit = git(workspace, "rev-parse", "HEAD")
    files = source_files(commit)
    task = converted(tasktrove_row(files)["task_binary"]).task
    tests_dir = tmp_path / "private"
    tests_dir.mkdir()
    for resource in task.resources.verifier:
        path = tests_dir / resource.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource_bytes(resource))
    spec = replace(verifier_spec(task), workspace=str(workspace), python=sys.executable, timeout=30)
    return RepositoryFixture(task, workspace, tests_dir, spec, trusted_tests)


@pytest.mark.parametrize("repair, tamper, expected", [(False, False, 0), (True, False, 1), (False, True, 0)])
def test_converted_grader_restores_tests_and_grades_product_code(repository_fixture, repair, tamper, expected):
    workspace = repository_fixture.workspace
    if repair:
        (workspace / "calc.py").write_text("def add(a, b): return a + b\n")
    if tamper:
        (workspace / "tests/test_calc.py").write_text("def test_add(): pass\n")
    reward = grade_pytest.grade(repository_fixture.spec, repository_fixture.tests_dir, workspace)
    assert reward.reward == expected
    assert (workspace / "tests/test_calc.py").read_text() == repository_fixture.trusted_tests
