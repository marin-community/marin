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

FIXTURES = Path(__file__).parent / "fixtures"


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
    resource = next(resource for resource in task.resources.verifier if resource.path == "verifier.toml")
    spec = parse_spec(resource_bytes(resource).decode())
    assert isinstance(spec, PytestSpec)
    return spec


def test_original_swesmith_keeps_build_and_private_roles():
    result = converted((FIXTURES / "swesmith.tar.gz").read_bytes())
    task = TaskSpec.model_validate_json(result.task.model_dump_json())
    assert isinstance(task.grader, ScriptGrader)
    assert task.answer_type == AnswerType.WORKSPACE_STATE
    assert [(item.source, item.target, item.kind) for item in task.grader.artifacts] == [
        ("/testbed", "/testbed", ArtifactKind.DIRECTORY)
    ]
    assert task.grader.answer_path is None
    public = {item.path: resource_bytes(item) for item in (*task.resources.all, *task.resources.worker)}
    private = {item.path: resource_bytes(item) for item in task.resources.verifier}
    oracle = {item.path: resource_bytes(item) for item in task.resources.oracle}
    assert set(public) == {"setup_files/requirements.txt"}
    assert "config.json" not in private and "test_state.py" not in private
    original = json.loads(oracle["source_archive/tests/config.json"])
    assert original["patch"]
    assert "solution/solve.sh" in oracle
    assert private["trusted_test_paths.txt"] == oracle["source_archive/tests/trusted_test_paths.txt"]
    spec = verifier_spec(task)
    assert spec.must_pass == tuple(original["FAIL_TO_PASS"])
    assert spec.must_not_break == tuple(original["PASS_TO_PASS"])
    assert spec.protected_paths_files == ("trusted_test_paths.txt",)
    assert task.environment_requirements.docker_image is None
    assert task.environment_requirements.docker_build == task.grader.environment.docker_build
    context = task.environment_requirements.docker_build
    assert context is not None
    build = {item.path: resource_bytes(item) for item in context.files}
    assert build["Dockerfile"].startswith(b"FROM python:3.10-bookworm\n")
    assert build["taskcompendium-verifyit/src/verifyit/modes/grade_pytest.py"]
    assert build["taskcompendium-public/setup_files/requirements.txt"] == public["setup_files/requirements.txt"]
    assert (
        build["taskcompendium-repository-setup.sh"]
        == oracle["instruction.md"].split(b"```bash\n", 1)[1].split(b"\n```", 1)[0]
    )
    assert not any(path.startswith(("tests/", "solution/")) for path in build)
    assert task.context.events[0].content.encode() == oracle["instruction.md"]


def test_original_uncollectable_fail_to_pass_is_rejected():
    result = convert_row(
        swesmith_pipeline(),
        {"path": "swesmith-doctest", "task_binary": (FIXTURES / "swesmith_doctest.tar.gz").read_bytes()},
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_variant"
    assert "FAIL_TO_PASS" in result.detail


def git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()


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
    files = {
        "instruction.md": (
            (
                "## Environment Setup (complete these steps first)\n\n```bash\n"
                f"cd /testbed\ngit checkout {commit}\n```\nFix add.\n"
            ).encode()
        ),
        "environment/Dockerfile": b"FROM python:3.10-bookworm\nRUN pip install --upgrade pip uv pytest\n",
        "environment/helper.sh": b"#!/bin/sh\nexit 0\n",
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


@pytest.mark.parametrize(
    "hook_state", ["untracked", "ignored", "modified", "committed", "assume-unchanged", "skip-worktree"]
)
def test_converted_grader_removes_candidate_pytest_hooks(repository_fixture, hook_state):
    workspace, tests_dir, spec = repository_fixture.workspace, repository_fixture.tests_dir, repository_fixture.spec
    path = "tests/conftest.py" if hook_state in ("untracked", "ignored") else "conftest.py"
    if hook_state == "ignored":
        (workspace / ".gitignore").write_text(path + "\n")
    if hook_state in ("assume-unchanged", "skip-worktree"):
        git(workspace, "update-index", f"--{hook_state}", path)
    (workspace / path).write_text(
        "def pytest_collection_modifyitems(items):\n" "    for item in items:\n" "        item.obj = lambda: None\n"
    )
    if hook_state == "committed":
        git(workspace, "add", path)
        git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "hook")
    product = (workspace / "calc.py").read_text()
    # Establish the attack's effect before checking the trusted restoration boundary.
    assert grade_pytest.grade(replace(spec, setup=None), tests_dir, workspace).reward == 1
    assert grade_pytest.grade(spec, tests_dir, workspace).reward == 0
    assert (workspace / "calc.py").read_text() == product
    if path == "conftest.py":
        assert (workspace / path).read_text() == "# Trusted pytest configuration.\n"
    else:
        assert not (workspace / path).exists()


@pytest.mark.parametrize("replacement_kind", ["commit", "blob"])
def test_converted_grader_ignores_candidate_git_replacement_refs(repository_fixture, replacement_kind):
    workspace, tests_dir, spec = repository_fixture.workspace, repository_fixture.tests_dir, repository_fixture.spec
    trusted = git(workspace, "rev-parse", "HEAD")
    original_blob = git(workspace, "rev-parse", f"{trusted}:tests/test_calc.py")
    forged_tests = "def test_add(): pass\n"
    (workspace / "tests/test_calc.py").write_text(forged_tests)
    git(workspace, "add", "tests/test_calc.py")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "forged")
    replacement = git(workspace, "rev-parse", "HEAD" if replacement_kind == "commit" else "HEAD:tests/test_calc.py")
    git(workspace, "replace", trusted if replacement_kind == "commit" else original_blob, replacement)
    assert git(workspace, "show", f"{trusted}:tests/test_calc.py") == forged_tests.strip()
    assert grade_pytest.grade(replace(spec, setup=None), tests_dir, workspace).reward == 1
    assert grade_pytest.grade(spec, tests_dir, workspace).reward == 0
    assert (workspace / "tests/test_calc.py").read_text() == repository_fixture.trusted_tests


def test_build_context_keeps_original_auxiliary_bytes(repository_fixture):
    task = repository_fixture.task
    assert task.environment_requirements.docker_build is not None
    files = {item.path: resource_bytes(item) for item in task.environment_requirements.docker_build.files}
    assert files["helper.sh"] == b"#!/bin/sh\nexit 0\n"
