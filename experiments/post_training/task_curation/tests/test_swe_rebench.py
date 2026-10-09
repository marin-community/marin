# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Patched repository graders preserve private tests and reject unresolved source results."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from taskcompendium.convert.tasktrove import archive_files, unpack_task_binary
from taskcompendium.models import AnswerType, ScriptGrader
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask
from taskcompendium.runtime.resources import resource_bytes
from verifyit.modes import grade_pytest
from verifyit.spec import PytestSpec, ScriptSpec, parse_spec

from experiments.post_training.task_curation.datasets.tasktrove.repositories import sources
from experiments.post_training.task_curation.tests.conversion import convert_row, tasktrove_row

FIXTURES = Path(__file__).parent / "fixtures"


def pipeline():
    source = next(source for source in sources() if source.name == "tasktrove-swe_rebench")
    assert source.pipeline is not None
    return source.pipeline


def original(language: str):
    blob = (FIXTURES / f"swe_rebench_{language}.tar.gz").read_bytes()
    return archive_files(unpack_task_binary({"path": "fixture", "task_binary": blob}, ConversionContext({}, None)))


def converted(files) -> NormalizedTask:
    result = convert_row(pipeline(), tasktrove_row(files))
    assert isinstance(result, NormalizedTask)
    return result


@pytest.mark.parametrize("language", ["python", "go"])
def test_source_archives_keep_private_graders_and_deferred_dependencies(language):
    source = original(language)
    task = converted(source.files).task
    assert isinstance(task.grader, ScriptGrader)
    assert task.answer_type == AnswerType.WORKSPACE_STATE
    assert task.grader.artifacts[0].source == task.grader.artifacts[0].target == "/testbed"
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    public = {resource.path for resource in task.resources.worker}
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    for path in ("install_trusted_test_paths.sh", "install_trusted_test_patch.sh", "test_patch.diff"):
        assert private[path] == source.files[f"tests/{path}"]
        assert path not in public
    assert oracle["source_archive/tests/test.sh"] == source.files["tests/test.sh"]
    assert oracle["solution/solve.sh"] == source.files["solution/solve.sh"]
    assert "verifier.toml" not in private
    spec = parse_spec(private["taskcompendium-verifier.toml"].decode())
    if language == "python":
        assert isinstance(spec, PytestSpec)
        config = json.loads(source.text("tests/config.json"))
        assert spec.must_pass == tuple(config["FAIL_TO_PASS"])
        assert spec.must_not_break == tuple(config["PASS_TO_PASS"])
    else:
        assert isinstance(spec, ScriptSpec)
        assert spec.path == "legacy_test.sh"
        assert "apt-get" not in private["legacy_test.sh"].decode()
    context = task.environment_requirements.docker_build
    assert context is not None and context == task.grader.environment.docker_build
    build = {resource.path: resource_bytes(resource) for resource in context.files}
    assert not any(path.startswith(("tests/", "solution/")) for path in build)
    assert b"apt-get" in build["taskcompendium-grader-setup.sh"]


@pytest.mark.parametrize("language", ["js", "ts"])
def test_source_language_exclusions_remain_explicit(language):
    source = original("go")
    config = json.loads(source.text("tests/config.json"))
    config["language"] = language
    source.files["tests/config.json"] = json.dumps(config).encode()
    result = convert_row(pipeline(), tasktrove_row(source.files))
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_variant"
    assert "golden sample" in result.detail


@pytest.mark.parametrize("problem", ["truncated_id", "unprotected_test"])
def test_python_source_contract_gaps_remain_explicit(problem):
    source = original("python")
    config = json.loads(source.text("tests/config.json"))
    config["FAIL_TO_PASS"] = [
        "tests/uncovered.py::test_value[unfinished" if problem == "truncated_id" else "tests/uncovered.py::test_value"
    ]
    source.files["tests/config.json"] = json.dumps(config).encode()
    result = convert_row(pipeline(), tasktrove_row(source.files))
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_variant"


@pytest.mark.parametrize(
    "log, resolved",
    [
        ("unrecognized output\n", False),
        ("--- FAIL: TestRepair (0.00s)\n", False),
        ("--- PASS: TestRepair (0.00s)\n", True),
    ],
)
def test_non_python_parser_requires_named_test_results(tmp_path, log, resolved):
    task = converted(original("go").files).task
    parser = next(resource_bytes(resource) for resource in task.resources.verifier if resource.path == "test_state.py")
    namespace = {}
    exec(compile(parser, "test_state.py", "exec"), namespace)
    # The source's old fallback read a fixed /logs path; model a successful command at that I/O boundary.
    namespace["_read_exit_code"] = lambda: 0
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"FAIL_TO_PASS": ["TestRepair"], "PASS_TO_PASS": [], "install_config": {"log_parser": "parse_log_gotest"}}
        )
    )
    output = tmp_path / "output.log"
    output.write_text(log)
    report = namespace["evaluate_test_results"](str(output), str(config))
    assert report["resolved"] is resolved


def git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()


@dataclass(frozen=True)
class PatchedRepository:
    workspace: Path
    private: Path
    spec: PytestSpec
    expected_tests: str


@pytest.fixture
def patched_repository(tmp_path) -> PatchedRepository:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    (workspace / "tests").mkdir()
    (workspace / "calc.py").write_text("def add(a, b): return a - b\n")
    trusted = "from calc import add\ndef test_existing(): assert add(1, 0) == 1\n"
    target = workspace / "tests/test_calc.py"
    target.write_text(trusted)
    git(workspace, "init", "-q")
    git(workspace, "add", ".")
    git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
    commit = git(workspace, "rev-parse", "HEAD")
    patched = trusted + "def test_hidden(): assert add(2, 3) == 5\n"
    target.write_text(patched)
    patch = git(workspace, "diff", "--", "tests/test_calc.py") + "\n"
    git(workspace, "checkout", "--", "tests/test_calc.py")
    source = original("python")
    source.files.update(
        {
            "instruction.md": (
                (
                    "## Environment Setup (complete these steps first)\n\n```bash\n"
                    f"cd /testbed\ngit checkout {commit}\n```\nFix add.\n"
                ).encode()
            ),
            "tests/config.json": (
                json.dumps(
                    {
                        "repo": "fixture/calc",
                        "language": "python",
                        "FAIL_TO_PASS": ["tests/test_calc.py::test_hidden"],
                        "PASS_TO_PASS": ["tests/test_calc.py::test_existing"],
                    }
                ).encode()
            ),
            "tests/test.sh": f"install_trusted_test_patch.sh /testbed /tests/test_patch.diff {commit}\n".encode(),
            "tests/test_patch.diff": patch.encode(),
            "tests/trusted_test_paths.txt": b"tests/test_calc.py\n",
            "tests/trusted_patch_paths.txt": b"tests/test_calc.py\n",
        }
    )
    task = converted(source.files).task
    private = tmp_path / "private"
    private.mkdir()
    for resource in task.resources.verifier:
        target = private / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
    spec = parse_spec((private / "taskcompendium-verifier.toml").read_text())
    assert isinstance(spec, PytestSpec)
    return PatchedRepository(
        workspace=workspace,
        private=private,
        spec=replace(spec, workspace=str(workspace), python=sys.executable, timeout=30),
        expected_tests=patched,
    )


@pytest.mark.parametrize(
    "repair, tamper, expected",
    [(False, None, 0), (True, None, 1), (False, "untracked_control", 0), (False, "index_flag", 0)],
)
def test_hidden_patch_restoration_preserves_product_edits(
    patched_repository: PatchedRepository, repair, tamper, expected
):
    workspace = patched_repository.workspace
    product = "def add(a, b): return a + b\n" if repair else "def add(a, b): return a - b\n"
    (workspace / "calc.py").write_text(product)
    if tamper == "untracked_control":
        (workspace / "conftest.py").write_text("def pytest_collection_modifyitems(items): items.clear()\n")
    elif tamper == "index_flag":
        (workspace / "tests/test_calc.py").write_text("def test_hidden(): pass\n")
        git(workspace, "update-index", "--assume-unchanged", "tests/test_calc.py")
    result = grade_pytest.grade(patched_repository.spec, patched_repository.private, workspace)
    assert result.reward == expected
    assert (workspace / "tests/test_calc.py").read_text() == patched_repository.expected_tests
    assert (workspace / "calc.py").read_text() == product
    assert not (workspace / "conftest.py").exists()
