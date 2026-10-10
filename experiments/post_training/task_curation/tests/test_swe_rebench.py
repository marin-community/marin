# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Patched repository graders preserve private tests and reject unresolved source results."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from taskcompendium.models import DockerBuildContext
from taskcompendium.pipeline.models import ImportRejection
from taskcompendium.runtime.resources import resource_bytes
from verifyit.modes import grade_pytest
from verifyit.spec import PytestSpec, parse_spec

from experiments.post_training.task_curation.datasets.tasktrove import repositories
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task, tasktrove_row

RECIPES = {source.name: cast(CurationRecipe, source.config) for source in repositories.sources()}

FIXTURES = Path(__file__).parent / "fixtures"


def python_source(commit: str) -> dict[str, bytes]:
    return {
        "instruction.md": (
            (
                "## Environment Setup (complete these steps first)\n\n```bash\n"
                f"cd /testbed\ngit checkout {commit}\n```\nFix add.\n"
            ).encode()
        ),
        "environment/Dockerfile": b"FROM python:3.12-slim\nRUN mkdir -p /testbed\n",
        "tests/config.json": (
            json.dumps(
                {
                    "repo": "fixture/calc",
                    "language": "python",
                    "FAIL_TO_PASS": ["tests/test_calc.py::test_hidden"],
                    "PASS_TO_PASS": [],
                }
            ).encode()
        ),
        "tests/test.sh": f"install_trusted_test_patch.sh /testbed /tests/test_patch.diff {commit}\n".encode(),
        "tests/test_patch.diff": b"private test patch",
        "tests/trusted_test_paths.txt": b"tests/test_calc.py\n",
        "tests/trusted_patch_paths.txt": b"tests/test_calc.py\n",
        "solution/solve.sh": b"private oracle",
    }


def go_source() -> dict[str, bytes]:
    # Preserve the real source parser because conversion removes its exact fail-open block.
    # Extracted from the pinned SWE-rebench Go task; unrelated task files are constructed here.
    return {
        **python_source("abcdef0"),
        "tests/config.json": (
            json.dumps(
                {
                    "repo": "fixture/calc",
                    "language": "go",
                    "FAIL_TO_PASS": ["TestRepair"],
                    "PASS_TO_PASS": [],
                    "install_config": {"log_parser": "parse_log_gotest"},
                }
            ).encode()
        ),
        "tests/test_state.py": (FIXTURES / "swe_rebench_parser.txt").read_bytes(),
        "tests/test.sh": (
            b"cd /tests\n"
            b"uv init --python 3.12 --no-progress >/dev/null 2>&1 || true\n"
            b"uv add --no-progress pytest==8.4.1 pytest-json-ctrf==0.3.5 >/dev/null 2>&1\n"
            b"uv run --no-progress pytest --ctrf /logs/verifier/ctrf.json test_state.py -rA\n"
        ),
        "tests/install_trusted_test_paths.sh": b"private trusted paths installer",
        "tests/install_trusted_test_patch.sh": b"private trusted patch installer",
    }


def elixir_source() -> dict[str, bytes]:
    files = go_source()
    config = json.loads(files["tests/config.json"])
    config.update(
        language="elixir",
        FAIL_TO_PASS=["issue #63"],
        PASS_TO_PASS=["handles arbitrary properties"],
        install_config={"log_parser": "parse_log_elixir"},
    )
    files["tests/config.json"] = json.dumps(config).encode()
    return files


@pytest.mark.parametrize("language", ["python", "go"])
def test_source_contract_keeps_private_graders_and_deferred_dependencies(language):
    files = python_source("abcdef0") if language == "python" else go_source()
    task = converted_task(
        RECIPES["tasktrove-swe_rebench"],
        tasktrove_row(files),
    )
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    public = {resource.path for resource in task.resources.worker}
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    for path in (
        ("test_patch.diff",)
        if language == "python"
        else ("install_trusted_test_paths.sh", "install_trusted_test_patch.sh", "test_patch.diff")
    ):
        assert private[path] == files[f"tests/{path}"]
        assert path not in public
    assert oracle["source_archive/tests/test.sh"] == files["tests/test.sh"]
    assert oracle["solution/solve.sh"] == files["solution/solve.sh"]
    assert "verifier.toml" not in private
    context = cast(DockerBuildContext, task.environment_requirements.docker_build)
    build = {resource.path: resource_bytes(resource) for resource in context.files}
    assert not any(path.startswith(("tests/", "solution/")) for path in build)
    assert "taskcompendium-grader-setup.sh" not in build
    assert "taskcompendium-repository-setup.sh" not in build


@pytest.mark.parametrize(
    "language, node_id",
    [
        ("js", "tests/test_calc.py::test_hidden"),
        ("ts", "tests/test_calc.py::test_hidden"),
        ("python", "tests/uncovered.py::test_value[unfinished"),
        ("python", "tests/uncovered.py::test_value"),
    ],
)
def test_unsupported_language_or_test_contract_is_rejected(language, node_id):
    files = python_source("abcdef0")
    config = json.loads(files["tests/config.json"])
    config.update(language=language, FAIL_TO_PASS=[node_id])
    files["tests/config.json"] = json.dumps(config).encode()
    result = convert_row(
        RECIPES["tasktrove-swe_rebench"],
        tasktrove_row(files),
    )
    assert cast(ImportRejection, result).reason == "unsupported_variant"


@pytest.mark.parametrize(
    "log, resolved",
    [
        ("unrecognized output\n", False),
        ("--- FAIL: TestRepair (0.00s)\n", False),
        ("--- PASS: TestRepair (0.00s)\n", True),
    ],
)
def test_non_python_parser_requires_named_test_results(tmp_path, log, resolved):
    task = converted_task(
        RECIPES["tasktrove-swe_rebench"],
        tasktrove_row(go_source()),
    )
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


@pytest.mark.parametrize(
    "log, resolved",
    [
        (
            "  * test issue #63 (0.1ms) [L#217]\n" "  * test handles arbitrary properties (0.1ms) [L#528]\n",
            True,
        ),
        (
            "  1) test issue #63 (Tailwind.FormatterTest)\n" "  * test handles arbitrary properties (0.1ms) [L#528]\n",
            False,
        ),
    ],
)
def test_elixir_trace_grades_named_tests(tmp_path, log, resolved):
    files = elixir_source()
    task = converted_task(all_pipelines()["tasktrove-swe_rebench"], tasktrove_row(files))
    parser = next(resource_bytes(resource) for resource in task.resources.verifier if resource.path == "test_state.py")
    namespace = {}
    exec(compile(parser, "test_state.py", "exec"), namespace)
    config = tmp_path / "config.json"
    config.write_bytes(files["tests/config.json"])
    output = tmp_path / "output.log"
    output.write_text(log)
    report = namespace["evaluate_test_results"](str(output), str(config))
    assert report["resolved"] is resolved
    assert report["PASS_TO_PASS"]["success"] == ["handles arbitrary properties"]
    assert report["FAIL_TO_PASS"]["success"] == (["issue #63"] if resolved else [])


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
    files = python_source(commit)
    config = json.loads(files["tests/config.json"])
    config["PASS_TO_PASS"] = ["tests/test_calc.py::test_existing"]
    files["tests/config.json"] = json.dumps(config).encode()
    files["tests/test_patch.diff"] = patch.encode()
    task = converted_task(
        RECIPES["tasktrove-swe_rebench"],
        tasktrove_row(files),
    )
    private = tmp_path / "private"
    private.mkdir()
    for resource in task.resources.verifier:
        target = private / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
    spec = cast(PytestSpec, parse_spec((private / "taskcompendium-verifier.toml").read_text()))
    return PatchedRepository(
        workspace=workspace,
        private=private,
        spec=replace(spec, workspace=str(workspace), python=sys.executable, timeout=30),
        expected_tests=patched,
    )


@pytest.mark.parametrize(
    "repair, tamper, expected",
    [(False, None, 0), (True, None, 1), (False, "manifest_test", 0)],
)
def test_hidden_patch_restoration_preserves_product_edits(
    patched_repository: PatchedRepository, repair, tamper, expected
):
    workspace = patched_repository.workspace
    product = "def add(a, b): return a + b\n" if repair else "def add(a, b): return a - b\n"
    (workspace / "calc.py").write_text(product)
    if tamper == "manifest_test":
        (workspace / "tests/test_calc.py").write_text("def test_hidden(): pass\n")
    result = grade_pytest.grade(patched_repository.spec, patched_repository.private, workspace)
    assert result.reward == expected
    assert (workspace / "tests/test_calc.py").read_text() == patched_repository.expected_tests
    assert (workspace / "calc.py").read_text() == product
