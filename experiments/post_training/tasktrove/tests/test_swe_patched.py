# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in SWE-rebench-V2 exemplar, plus the ``setup`` mechanics."""

import json
import subprocess
import sys
from pathlib import Path

from tasktrove_verify.modes import pytest_report
from tasktrove_verify.spec import PytestSpec, ScriptSpec, parse_spec

from experiments.post_training.tasktrove import verify
from experiments.post_training.tasktrove.contract import VERIFIER_TOML
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.converters.swe_patched import _setup_script
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import read_task_binary, write_task_binary

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"
SOURCE = "DCAgent__swe_rebench_v2_patched_oracle-v2"


def _fixture() -> bytes:
    return (FIXTURES / "swe_patched.tar.gz").read_bytes()


def _info(family: str = "swe-repo") -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, family, "")


def _convert(blob: bytes):
    return convert_one(_info(), "t.tar.gz", blob, converter_index(), TOOL_REF)


def test_exemplar_converts_to_pytest_spec_with_swe_tags():
    record = _convert(_fixture())
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "swe_patched" and record.mode == "pytest"
    assert set(record.tags) >= {"swe", "python"}
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, PytestSpec)
    assert spec.must_pass == (
        "tests/unit/test_linter.py::LinterFunctionsTestCase::test_read_migrations_no_file",
        "tests/unit/test_linter.py::LinterFunctionsTestCase::test_read_migrations_unknown_file",
    )
    assert spec.must_not_break == (
        "tests/unit/test_linter.py::LinterFunctionsTestCase::test_read_migrations_empty_file",
        "tests/unit/test_linter.py::LinterFunctionsTestCase::test_read_migrations_from_file",
    )
    assert spec.paths == ("tests/unit/test_linter.py",)
    assert spec.workspace == "/testbed"
    assert "799957a5564e8ca1ea20d7cf643abbc21db4e40f" in spec.setup


def test_old_grader_files_are_gone():
    record = _convert(_fixture())
    task = read_task_binary(record.task_binary)
    for old_file in (
        "tests/test_state.py",
        "tests/install_trusted_test_patch.sh",
        "tests/install_trusted_test_paths.sh",
    ):
        assert old_file not in task.files, f"{old_file} is old grader code and must not ship"


def test_spec_referenced_data_files_exist():
    record = _convert(_fixture())
    task = read_task_binary(record.task_binary)
    for data_file in ("tests/test_patch.diff", "tests/trusted_test_paths.txt", "tests/trusted_patch_paths.txt"):
        assert data_file in task.files


def test_verify_task_accepts_the_converted_binary():
    record = _convert(_fixture())
    assert verify.verify_task(record.task_binary) is None


def _non_python_fixture(language: str = "go") -> bytes:
    task = read_task_binary(_fixture())
    config = json.loads(task.text("tests/config.json"))
    config["language"] = language
    task.files["tests/config.json"] = json.dumps(config).encode()
    return write_task_binary(task)


def test_non_python_language_uses_fail_closed_script_fallback():
    record = _convert(_non_python_fixture())
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "script" and set(record.tags) >= {"go", "script-fallback"}

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ScriptSpec)
    assert spec.path == "legacy_test.sh" and spec.workspace == "/testbed"
    assert "tests/legacy_test.sh" in task.files
    assert "tests/test_state.py" in task.files
    assert "tests/config.json" in task.files

    legacy_test = task.text("tests/legacy_test.sh")
    assert "/opt/tasktrove-legacy-grader/bin/python -m pytest" in legacy_test
    assert '"$TASKTROVE_LOGS_DIR/reward.txt"' in legacy_test
    assert "/logs/verifier/test_output.log" in legacy_test
    assert "uv init --python 3.12" not in legacy_test

    test_state = task.text("tests/test_state.py")
    assert "if not resolved and not statuses:" not in test_state
    assert "fallback_exit_code" not in test_state
    assert "def _read_exit_code" not in test_state
    assert 'evaluate_test_results("/logs/verifier/test_output.log")' in test_state
    assert "pytest-json-ctrf==0.3.5" in task.text("environment/Dockerfile")


def test_non_python_unknown_grader_shape_is_rejected():
    task = read_task_binary(_non_python_fixture())
    old = b"uv init --python 3.12"
    new = b"uv init --python 3.11"
    task.files["tests/test.sh"] = task.files["tests/test.sh"].replace(old, new)
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_non_python_missing_grader_file_is_rejected():
    task = read_task_binary(_non_python_fixture())
    del task.files["tests/test_state.py"]
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_language_task_sets_with_unreliable_goldens_are_rejected():
    for language in ("js", "ts"):
        record = _convert(_non_python_fixture(language))
        assert record.status == ConvertStatus.UNSUPPORTED_VARIANT
        assert record.task_binary is None and "golden sample" in record.error


def test_empty_fail_to_pass_is_rejected():
    task = read_task_binary(_fixture())
    config = json.loads(task.text("tests/config.json"))
    config["FAIL_TO_PASS"] = []
    task.files["tests/config.json"] = json.dumps(config).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.TOO_FEW_CASES and record.task_binary is None


def test_truncated_parametrized_ids_are_dropped_or_rejected():
    """meltano's PASS_TO_PASS carried ``test_get_column_ddl[...-ALTER`` (the id was split on the space
    inside its parameter); such an id can never be collected, so it leaves ``must_not_break`` and
    makes the task unconvertible when it is a FAIL_TO_PASS id."""
    truncated = "tests/test_calc.py::test_ddl[get_column_add_ddl-kwargs0-context0-ALTER"
    task = read_task_binary(_fixture())
    config = json.loads(task.text("tests/config.json"))
    config["PASS_TO_PASS"] = [*config["PASS_TO_PASS"], truncated]
    task.files["tests/config.json"] = json.dumps(config).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, PytestSpec) and truncated not in spec.must_not_break

    config["FAIL_TO_PASS"] = [truncated]
    task.files["tests/config.json"] = json.dumps(config).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_top_level_setup_files_script_is_rejected():
    """The OpenSWE template's ``instruction.md`` tells the agent to run a root-level
    ``/setup_files/setup.sh`` that the pytest mode's sandbox only ever mounts ``tests/`` and
    ``solution/`` for, so that shape is unconvertible rather than silently broken."""
    task = read_task_binary(_fixture())
    task.files["setup_files/setup.sh"] = b"#!/bin/bash\necho hi\n"
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


# --- setup-script mechanics: restore trusted paths from the agent's own clone, apply the hidden
# test patch, and score with the real pytest_report grader. The exemplar's repo lives on GitHub, so
# these use a throwaway local repo instead of a network clone. ---

_BASE_TEST_FILE = """import unittest
from calc import Calc


class CalcTestCase(unittest.TestCase):
    def test_add_existing(self):
        self.assertEqual(Calc.add(2, 3), 5)
"""

_PATCHED_TEST_FILE = """import unittest
from calc import Calc


class CalcTestCase(unittest.TestCase):
    def test_add_existing(self):
        self.assertEqual(Calc.add(2, 3), 5)

    def test_add_new(self):
        self.assertEqual(Calc.add(4, 4), 8)
"""

_BUGGY_CALC = "class Calc:\n    @staticmethod\n    def add(a, b):\n        return a - b\n"
_FIXED_CALC = "class Calc:\n    @staticmethod\n    def add(a, b):\n        return a + b\n"

_MUST_PASS = ("tests/test_calc.py::CalcTestCase::test_add_new",)
_MUST_NOT_BREAK = ("tests/test_calc.py::CalcTestCase::test_add_existing",)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _repo_and_patch(tmp_path: Path) -> tuple[Path, Path, str]:
    """A local git repo at the trusted commit, the hidden-test diff on top of it, and that sha."""
    workspace = tmp_path / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "calc.py").write_text(_BUGGY_CALC)
    (workspace / "tests" / "test_calc.py").write_text(_BASE_TEST_FILE)
    _git("init", "-q", cwd=workspace)
    _git("config", "user.email", "a@b.c", cwd=workspace)
    _git("config", "user.name", "a", cwd=workspace)
    _git("add", "-A", cwd=workspace)
    _git("commit", "-q", "-m", "base", cwd=workspace)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()

    (workspace / "tests" / "test_calc.py").write_text(_PATCHED_TEST_FILE)
    diff = subprocess.run(
        ["git", "diff", "--", "tests/test_calc.py"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout
    _git("checkout", "--", "tests/test_calc.py", cwd=workspace)  # back to the trusted, pre-patch state

    tests_dir = tmp_path / "tests_dir"
    tests_dir.mkdir()
    (tests_dir / "test_patch.diff").write_text(diff)
    (tests_dir / "trusted_test_paths.txt").write_text("")
    (tests_dir / "trusted_patch_paths.txt").write_text("tests/test_calc.py\n")
    return workspace, tests_dir, base_sha


def _spec(base_sha: str) -> PytestSpec:
    return PytestSpec(
        paths=("tests/test_calc.py",),
        must_pass=_MUST_PASS,
        must_not_break=_MUST_NOT_BREAK,
        setup=_setup_script(base_sha),
        python=sys.executable,
        timeout=60.0,
    )


def test_setup_restores_and_patches_then_buggy_product_code_scores_zero(tmp_path):
    workspace, tests_dir, base_sha = _repo_and_patch(tmp_path)
    reward = pytest_report.grade(_spec(base_sha), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == _MUST_PASS[0]


def test_setup_restores_and_patches_then_fixed_product_code_scores_one(tmp_path):
    workspace, tests_dir, base_sha = _repo_and_patch(tmp_path)
    (workspace / "calc.py").write_text(_FIXED_CALC)
    reward = pytest_report.grade(_spec(base_sha), tests_dir, workspace)
    assert reward.reward == 1.0


def test_setup_fails_closed_on_an_empty_workspace(tmp_path):
    _workspace, tests_dir, base_sha = _repo_and_patch(tmp_path)
    empty_workspace = tmp_path / "empty"
    empty_workspace.mkdir()
    reward = pytest_report.grade(_spec(base_sha), tests_dir, empty_workspace)
    assert reward.reward == 0.0 and reward.detail["reason"] == "setup_failed"


def test_setup_discards_agent_tampering_with_the_hidden_test(tmp_path):
    workspace, tests_dir, base_sha = _repo_and_patch(tmp_path)
    (workspace / "calc.py").write_text(_FIXED_CALC)
    (workspace / "tests" / "test_calc.py").write_text(
        "import unittest\n\n\nclass CalcTestCase(unittest.TestCase):\n    def test_add_new(self):\n        pass\n"
    )
    reward = pytest_report.grade(_spec(base_sha), tests_dir, workspace)
    assert reward.reward == 1.0
    assert (workspace / "tests" / "test_calc.py").read_text() == _PATCHED_TEST_FILE
