# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode pytest: run pytest with ``pytest-json-report`` and grade the node ids in the report.

This is the SWE-bench shape: ``must_pass`` is FAIL_TO_PASS (the bug the agent had to fix) and
``must_not_break`` is PASS_TO_PASS (what it must not regress). Node ids look like
``tests/test_x.py::test_y``. A missing pytest or json-report plugin is a defect in the task image,
not a failed attempt, so it raises instead of scoring zero.
"""

import ast
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from harbor_config.errors import ErrorCategory

from verifyit.execution.command import run_command
from verifyit.file_ops.read import read_text
from verifyit.file_ops.restore import restore
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.run import STDERR_TAIL, check_ids, run_setup, workdir
from verifyit.spec import PytestSpec, TestIdMatching

REPORT_NAME = "report.json"
PASS_OUTCOMES = frozenset({"passed", "xpassed"})
FAIL_OUTCOMES = frozenset({"failed", "error"})


class _FailureKind(StrEnum):
    ERROR = "error"
    DEPENDENCY = "dependency"
    INTERRUPT = "interrupt"


@dataclass(frozen=True)
class _PytestFailure:
    kind: _FailureKind
    filename: str | None
    nodeid: str | None
    module: str = ""


# The recorder uses builtins so candidate modules can shadow stdlib dependencies.
PYTEST_RUNNER = """import sys

def record(error, nodeid=None):
    kind = "error"
    if isinstance(error, SyntaxError):
        filename = error.filename
    else:
        if isinstance(error, ModuleNotFoundError):
            kind = "dependency"
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            kind = "interrupt"
        traceback = error.__traceback__
        while traceback and traceback.tb_next:
            traceback = traceback.tb_next
        filename = traceback.tb_frame.f_code.co_filename if traceback else ""
        if isinstance(error, ImportError) and error.path:
            filename = error.path
    failure = {"kind": kind, "filename": filename, "nodeid": nodeid}
    if isinstance(error, AttributeError) and isinstance(error.obj, type(sys)):
        failure["module"] = object.__getattribute__(error.obj, "__dict__").get("__file__", "")
    with open(sys.argv[1], "a") as output:
        output.write(repr(failure) + "\\n")

try:
    import pytest
except BaseException as error:
    record(error)
    raise
from _pytest.config import ConftestImportFailure, UsageError
from _pytest.main import Interrupted
from _pytest.nodes import Collector

class FailureCapture:
    def pytest_keyboard_interrupt(self, excinfo):
        if not isinstance(excinfo.value, Interrupted):
            record(excinfo.value)

    def pytest_exception_interact(self, call, report):
        if call.when == "collect":
            error = call.excinfo.value
            if isinstance(error, Collector.CollectError):
                error = error.__cause__ or error.__context__ or error
            record(error, report.nodeid)

    # Record before older pluggy stops unwinding on pytest's help-hook reraise.
    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_cmdline_parse(self):
        outcome = yield
        if outcome.excinfo:
            error = outcome.excinfo[1]
            if isinstance(error, (ConftestImportFailure, UsageError)):
                error = error.__cause__ or error.__context__ or error
            record(error)

try:
    exit_code = pytest.main(sys.argv[2:], plugins=[FailureCapture()])
except BaseException as error:
    record(error)
    raise
raise SystemExit(exit_code)
"""


def grade(spec: PytestSpec, tests_dir: Path, workspace: Path) -> Reward:
    try:
        valid_timeout = (
            not isinstance(spec.timeout, bool)
            and isinstance(spec.timeout, (int, float))
            and math.isfinite(spec.timeout)
            and spec.timeout > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise InvalidTask("pytest timeout must be finite and positive")
    if type(spec.batch_size) is not int or spec.batch_size < 0:
        raise InvalidTask("pytest batch_size must be a nonnegative integer")
    try:
        matching = TestIdMatching(spec.id_matching)
    except ValueError as error:
        raise InvalidTask("unsupported pytest id_matching") from error
    deadline = time.monotonic() + spec.timeout
    directory = workdir(spec, workspace)
    restore(spec.restore, tests_dir, directory)
    if spec.setup:
        setup = run_setup(spec.setup, tests_dir, directory, max(0.0, deadline - time.monotonic()))
        if setup.timed_out or setup.returncode != 0:
            if spec.setup_failure_is_infra:
                reason = "timed out" if setup.timed_out else f"exited {setup.returncode}"
                raise RuntimeError(f"pytest setup {reason}: {_tail(setup.stderr or setup.stdout, STDERR_TAIL)}")
            return scored(0.0, reason="setup_failed", stderr=setup.stderr[-STDERR_TAIL:], passed=0, total=0)
    protected = _protected_paths(spec, tests_dir, directory)
    size = spec.batch_size or max(1, len(spec.paths))
    batches = [spec.paths[start : start + size] for start in range(0, len(spec.paths), size)] or [()]
    outcomes: dict[str, bool] = {}
    reported_ids: set[str] = set()
    unexecuted_ids: set[str] = set()
    exit_code = 0
    output = ""
    candidate_failure: Reward | None = None
    with tempfile.TemporaryDirectory(prefix="tasktrove-pytest-") as scratch:
        for index, paths in enumerate(batches):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return scored(0.0, reason="timeout", passed=0, total=0)
            report_path = Path(scratch) / f"{index}-{REPORT_NAME}"
            failures_path = Path(scratch) / f"{index}-failures.txt"
            argv = [spec.python, "-c", PYTEST_RUNNER, str(failures_path), *_pytest_args(report_path), *spec.args, *paths]
            result = run_command(argv, directory, remaining)
            if result.timed_out:
                return scored(0.0, reason="timeout", passed=0, total=0)
            failures = []
            if failures_path.is_file():
                for line in read_text(failures_path).splitlines():
                    record = ast.literal_eval(line)
                    failures.append(
                        _PytestFailure(
                            _FailureKind(record["kind"]), record["filename"], record["nodeid"], record.get("module", "")
                        )
                    )
            if any(failure.kind is _FailureKind.INTERRUPT for failure in failures) or result.returncode < 0:
                raise RuntimeError("pytest producer was interrupted")
            if result.returncode not in (0, 1, 2, 5) or not report_path.is_file():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return scored(0.0, reason="timeout", passed=0, total=0)
                files = _candidate_error_files(failures, directory, protected)
                if files and _pytest_available(spec.python, Path(scratch), remaining):
                    candidate_failure = scored(
                        0.0,
                        reason="startup_error",
                        category=ErrorCategory.AGENT,
                        exit_code=result.returncode,
                        output=_tail(result.stderr or result.stdout),
                        files=files,
                        passed=0,
                        total=0,
                    )
                    continue
                raise RuntimeError(
                    f"pytest wrote no json report (exit {result.returncode}): "
                    f"{_tail(result.stderr or result.stdout)}"
                )
            report = json.loads(read_text(report_path))
            failed_collectors = [
                collector for collector in report.get("collectors", []) if collector.get("outcome") == "failed"
            ]
            if failed_collectors:
                collection_failures = [failure for failure in failures if failure.nodeid is not None]
                if {failure.nodeid for failure in collection_failures} != {
                    collector["nodeid"] for collector in failed_collectors
                }:
                    raise RuntimeError("pytest collection failure has no exception provenance")
                files = _candidate_error_files(collection_failures, directory, protected)
                if not files:
                    raise RuntimeError(
                        "pytest collection failed in trusted code or dependencies: "
                        f"{_tail(result.stderr or result.stdout)}"
                    )
                candidate_failure = scored(
                    0.0,
                    reason="collection_error",
                    category=ErrorCategory.AGENT,
                    exit_code=result.returncode,
                    output=_tail(result.stderr or result.stdout),
                    files=files,
                )
                continue
            if result.returncode == 2:
                raise RuntimeError("pytest interrupted without a reported collection failure")
            if result.returncode == 5:
                candidate_failure = candidate_failure or scored(0.0, reason="no_tests", passed=0, total=0, exit_code=5)
                continue
            _validate_summary(report)
            root = Path(report.get("root", directory))
            reported_ids.update(_rebase(test["nodeid"], root, directory) for test in report.get("tests", []))
            unexecuted_ids.update(
                _rebase(test["nodeid"], root, directory)
                for test in report.get("tests", [])
                if test.get("outcome") in {"skipped", "xfailed"}
            )
            batch_outcomes = _outcomes(report, directory)
            if result.returncode == 1 and batch_outcomes and all(batch_outcomes.values()):
                raise RuntimeError("pytest producer failed without reporting a failing test")
            for test_id, passed in batch_outcomes.items():
                outcomes[test_id] = outcomes.get(test_id, True) and passed
            exit_code = max(exit_code, result.returncode)
            output = (output + result.stdout + result.stderr)[-STDERR_TAIL:]
    if candidate_failure is not None:
        return candidate_failure
    for test_id in unexecuted_ids & outcomes.keys():
        outcomes[test_id] = False
    if matching is TestIdMatching.UNIQUE_PREFIX:
        outcomes = _match_partial_ids(outcomes, reported_ids, (*spec.must_pass, *spec.must_not_break))
    reward = check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=exit_code)
    if reward.reward < 1.0:
        reward = replace(reward, detail={**reward.detail, "output": _tail(output, STDERR_TAIL)})
    return reward


def _pytest_args(report: Path) -> list[str]:
    return [
        "--json-report",
        f"--json-report-file={report}",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
    ]


def _pytest_available(python: str, directory: Path, timeout: float) -> bool:
    # Candidate modules can shadow pytest's dependencies before collection.
    # Probe the installed interpreter/plugin without importing any task code.
    test = directory / "test_availability.py"
    test.write_text("def test_available():\n    assert True\n")
    config = directory / "pytest.ini"
    config.write_text("[pytest]\n")
    report = directory / "availability.json"
    result = run_command(
        [python, "-I", "-m", "pytest", *_pytest_args(report), "-c", str(config), str(test)],
        directory,
        timeout,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "", "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": ""},
    )
    return not result.timed_out and result.returncode == 0 and report.is_file()


def _protected_paths(spec: PytestSpec, tests_dir: Path, workspace: Path) -> set[Path]:
    root = workspace.resolve()
    protected = set()
    for entry in spec.restore:
        source = tests_dir / entry
        files = (source,) if source.is_file() else (path for path in source.rglob("*") if path.is_file())
        protected.update((root / path.relative_to(tests_dir)).resolve() for path in files)
    for manifest in spec.protected_paths_files:
        for entry in read_text(tests_dir / manifest).splitlines():
            if not entry.strip():
                continue
            path = (root / entry).resolve()
            if not path.is_relative_to(root):
                raise InvalidTask("pytest protected paths must stay inside the workspace")
            files = (candidate for candidate in path.rglob("*") if candidate.is_file()) if path.is_dir() else (path,)
            protected.update(candidate.resolve() for candidate in files)
    return protected


def _candidate_error_files(failures: list[_PytestFailure], workspace: Path, protected: set[Path]) -> list[str]:
    """Return files only when every failure comes from editable source; otherwise return []."""
    root = workspace.resolve()
    files = set()
    for failure in failures:
        if failure.kind in {_FailureKind.DEPENDENCY, _FailureKind.INTERRUPT}:
            return []
        filename = failure.filename
        path = (root / filename).resolve() if filename else root
        if failure.module:
            module = (root / failure.module).resolve()
            if module.is_relative_to(root):
                path = module
        if not path.is_relative_to(root) or not path.is_file() or path in protected:
            return []
        files.add(str(path.relative_to(root)))
    return sorted(files)


def _match_partial_ids(outcomes: dict[str, bool], reported_ids: set[str], required: tuple[str, ...]) -> dict[str, bool]:
    """Resolve exact escaped Unicode or one bracket-truncated identity, never a function prefix."""
    matched = dict(outcomes)
    claims: dict[str, set[str]] = {}
    for reference in required:
        escaped = "".join(
            char.encode("unicode_escape").decode("ascii") if ord(char) > 127 else char for char in reference
        )
        target = None
        if reference in reported_ids:
            if escaped != reference and escaped in reported_ids:
                matched[reference] = False
                continue
            target = reference
        elif escaped in reported_ids:
            target = escaped
        elif "[" in reference and not reference.endswith("]"):
            candidates = [test_id for test_id in reported_ids if test_id.startswith(reference)]
            if len(candidates) == 1:
                target = candidates[0]
        if target is not None:
            claims.setdefault(target, set()).add(reference)
            matched[reference] = outcomes.get(target, False)
    for references in claims.values():
        if len(references) > 1:
            for reference in references:
                matched[reference] = False
    return matched


def _outcomes(report: dict, workspace: Path) -> dict[str, bool]:
    """Node id to pass/fail. Skipped and xfailed tests are neither and stay out of the map.

    pytest writes node ids relative to its rootdir, which a ``tests/pytest.ini`` moves below the
    workspace (``unit/test_x.py::test_y`` for ``tests/unit/test_x.py``); the spec's ids are relative
    to the workspace, so the ids are rebased before they are compared.
    """
    root = Path(report.get("root", workspace))
    outcomes = {}
    for test in report.get("tests", []):
        outcome = test.get("outcome")
        if outcome in PASS_OUTCOMES:
            test_id = _rebase(test["nodeid"], root, workspace)
            outcomes[test_id] = outcomes.get(test_id, True)
        elif outcome in FAIL_OUTCOMES:
            outcomes[_rebase(test["nodeid"], root, workspace)] = False
    return outcomes


def _rebase(nodeid: str, root: Path, workspace: Path) -> str:
    file, separator, rest = nodeid.partition("::")
    return os.path.relpath(root / file, workspace) + separator + rest


def _tail(text: str, limit: int = 500) -> str:
    return text.strip()[-limit:]


def _validate_summary(report: dict) -> None:
    tests = report.get("tests", [])
    counts = Counter(test.get("outcome") for test in tests)
    allowed = PASS_OUTCOMES | FAIL_OUTCOMES | {"skipped", "xfailed"}
    if set(counts) - allowed:
        raise RuntimeError("pytest report contains unsupported test outcomes")
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        raise RuntimeError("pytest report summary must be an object")
    expected = {"total": len(tests), **{name: counts[name] for name in allowed}}
    for field, observed in expected.items():
        if field not in summary:
            continue
        declared = summary[field]
        if isinstance(declared, bool) or not isinstance(declared, int) or declared != observed:
            raise RuntimeError(f"incomplete pytest report: declared {field}={declared}, observed {observed}")
