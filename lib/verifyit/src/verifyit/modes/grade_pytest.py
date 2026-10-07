# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode pytest: run pytest with ``pytest-json-report`` and grade the node ids in the report.

This is the SWE-bench shape: ``must_pass`` is FAIL_TO_PASS (the bug the agent had to fix) and
``must_not_break`` is PASS_TO_PASS (what it must not regress). Node ids look like
``tests/test_x.py::test_y``. A missing pytest or json-report plugin is a defect in the task image,
not a failed attempt, so it raises instead of scoring zero.
"""

import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from verifyit.execution.command import run_command
from verifyit.file_ops.read import read_text
from verifyit.file_ops.restore import restore
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.run import STDERR_TAIL, check_ids, run_setup, workdir
from verifyit.spec import PytestSpec, TestIdMatching

REPORT_NAME = "report.json"
PASS_OUTCOMES = frozenset({"passed", "xpassed"})
FAIL_OUTCOMES = frozenset({"failed", "error"})


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
    size = spec.batch_size or max(1, len(spec.paths))
    batches = [spec.paths[start : start + size] for start in range(0, len(spec.paths), size)] or [()]
    outcomes: dict[str, bool] = {}
    reported_ids: set[str] = set()
    unexecuted_ids: set[str] = set()
    exit_code = 0
    output = ""
    with tempfile.TemporaryDirectory(prefix="tasktrove-pytest-") as scratch:
        for index, paths in enumerate(batches):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return scored(0.0, reason="timeout", passed=0, total=0)
            report_path = Path(scratch) / f"{index}-{REPORT_NAME}"
            argv = [
                spec.python,
                "-m",
                "pytest",
                "--json-report",
                f"--json-report-file={report_path}",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                *spec.args,
                *paths,
            ]
            result = run_command(argv, directory, remaining)
            if result.timed_out:
                return scored(0.0, reason="timeout", passed=0, total=0)
            if result.returncode not in (0, 1, 5):
                raise RuntimeError(
                    f"pytest producer failed before a usable json report (exit {result.returncode}): "
                    f"{_tail(result.stderr or result.stdout)}"
                )
            if not report_path.is_file():
                raise RuntimeError(
                    f"pytest wrote no json report (exit {result.returncode}): "
                    f"{_tail(result.stderr or result.stdout)}"
                )
            report = json.loads(read_text(report_path))
            if result.returncode == 5:
                return scored(0.0, reason="no_tests", passed=0, total=0, exit_code=5)
            if any(collector.get("outcome") == "failed" for collector in report.get("collectors", [])):
                raise RuntimeError("pytest report contains collection failures")
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
    for test_id in unexecuted_ids & outcomes.keys():
        outcomes[test_id] = False
    if matching is TestIdMatching.UNIQUE_PREFIX:
        outcomes = _match_partial_ids(outcomes, reported_ids, (*spec.must_pass, *spec.must_not_break))
    reward = check_ids(outcomes, spec.must_pass, spec.must_not_break, exit_code=exit_code)
    if reward.reward < 1.0:
        reward = replace(reward, detail={**reward.detail, "output": _tail(output, STDERR_TAIL)})
    return reward


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
