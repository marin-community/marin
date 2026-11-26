# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from verifyit.grade import Status, main, run
from verifyit.modes import grade_gotest
from verifyit.spec import GotestSpec

# A real `go test -json ./...` stream: one package with a passing test, a failing test with a
# subtest, and a skipped test.
EVENT_STREAM = """{"Time":"2026-01-01T00:00:00Z","Action":"start","Package":"example.com/m/calc"}
{"Time":"2026-01-01T00:00:00Z","Action":"run","Package":"example.com/m/calc","Test":"TestAdd"}
{"Action":"output","Package":"example.com/m/calc","Test":"TestAdd","Output":"=== RUN   TestAdd\\n"}
{"Time":"2026-01-01T00:00:00Z","Action":"pass","Package":"example.com/m/calc","Test":"TestAdd","Elapsed":0}
{"Time":"2026-01-01T00:00:00Z","Action":"run","Package":"example.com/m/calc","Test":"TestSub"}
{"Time":"2026-01-01T00:00:00Z","Action":"run","Package":"example.com/m/calc","Test":"TestSub/negative"}
{"Action":"output","Package":"example.com/m/calc","Test":"TestSub/negative","Output":"got 5 want -5\\n"}
{"Time":"2026-01-01T00:00:00Z","Action":"fail","Package":"example.com/m/calc","Test":"TestSub/negative","Elapsed":0}
{"Time":"2026-01-01T00:00:00Z","Action":"fail","Package":"example.com/m/calc","Test":"TestSub","Elapsed":0}
{"Time":"2026-01-01T00:00:00Z","Action":"run","Package":"example.com/m/calc","Test":"TestPending"}
{"Time":"2026-01-01T00:00:00Z","Action":"skip","Package":"example.com/m/calc","Test":"TestPending","Elapsed":0}
{"Time":"2026-01-01T00:00:00Z","Action":"fail","Package":"example.com/m/calc","Elapsed":0.2}
"""

BUILD_FAILURE_STREAM = """{"Time":"2026-01-01T00:00:00Z","Action":"start","Package":"example.com/m/calc"}
{"Action":"output","Package":"example.com/m/calc","Output":"calc.go:7:2: undefined: helper\\n"}
{"Time":"2026-01-01T00:00:00Z","Action":"fail","Package":"example.com/m/calc","Elapsed":0}
"""

ADD = "example.com/m/calc.TestAdd"
SUB_NEGATIVE = "example.com/m/calc.TestSub/negative"

GO_MOD = "module example.com/m\n\ngo 1.21\n"
CALC_GO = "package calc\n\nfunc Add(a, b int) int { return a + b }\n"
CALC_TEST_GO = """package calc

import "testing"

func TestAdd(t *testing.T) {
	if Add(2, 3) != 5 {
		t.Fatal("bad sum")
	}
}
"""


def _fake_go(tmp_path: Path, stream: str, exit_code: int = 1) -> Path:
    """A stand-in for the go toolchain that replays a captured event stream."""
    directory = tmp_path / "bin"
    directory.mkdir()
    script = directory / "go"
    stream_path = tmp_path / "stream.jsonl"
    stream_path.write_text(stream)
    script.write_text(f"#!/bin/sh\ncat {stream_path}\nexit {exit_code}\n")
    script.chmod(0o755)
    return directory


def _use_fake_go(monkeypatch, tmp_path: Path, stream: str, exit_code: int = 1) -> Path:
    monkeypatch.setenv("PATH", f"{_fake_go(tmp_path, stream, exit_code)}:{os.environ['PATH']}")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_gotest_required_test_passing_scores_one(tmp_path, monkeypatch):
    workspace = _use_fake_go(monkeypatch, tmp_path, EVENT_STREAM)
    reward = grade_gotest.grade(GotestSpec(must_pass=(ADD,)), tmp_path, workspace)
    assert reward.reward == 1.0
    assert reward.detail["passed"] == 1


def test_gotest_failing_subtest_fails_the_run(tmp_path, monkeypatch):
    workspace = _use_fake_go(monkeypatch, tmp_path, EVENT_STREAM)
    reward = grade_gotest.grade(GotestSpec(must_pass=(ADD,), must_not_break=(SUB_NEGATIVE,)), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == SUB_NEGATIVE


def test_gotest_without_id_lists_requires_the_whole_package_to_pass(tmp_path, monkeypatch):
    workspace = _use_fake_go(monkeypatch, tmp_path, EVENT_STREAM)
    reward = grade_gotest.grade(GotestSpec(), tmp_path, workspace)
    assert reward.reward == 0.0
    # TestAdd passed, TestSub and TestSub/negative failed, TestPending was skipped and is not counted.
    assert (reward.detail["passed"], reward.detail["total"]) == (1, 3)


def test_gotest_build_failure_reports_no_tests(tmp_path, monkeypatch):
    workspace = _use_fake_go(monkeypatch, tmp_path, BUILD_FAILURE_STREAM)
    reward = grade_gotest.grade(GotestSpec(), tmp_path, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "no_tests"


def test_gotest_missing_test_id_counts_as_failed(tmp_path, monkeypatch):
    workspace = _use_fake_go(monkeypatch, tmp_path, EVENT_STREAM)
    missing = "example.com/m/calc.TestNeverWritten"
    reward = grade_gotest.grade(GotestSpec(must_pass=(missing,)), tmp_path, workspace)
    assert reward.detail["first_failure"] == missing


@pytest.mark.skipif(shutil.which("go") is None, reason="the go toolchain is not installed")
def test_gotest_real_go_module_passes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "go.mod").write_text(GO_MOD)
    (workspace / "calc.go").write_text(CALC_GO)
    (workspace / "calc_test.go").write_text(CALC_TEST_GO)
    reward = grade_gotest.grade(GotestSpec(must_pass=("example.com/m.TestAdd",)), tmp_path, workspace)
    assert reward.reward == 1.0


@pytest.mark.parametrize("actions", [("fail", "pass"), ("pass", "fail"), ("fail", "skip", "pass")])
def test_gotest_repeated_test_failure_is_not_erased(tmp_path, monkeypatch, actions):
    stream = (
        "\n".join(f'{{"Action": "{action}", "Package": "one", "Test": "TestRepeated"}}' for action in actions)
        + '\n{"Action": "pass", "Package": "two", "Test": "TestRepeated"}\n'
    )
    stream += '{"Action":"fail","Package":"one"}\n{"Action":"pass","Package":"two"}\n'
    workspace = _use_fake_go(monkeypatch, tmp_path, stream)
    reward = grade_gotest.grade(GotestSpec(must_not_break=("one.TestRepeated",)), tmp_path, workspace)
    assert (reward.reward, reward.detail["first_failure"]) == (0.0, "one.TestRepeated")
    assert grade_gotest.grade(GotestSpec(must_pass=("two.TestRepeated",)), tmp_path, workspace).reward == 1.0


@pytest.mark.parametrize(
    "suffix,exit_code",
    [
        ('{"Action":"fail","Package":"p"}\n', 1),
        ("", 0),
        ('{"Action":"pass","Package":"p"}\n', 2),
        ('{"Action":"run","Package":"p","Test":"TestPending"}\n{"Action":"fail","Package":"p"}\n', 1),
    ],
)
def test_gotest_failclosed_incomplete_or_crashed_runner_cannot_report_success(tmp_path, monkeypatch, suffix, exit_code):
    stream = '{"Action":"pass","Package":"p","Test":"TestRequired"}\n' + suffix
    workspace = _use_fake_go(monkeypatch, tmp_path, stream, exit_code=exit_code)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verifier.toml").write_text('mode="gotest"\nmust_pass=["p.TestRequired"]\n')
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status == Status.INFRA_ERROR
    assert reward.reward == 0


def test_gotest_failclosed_one_package_error_is_not_hidden_by_another_packages_test_failure(tmp_path, monkeypatch):
    stream = (
        '{"Action":"pass","Package":"required","Test":"TestRequired"}\n'
        '{"Action":"fail","Package":"required"}\n'
        '{"Action":"fail","Package":"other","Test":"TestOther"}\n'
        '{"Action":"fail","Package":"other"}\n'
    )
    workspace = _use_fake_go(monkeypatch, tmp_path, stream)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verifier.toml").write_text('mode="gotest"\nmust_pass=["required.TestRequired"]\n')
    reward = run(tests / "verifier.toml", workspace)
    assert reward.status == Status.INFRA_ERROR
    assert reward.reward == 0


@pytest.mark.skipif(shutil.which("go") is None, reason="the go toolchain is not installed")
def test_gotest_undiscovered_test_function_cannot_leave_positive_reward(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "go.mod").write_text("module example.com/m\n\ngo 1.18\n")
    (workspace / "calc.go").write_text(CALC_GO)
    # Go compiles this Test function but does not execute it without _test.go.
    (workspace / "checks.go").write_text(CALC_TEST_GO)
    native = subprocess.run(["go", "test", "-json", "./..."], cwd=workspace, capture_output=True, text=True)
    assert native.returncode == 0
    assert "[no test files]" in native.stdout
    spec = tmp_path / "verifier.toml"
    spec.write_text('mode = "gotest"\n')
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.txt").write_text("1.0")
    assert main([str(spec), "--workspace", str(workspace), "--logs-dir", str(logs)]) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert verdict["status"] == "scored"
    assert verdict["reward"] == 0.0
    assert verdict["detail"]["reason"] == "no_tests"
    assert float((logs / "reward.txt").read_text()) == 0.0
