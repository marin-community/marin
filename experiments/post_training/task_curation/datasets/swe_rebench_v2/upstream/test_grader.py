# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the standalone SWE-rebench V2.1 grader."""

import json
import subprocess
import sys

import pytest

from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import grader_source


def run_grader(tmp_path, parser, config, log):
    grader = tmp_path / "grade.py"
    grader.write_text(grader_source(parser))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    log_path = tmp_path / "test_output.log"
    log_path.write_text(log)
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [sys.executable, "-I", "-S", str(grader), str(config_path), str(log_path), str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(report_path.read_text())


def go_json_log(*events):
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_go_json_named_results_match_configured_ids(tmp_path):
    config = {"FAIL_TO_PASS": ["TestRepair"], "PASS_TO_PASS": ["TestExisting"]}
    log = go_json_log(
        {"Action": "pass", "Package": "example.com/first", "Test": "TestRepair"},
        {"Action": "pass", "Package": "example.com/second", "Test": "TestExisting"},
        {"Action": "pass", "Package": "example.com/first"},
    )

    result, report = run_grader(tmp_path, "parse_log_gotest", config, log)

    assert result.returncode == 0
    assert report["resolved"] is True
    assert report["fail_to_pass"] == {"TestRepair": True}
    assert report["pass_to_pass"] == {"TestExisting": True}
    assert report["parsed_counts"] == {"FAILED": 0, "PASSED": 2, "SKIPPED": 0, "ERROR": 0}


@pytest.mark.parametrize(
    "parser,log",
    [
        ("parse_log_js_4", "PASS src/repair.test.ts\nTest Suites: 1 passed, 1 total\nTests: 1 passed, 1 total\n"),
        ("parse_log_gotest", "ok  example.com/repair  0.123s\n"),
        ("parse_log_gotest", go_json_log({"Action": "pass", "Package": "example.com/repair"})),
    ],
)
def test_summary_only_log_is_unscorable_with_named_ids(tmp_path, parser, log):
    result, report = run_grader(tmp_path, parser, {"FAIL_TO_PASS": ["TestRepair"], "PASS_TO_PASS": []}, log)

    assert result.returncode == 1
    assert report["unscorable"] is True
    assert report["resolved"] is False
    assert report["fail_to_pass"] == {"TestRepair": False}
    assert "UNSCORABLE" in result.stdout


@pytest.mark.parametrize("action", ["fail", "skip"])
def test_go_json_failure_or_skip_cannot_be_overwritten_by_another_package_pass(tmp_path, action):
    config = {"FAIL_TO_PASS": ["TestRepair"], "PASS_TO_PASS": []}
    log = go_json_log(
        {"Action": action, "Package": "example.com/first", "Test": "TestRepair"},
        {"Action": "pass", "Package": "example.com/second", "Test": "TestRepair"},
    )

    result, report = run_grader(tmp_path, "parse_log_gotest", config, log)

    assert result.returncode == 1
    assert report["resolved"] is False
    assert report["fail_to_pass"] == {"TestRepair": False}
    assert report["parsed_counts"]["FAILED" if action == "fail" else "SKIPPED"] == 1


@pytest.mark.parametrize("named_pass", [False, True])
def test_go_json_build_error_keeps_suite_unresolved(tmp_path, named_pass):
    config = {"FAIL_TO_PASS": ["TestRepair"], "PASS_TO_PASS": []}
    events = []
    if named_pass:
        events.append({"Action": "pass", "Package": "example.com/first", "Test": "TestRepair"})
    events.append({"Action": "fail", "Package": "example.com/second", "FailedBuild": "example.com/second"})

    result, report = run_grader(tmp_path, "parse_log_gotest", config, go_json_log(*events))

    assert result.returncode == 1
    assert report["unscorable"] is not named_pass
    assert report["resolved"] is False
    assert report["parsed_counts"]["ERROR"] == 1
