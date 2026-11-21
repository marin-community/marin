# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the standalone SWE-rebench V2.1 grader emitted for Harbor tasks."""

import json
import subprocess
import sys

import pytest

from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import grader_source


@pytest.mark.parametrize(
    "config, log, exit_code, unscorable, resolved",
    [
        ({"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, "", 1, True, False),
        ({"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, "PASSED tests/test_a.py::test_one\n", 1, False, False),
        (
            {"FAIL_TO_PASS": ["tests/test_a.py::test_one"], "PASS_TO_PASS": ["tests/test_a.py::test_two"]},
            "PASSED tests/test_a.py::test_one\nPASSED tests/test_a.py::test_two\n",
            0,
            False,
            True,
        ),
    ],
)
def test_swe_rebench_grader_exits_with_evidence_based_verdict(tmp_path, config, log, exit_code, unscorable, resolved):
    grader = tmp_path / "grade.py"
    grader.write_text(grader_source("parse_log_pytest"))
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

    report = json.loads(report_path.read_text())
    assert (result.returncode, report["unscorable"], report["resolved"]) == (exit_code, unscorable, resolved)
    assert ("UNSCORABLE" in result.stdout) is unscorable
