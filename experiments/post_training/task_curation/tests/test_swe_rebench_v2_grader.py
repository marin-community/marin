# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the standalone SWE-rebench V2.1 grader emitted for Harbor tasks."""

import json
import subprocess
import sys

import pytest

from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import grader_source
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import test_sh as render_test_sh


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


@pytest.mark.parametrize(
    "result_line, test_exit_code, expected_resolved, expected_reward",
    [
        (
            "[ERROR] The forked VM terminated without properly saying goodbye. VM crash or System.exit called?",
            1,
            False,
            0,
        ),
        ("[INFO] BUILD SUCCESS", 0, True, 1),
    ],
)
def test_swe_rebench_maven_runner_requires_completed_test_evidence(
    tmp_path, result_line, test_exit_code, expected_resolved, expected_reward
):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "config.json").write_text(json.dumps({"FAIL_TO_PASS": ["com.example.WidgetTest"], "PASS_TO_PASS": []}))
    (tests / "grade.py").write_text(grader_source("parse_java_mvn"))
    (tests / "test.patch").write_text("--- a/marker.txt\n+++ b/marker.txt\n@@ -1 +1 @@\n-before\n+after\n")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("before\n")
    subprocess.run(["git", "init", "-q", str(workdir)], check=True)

    test_cmd = f"printf '%s\\n' '[INFO] Running com.example.WidgetTest' '{result_line}'; exit {test_exit_code}"
    script = render_test_sh({"workdir": str(workdir), "test_files": [], "test_cmd": test_cmd})
    script = script.replace("/logs/verifier", str(tmp_path / "logs"))
    script = script.replace("/tests/", str(tests) + "/")
    script = script.replace("/tmp/test.patch", str(tmp_path / "test.patch"))
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "logs/report.json").read_text())
    assert report["resolved"] is expected_resolved
    assert float((tmp_path / "logs/reward.txt").read_text()) == expected_reward
