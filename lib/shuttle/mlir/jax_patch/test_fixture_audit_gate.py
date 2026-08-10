# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the native six-fixture audit boundary."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from fixture_audit_gate import SHUTTLE_TEST_OPT_LABEL

GATE = Path(__file__).with_name("fixture_audit_gate.py")
GENERATOR = Path(__file__).parents[1] / "test" / "Inputs" / "regenerate-jax-fixtures.py"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _run_gate(tmp_path: Path, output_name: str) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    xla_source = tmp_path / "xla"
    normalizer = xla_source / "bazel-out" / "k8-opt" / "bin" / "external" / "shuttle_mlir" / output_name
    normalizer.parent.mkdir(parents=True)
    _write_executable(normalizer, "#!/bin/sh\nexit 0\n")
    bazel_log = tmp_path / "bazel.jsonl"
    bazel = tmp_path / "bazel"
    _write_executable(
        bazel,
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["FAKE_BAZEL_LOG"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if "cquery" in sys.argv:
    print(os.environ["FAKE_BAZEL_OUTPUT"])
""",
    )
    generator_record = tmp_path / "generator-normalizer.txt"
    generator = tmp_path / "generator.py"
    generator.write_text(
        """import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--normalizer", required=True)
arguments = parser.parse_args()
Path(os.environ["GENERATOR_RECORD"]).write_text(arguments.normalizer)
"""
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_BAZEL_LOG": str(bazel_log),
            "FAKE_BAZEL_OUTPUT": str(normalizer.relative_to(xla_source)),
            "GENERATOR_RECORD": str(generator_record),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--bazel",
            str(bazel),
            "--xla-source",
            str(xla_source),
            "--output-user-root",
            str(tmp_path / "output-root"),
            "--repository-cache",
            str(tmp_path / "repository-cache"),
            "--jobs",
            "24",
            "--ram-mb",
            "65536",
            "--python",
            sys.executable,
            "--generator",
            str(generator),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, bazel_log, generator_record


def test_fixture_audit_gate_builds_and_passes_shuttle_test_opt(tmp_path):
    result, bazel_log, generator_record = _run_gate(tmp_path, "shuttle-test-opt")

    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in bazel_log.read_text().splitlines()]
    assert commands[0][-2:] == ["--show_result=0", SHUTTLE_TEST_OPT_LABEL]
    assert commands[1][-3:] == ["cquery", "--output=files", SHUTTLE_TEST_OPT_LABEL]
    selected = Path(generator_record.read_text())
    assert selected.name == "shuttle-test-opt"
    assert f"fixture_audit_normalizer={selected}" in result.stdout
    assert "six_fixture_default_audit=PASS" in result.stdout


def test_fixture_audit_gate_rejects_production_shuttle_opt(tmp_path):
    result, _, generator_record = _run_gate(tmp_path, "shuttle-opt")

    assert result.returncode != 0
    assert "output must be named shuttle-test-opt" in result.stderr
    assert not generator_record.exists()


def test_fixture_generator_bounds_failed_normalizer_diagnostics(tmp_path):
    normalizer = tmp_path / "wrong-normalizer"
    _write_executable(
        normalizer,
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdout.write('s' * 10000)\n"
        "sys.stderr.write('e' * 10000)\n"
        "raise SystemExit(7)\n",
    )
    normalized_fingerprint = runpy.run_path(str(GENERATOR))["normalized_fingerprint"]

    with pytest.raises(RuntimeError) as raised:
        normalized_fingerprint("module {}\n", normalizer)

    message = str(raised.value)
    assert "exit_code=7" in message
    assert str(normalizer) in message
    assert "--shuttle-test-report-normalized-fingerprint" in message
    assert "characters omitted" in message
    assert len(message) < 5_000
