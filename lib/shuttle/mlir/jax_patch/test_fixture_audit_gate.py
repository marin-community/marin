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


def _run_gate(
    tmp_path: Path,
    output_paths: tuple[str, ...],
    *,
    executable: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    xla_source = tmp_path / "xla"
    xla_source.mkdir()
    real_bazel_out = tmp_path / "real-bazel-out"
    real_bazel_out.mkdir()
    (xla_source / "bazel-out").symlink_to(real_bazel_out, target_is_directory=True)
    for output in output_paths:
        relative_path = Path(output)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        target = xla_source / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o755 if executable else 0o644)
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
if "cquery" in sys.argv and os.environ["FAKE_BAZEL_OUTPUT"]:
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
            "FAKE_BAZEL_OUTPUT": "\n".join(output_paths),
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
    output_path = "bazel-out/k8-opt/bin/external/shuttle_mlir/shuttle-test-opt"
    result, bazel_log, generator_record = _run_gate(tmp_path, (output_path,))

    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in bazel_log.read_text().splitlines()]
    assert commands == [
        [
            f"--output_user_root={tmp_path / 'output-root'}",
            "build",
            f"--repository_cache={tmp_path / 'repository-cache'}",
            "--jobs=24",
            "--local_cpu_resources=24",
            "--local_ram_resources=65536",
            "--noshow_progress",
            "--show_result=0",
            SHUTTLE_TEST_OPT_LABEL,
        ],
        [
            f"--output_user_root={tmp_path / 'output-root'}",
            "cquery",
            "--output=files",
            SHUTTLE_TEST_OPT_LABEL,
        ],
    ]
    selected = Path(generator_record.read_text())
    assert selected == tmp_path / "xla" / output_path
    assert selected.name == "shuttle-test-opt"
    assert selected.resolve().is_relative_to(tmp_path / "real-bazel-out")
    assert f"fixture_audit_normalizer={selected}" in result.stdout
    assert "six_fixture_default_audit=PASS" in result.stdout


def test_fixture_audit_gate_rejects_production_shuttle_opt(tmp_path):
    output_path = "bazel-out/k8-opt/bin/external/shuttle_mlir/shuttle-opt"
    result, _, generator_record = _run_gate(tmp_path, (output_path,))

    assert result.returncode != 0
    assert "output must be named shuttle-test-opt" in result.stderr
    assert not generator_record.exists()


@pytest.mark.parametrize(
    ("output_paths", "diagnostic"),
    (
        ((), "produced 0 output paths"),
        (
            (
                "bazel-out/k8-opt/bin/external/shuttle_mlir/shuttle-test-opt",
                "bazel-out/k8-opt/bin/external/shuttle_mlir/other/shuttle-test-opt",
            ),
            "produced 2 output paths",
        ),
        (("/absolute/shuttle-test-opt",), "unsafe output path"),
        (("../outside/shuttle-test-opt",), "unsafe output path"),
    ),
)
def test_fixture_audit_gate_rejects_ambiguous_or_unsafe_outputs(tmp_path, output_paths, diagnostic):
    result, _, generator_record = _run_gate(tmp_path, output_paths)

    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not generator_record.exists()


def test_fixture_audit_gate_rejects_non_executable_output(tmp_path):
    output_path = "bazel-out/k8-opt/bin/external/shuttle_mlir/shuttle-test-opt"
    result, _, generator_record = _run_gate(tmp_path, (output_path,), executable=False)

    assert result.returncode != 0
    assert "output is not executable" in result.stderr
    assert not generator_record.exists()


def test_fixture_generator_bounds_failed_normalizer_diagnostics(tmp_path):
    normalizer = tmp_path / "wrong-normalizer"
    _write_executable(
        normalizer,
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdout.write('\\x00' * 10000)\n"
        "sys.stderr.write('🙂' * 10000)\n"
        "raise SystemExit(7)\n",
    )
    normalized_fingerprint = runpy.run_path(str(GENERATOR))["normalized_fingerprint"]

    with pytest.raises(RuntimeError) as raised:
        normalized_fingerprint("module {}\n", normalizer)

    message = str(raised.value)
    assert "exit_code=7" in message
    assert str(normalizer) in message
    assert "--shuttle-test-report-normalized-fingerprint" in message
    assert "stdout=" in message
    assert "stderr=" in message
    assert "serialized field truncated" in message
    assert len(message) <= 4_096
