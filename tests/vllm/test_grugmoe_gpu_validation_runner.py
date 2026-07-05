# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "iris" / "run_grugmoe_gpu_validation_remote.py"


def _load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_grugmoe_gpu_validation_remote",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def test_grugmoe_gpu_validation_runner_reads_pinned_vllm_rev(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.sources]
vllm = { git = "https://github.com/marin-community/vllm.git", rev = "abc123" }
""".lstrip()
    )

    assert runner._pinned_vllm_rev(tmp_path) == "abc123"


def test_grugmoe_gpu_validation_runner_checks_vllm_ref_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.sources]
vllm = { git = "https://github.com/marin-community/vllm.git", rev = "abc123" }
""".lstrip()
    )

    def fake_check_output(command: list[str], **_: Any) -> str:
        assert command[:2] == ["git", "rev-parse"]
        return "abc123"

    monkeypatch.setattr(runner, "_check_output", fake_check_output)

    alignment = runner._vllm_ref_alignment(
        repo_root=tmp_path,
        vllm_dir=tmp_path / "vllm",
        requested_ref="codex/grugmoe-gpu-20260702",
    )

    assert alignment == {
        "requested_ref": "codex/grugmoe-gpu-20260702",
        "checked_out_sha": "abc123",
        "pyproject_pinned_rev": "abc123",
        "pyproject_pinned_sha": "abc123",
        "matches_pyproject_pin": True,
    }


def test_grugmoe_gpu_validation_runner_writes_command_output_to_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "command.log"
    sentinel = "runner-output-sentinel"

    report = runner._run(
        [sys.executable, "-c", "print('runner-' + 'output-' + 'sentinel')"],
        cwd=tmp_path,
        log_path=log_path,
    )

    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert sentinel in log_path.read_text()
    assert report["returncode"] == 0
    assert report["log_path"] == str(log_path)


def test_grugmoe_gpu_validation_runner_prints_failure_log_tail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "command.log"

    with pytest.raises(subprocess.CalledProcessError):
        runner._run(
            [
                sys.executable,
                "-c",
                "print('tail-' + str(40 + 2)); raise SystemExit(7)",
            ],
            cwd=tmp_path,
            log_path=log_path,
            failure_tail_lines=1,
        )

    captured = capsys.readouterr()
    assert "tail-42" in captured.out
    assert f"last 1 lines from {log_path}" in captured.out
    assert "tail-42" in log_path.read_text()


def test_grugmoe_gpu_validation_runner_prints_failure_tail_without_checking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "command.log"

    report = runner._run(
        [
            sys.executable,
            "-c",
            "print('tail-' + str(100 + 23)); raise SystemExit(7)",
        ],
        cwd=tmp_path,
        log_path=log_path,
        check=False,
        failure_tail_lines=1,
    )

    captured = capsys.readouterr()
    assert report["returncode"] == 7
    assert "tail-123" in captured.out
    assert f"last 1 lines from {log_path}" in captured.out
    assert "tail-123" in log_path.read_text()
