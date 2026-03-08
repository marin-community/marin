# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import threading

from scripts.gdn import gdnctl


def _args(*, codex_ephemeral: bool, search: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        codex_ephemeral=codex_ephemeral,
        model="gpt-5.3-codex",
        reasoning_effort="xhigh",
        codex_profile=None,
        search=search,
    )


def test_build_codex_exec_cmd_defaults_to_ephemeral() -> None:
    cmd = gdnctl._build_codex_exec_cmd(
        codex_bin="/tmp/codex",
        workdir=Path("/repo"),
        message_path=Path("/tmp/last-message.txt"),
        args=_args(codex_ephemeral=True),
        search_supported=False,
    )

    assert "--ephemeral" in cmd
    assert cmd[-1] == "-"


def test_build_codex_exec_cmd_allows_disabling_ephemeral() -> None:
    cmd = gdnctl._build_codex_exec_cmd(
        codex_bin="/tmp/codex",
        workdir=Path("/repo"),
        message_path=Path("/tmp/last-message.txt"),
        args=_args(codex_ephemeral=False, search=True),
        search_supported=True,
    )

    assert "--ephemeral" not in cmd
    assert "--search" in cmd


def test_stream_subprocess_output_to_file_can_suppress_after_ready(tmp_path: Path) -> None:
    log_path = tmp_path / "allocate.log"
    ready_event = threading.Event()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "print('hello', flush=True); "
                "print('TPU allocation is active. Press Ctrl-C to release...', flush=True); "
                "print('noisy line 1', flush=True); "
                "print('noisy line 2', flush=True)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    thread = gdnctl._stream_subprocess_output_to_file(
        proc,
        output_path=log_path,
        ready_markers=gdnctl.DEV_TPU_READY_MARKERS,
        ready_event=ready_event,
        suppress_output_after_ready=True,
    )
    proc.wait(timeout=10)
    thread.join(timeout=10)

    assert ready_event.is_set()
    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text
    assert "TPU allocation is active. Press Ctrl-C to release..." in text
    assert "suppressing further allocation log output" in text
    assert "noisy line 1" not in text
    assert "noisy line 2" not in text
