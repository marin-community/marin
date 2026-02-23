# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.gdn import gdnctl


def test_sync_branch_from_remote_runs_fetch_then_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run_git(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[:2] == ["merge", "--no-edit"]:
            return subprocess.CompletedProcess(["git", *args], 0, stdout="Already up to date.\n", stderr="")
        return subprocess.CompletedProcess(["git", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(gdnctl, "_run_git", _fake_run_git)
    monkeypatch.setattr(gdnctl, "_echo_cmd", lambda _cmd: None)

    gdnctl._sync_branch_from_remote(workdir=tmp_path, remote="origin", branch="main")

    assert calls[0] == ["fetch", "origin", "main"]
    assert calls[1] == ["merge", "--no-edit", "origin/main"]


def test_sync_branch_from_remote_aborts_on_merge_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run_git(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[:2] == ["merge", "--no-edit"]:
            return subprocess.CompletedProcess(["git", *args], 1, stdout="", stderr="conflict")
        return subprocess.CompletedProcess(["git", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(gdnctl, "_run_git", _fake_run_git)
    monkeypatch.setattr(gdnctl, "_echo_cmd", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="failed to merge origin/main"):
        gdnctl._sync_branch_from_remote(workdir=tmp_path, remote="origin", branch="main")

    assert ["merge", "--abort"] in calls
