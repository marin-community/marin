# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

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
