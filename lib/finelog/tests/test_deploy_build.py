# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

from finelog.deploy.build import build_image


def test_pushed_image_targets_both_coreweave_control_node_architectures(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    build_image(image="example.invalid/finelog:test")

    assert calls[0][calls[0].index("--platform") + 1] == "linux/amd64,linux/arm64"
