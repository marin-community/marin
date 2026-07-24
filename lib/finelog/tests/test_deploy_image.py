# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

from finelog.deploy.image import resolve_image_digest


def test_resolve_image_digest_preserves_multiarch_image_index(monkeypatch) -> None:
    index_digest = "sha256:" + "a" * 64

    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args[0], 0, stdout=f'{{"digest":"{index_digest}"}}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert resolve_image_digest("example.invalid/finelog:latest") == f"example.invalid/finelog@{index_digest}"
