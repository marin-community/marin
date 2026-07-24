# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

from finelog.deploy.build import build_image


def test_pushed_image_defaults_to_amd64(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    build_image(
        image="example.invalid/finelog:test",
        additional_tags=("example.invalid/finelog:alias",),
        cache_image="example.invalid/finelog-cache:test",
    )

    [argv] = calls
    assert argv[argv.index("--platform") + 1] == "linux/amd64"
    assert [argv[index + 1] for index, argument in enumerate(argv) if argument == "--tag"] == [
        "example.invalid/finelog:test",
        "example.invalid/finelog:alias",
    ]
    assert argv[argv.index("--cache-from") + 1] == "type=registry,ref=example.invalid/finelog-cache:test"
    assert argv[argv.index("--cache-to") + 1] == (
        "type=registry,ref=example.invalid/finelog-cache:test,mode=max,compression=zstd,"
        "compression-level=3,oci-mediatypes=true,image-manifest=true"
    )
