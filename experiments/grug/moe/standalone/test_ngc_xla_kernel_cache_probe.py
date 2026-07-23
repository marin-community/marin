# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from experiments.grug.moe.standalone.ngc_xla_kernel_cache_probe import copy_artifact


def test_copy_artifact_writes_verified_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "overlay" / "xla_cuda_plugin.so"
    payload = b"patched-cuda-plugin"
    source.write_bytes(payload)

    copied_sha256 = copy_artifact(str(source), destination, hashlib.sha256(payload).hexdigest())

    assert copied_sha256 == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload


def test_copy_artifact_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "overlay" / "xla_cuda_plugin.so"
    source.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        copy_artifact(str(source), destination, "0" * 64)

    assert not destination.exists()
