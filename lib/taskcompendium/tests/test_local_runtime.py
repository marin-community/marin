# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integrity checks at the lock artifact boundary."""

import hashlib
import json
from dataclasses import replace

import pytest

from taskcompendium.runtime.local import local_runtime


def test_runtime_rejects_lock_changed_after_artifact_build(tmp_path):
    lock = tmp_path / "requirements.lock"
    original = b"six==1.17.0\n"
    lock.write_bytes(original)
    (tmp_path / ".artifact.json").write_text(
        json.dumps({"result": {"lock_sha256": hashlib.sha256(original).hexdigest(), "data": []}})
    )
    runtime = replace(local_runtime(str(lock)), parent=tmp_path / "runtime")
    lock.write_text("six==1.16.0\n")

    with pytest.raises(RuntimeError, match="SHA-256"):
        runtime.ensure_built()
    assert not (runtime.root / ".complete").exists()
