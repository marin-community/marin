# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from infra.ci.shellsim_test_runtime import shellsim_factory  # noqa: F401
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable

from taskcompendium.models import CommandSemantics, EnvironmentRequirements


@pytest.fixture(autouse=True)
def local_pipeline_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = str(tmp_path / "marin")
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setenv("MARIN_TEMP_PREFIX", prefix)


@pytest.fixture
def local_machine_environment(tmp_path: Path) -> tuple[EnvironmentRequirements, LocalMachineFactory]:
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"")
    (tmp_path / ".artifact.json").write_text(
        json.dumps({"result": {"lock_sha256": hashlib.sha256(b"").hexdigest(), "data": []}})
    )
    try:
        factory = LocalMachineFactory()
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    return EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=str(lock)), factory
