# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from collections.abc import Sequence
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "iap_gclb.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("iap_gclb", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
iap_gclb = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(iap_gclb)


def test_controller_backends_reconcile_one_hour_request_timeout(monkeypatch) -> None:
    backend = iap_gclb.Backend(cluster="test", project="test-project", zone="test-zone")
    commands: list[list[str]] = []

    def fake_run(command: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        command = list(command)
        commands.append(command)
        stdout = ""
        if "list-network-endpoints" in command:
            stdout = "10.0.0.1"
        elif "--format=value(backends[].group)" in command:
            stdout = backend.neg
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(iap_gclb.subprocess, "run", fake_run)

    iap_gclb.ensure_backend(backend, "controller", "10.0.0.1", dry_run=False)
    iap_gclb.ensure_token_proxy_backend(backend, dry_run=False)

    timeout_by_service = {
        command[4]: next(arg.removeprefix("--timeout=") for arg in command if arg.startswith("--timeout="))
        for command in commands
        if command[2:4] == ["backend-services", "update"] and any(arg.startswith("--timeout=") for arg in command)
    }
    assert timeout_by_service == {
        backend.service: "3600s",
        backend.proxy_service: "3600s",
    }
