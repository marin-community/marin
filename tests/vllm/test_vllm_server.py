# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys

from marin.inference import vllm_server


def test_vllm_native_command_uses_vllm_binary_without_preload(monkeypatch):
    monkeypatch.setattr(vllm_server.shutil, "which", lambda name: f"/usr/bin/{name}")

    command = vllm_server._vllm_native_command(["serve", "model"], {})

    assert command == ["/usr/bin/vllm", "serve", "model"]


def test_vllm_native_command_wraps_cli_when_preload_modules_are_set():
    command = vllm_server._vllm_native_command(
        ["serve", "model"],
        {vllm_server._PRELOAD_MODULES_ENV: "experiments.grug.moe.vllm_registry"},
    )

    assert command[:3] == [sys.executable, "-c", vllm_server._VLLM_PRELOAD_CLI]
    assert command[3:] == ["serve", "model"]
    assert vllm_server._PRELOAD_MODULES_ENV in command[2]
