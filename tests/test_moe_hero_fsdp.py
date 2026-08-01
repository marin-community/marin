# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from experiments.grug.moe_hero_fsdp import launch, train


def test_launcher_requires_run_id():
    result = CliRunner().invoke(launch.main, ["--dp-racks", "1", "--version", "dev"])

    assert result.exit_code == 2
    assert "Missing option '--run-id'" in result.output


def test_build_hero_run_uses_cli_run_id(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "grug/cli-run"


def test_run_grug_disables_command_buffers_and_preserves_xla_flags(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true")
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["XLA_FLAGS"].split() == [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
    ]


def test_run_grug_keeps_explicit_command_buffer_setting(monkeypatch):
    explicit_flags = "--xla_gpu_enable_command_buffer=FUSION"
    monkeypatch.setenv("XLA_FLAGS", explicit_flags)
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["XLA_FLAGS"] == explicit_flags
