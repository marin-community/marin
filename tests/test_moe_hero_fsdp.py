# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_fsdp import launch, train


def test_build_hero_run_uses_run_id_argument(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "grug/cli-run"


def test_build_hero_run_starts_one_process_per_gpu():
    step = launch.build_supervised_hero_run(
        run_id="one-process-per-gpu",
        dp_racks=2,
        num_steps=1,
        version="2026.08.07",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    train_resources = step.runtime_args["train_resources"]

    assert config.processes_per_task == train_resources.device.chip_count() == 4


def test_run_grug_applies_xla_command_buffer_default_and_keeps_override(monkeypatch):
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

        explicit_flags = "--xla_gpu_enable_command_buffer=FUSION"
        monkeypatch.setenv("XLA_FLAGS", explicit_flags)
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"] == explicit_flags
