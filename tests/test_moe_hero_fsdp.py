# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

from experiments.grug.moe_hero_fsdp import launch, train


def test_build_hero_run_puts_fixed_version_checkpoint_under_user_directory(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "users/alice/grug/cli-run"


def test_run_grug_applies_runtime_defaults_and_keeps_overrides(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true")
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("MALLOC_CONF", raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        run_mode=train.GrugRunMode.DEFAULT,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"].split() == [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
        ]
        assert os.environ["LD_PRELOAD"] == "libjemalloc.so.2"
        assert os.environ["MALLOC_CONF"] == "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2"

        explicit_flags = "--xla_gpu_enable_command_buffer=FUSION"
        monkeypatch.setenv("XLA_FLAGS", explicit_flags)
        monkeypatch.setenv("LD_PRELOAD", "/opt/custom/liballocator.so")
        monkeypatch.setenv("MALLOC_CONF", "narenas:8")
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"] == explicit_flags
        assert os.environ["LD_PRELOAD"] == "/opt/custom/liballocator.so"
        assert os.environ["MALLOC_CONF"] == "narenas:8"
