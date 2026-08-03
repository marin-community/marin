# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_fsdp import checkpoint_benchmark, launch, train


def test_build_hero_run_uses_run_id_argument(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "grug/cli-run"


def test_checkpoint_benchmark_uses_ttl_output_and_low_overhead_timing(monkeypatch):
    monkeypatch.setattr(
        checkpoint_benchmark,
        "marin_temp_bucket",
        lambda ttl_days, prefix: f"s3://temp/ttl={ttl_days}d/{prefix}",
    )

    step = checkpoint_benchmark.build_checkpoint_benchmark_run(
        run_id="checkpoint-test",
        dp_racks=1,
        num_steps=12,
        checkpoint_every_steps=8,
        version="2026.08.03",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert step.override_path == "s3://temp/ttl=1d/grug/checkpoint-benchmark/checkpoint-test/2026.08.03"
    assert config.model.num_experts == 128
    assert config.model.num_experts_per_token == 1
    assert config.model.hidden_dim == 2048
    assert config.model.num_layers == 32
    assert config.trainer.offload_opt_state is True
    assert config.trainer.trainer.checkpointer.keep == [{"every": 8}]
    assert config.trainer.trainer.checkpointer.debug.enabled is True
    assert config.trainer.trainer.checkpointer.debug.tracemalloc_frames is None
    assert config.trainer.trainer.tracker[0].mode == "disabled"


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
