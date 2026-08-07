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


def test_ablation_sweep_builds_one_named_run_per_arm():
    step = launch.build_ablation_sweep_hero_run(
        run_id="sweep",
        dp_racks=2,
        steps_per_arm=7,
        ablation_names=("baseline", "nccl-proto-simple"),
        version="2026.08.07",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert [arm.name for arm in config.arms] == ["baseline", "nccl-proto-simple"]
    assert config.arms[1].env == {"NCCL_PROTO": "Simple"}
    # Each arm needs its own trainer/W&B identity or the arms overwrite each other's run.
    assert [run.trainer.trainer.id for run in config.runs] == ["sweep-baseline", "sweep-nccl-proto-simple"]
    assert all(run.trainer.trainer.num_train_steps == 7 for run in config.runs)
    # A sweep is a diagnostic; the hero checkpoint is ~2.7 TiB per arm.
    assert not any(run.trainer.save_checkpoints for run in config.runs)


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


def test_stock_control_carries_no_recovery_flags_and_no_retries(monkeypatch):
    """The instrumentation comparison needs one leg carrying none of it, and a retry restarts exposure."""
    monkeypatch.setenv("XLA_FLAGS", "")
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="stock")),
        resources=object(),
        processes_per_task=4,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        train.run_grug_stock_control(config)

    assert dispatch.call_args.kwargs["max_retries_failure"] == 0
    assert "xla_gpu_execution_terminate_timeout" not in os.environ["XLA_FLAGS"]
    assert "xla_gpu_nccl_termination_timeout_seconds" not in os.environ["XLA_FLAGS"]


def test_failsafe_control_carries_the_recovery_flags(monkeypatch):
    """The middle leg isolates the XLA flags from the supervisor parent, so it must still set them."""
    monkeypatch.setenv("XLA_FLAGS", "")
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="failsafe")),
        resources=object(),
        processes_per_task=4,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        train.run_grug_failsafe_control(config)

    assert dispatch.call_args.kwargs["max_retries_failure"] == 0
    assert (
        f"--xla_gpu_execution_terminate_timeout={int(train.HERO_EXECUTION_TERMINATE_TIMEOUT)}s"
        in os.environ["XLA_FLAGS"]
    )


def test_hero_detection_clears_the_measured_first_execution():
    """The deadman must exceed a cold first execution or it fires before step 1 every run."""
    measured_first_execution = 261.0
    assert train.HERO_DETECTION_CONFIG.execution_terminate_timeout_seconds > measured_first_execution
    # Thunk reporting throws std::bad_alloc on a module this size, destroying the diagnostic.
    assert train.HERO_DETECTION_CONFIG.progress_tracking == 0
