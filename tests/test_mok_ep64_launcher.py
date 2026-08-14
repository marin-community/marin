# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeForwardXStorage,
    MokLikeTopology,
)
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_mok_ep64 import (
    EP64_NODES,
    EXPERT_AXIS_SIZE,
    GLOBAL_BATCH_SIZE,
    PROCESSES_PER_TASK,
    build_mok_ep64_run,
    main,
)


def test_ep64_launcher_builds_one_rack_staged_contract():
    step = build_mok_ep64_run(
        run_id="ep64-test",
        num_steps=25,
        num_layers=1,
        schedule_capacity_factor=4,
        version="2026.08.13",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert step.runtime_args["train_resources"].replicas == EP64_NODES
    assert config.processes_per_task == PROCESSES_PER_TASK
    assert config.trainer.expert_axis_size == EXPERT_AXIS_SIZE
    assert config.trainer.replica_axis_size == 1
    assert config.trainer.trainer.train_batch_size == GLOBAL_BATCH_SIZE
    assert config.model.num_experts == 128
    assert config.model.num_layers == 1
    assert config.model.mok_like is not None
    assert config.model.mok_like.topology is MokLikeTopology.NVLINK_EP64
    assert config.model.mok_like.workspace_slots == 1
    assert config.model.mok_like.forward_x_storage is MokLikeForwardXStorage.RUNTIME_STAGED
    assert config.model.mok_like.backward_peer_storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED
    assert config.model.mok_like.schedule_capacity_factor == 4
    assert config.max_retries_failure == 0
    assert config.max_retries_preemption == 0
    assert config.max_task_failures == 0


def test_ep64_launcher_names_strict_capacity_contract():
    step = build_mok_ep64_run(
        run_id="ep64-strict-test",
        num_steps=25,
        num_layers=1,
        schedule_capacity_factor=64,
        version="2026.08.13",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))
    tags = config.trainer.trainer.tracker.tags

    assert "strict-dropless-ep64" in tags
    assert "mok-like-schedule-capacity-64" in tags


def test_ep64_launcher_clamps_optimizer_learning_rates():
    step = build_mok_ep64_run(
        run_id="ep64-low-lr-test",
        num_steps=3,
        num_layers=1,
        schedule_capacity_factor=4,
        max_learning_rate=1e-6,
        version="2026.08.14",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert config.optimizer.learning_rate == 1e-6
    assert config.optimizer.adam_lr == 1e-6
    assert "max-learning-rate-1e-06" in config.trainer.trainer.tracker.tags


def test_ep64_cli_accepts_shared_version_option():
    result = CliRunner().invoke(
        main,
        [
            "--run-id",
            "mok-ep64-cli",
            "--num-steps",
            "2",
            "--num-layers",
            "2",
            "--max-learning-rate",
            "1e-6",
            "--version",
            "dev",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "grug/mok-ep64/mok-ep64-cli" in result.output
