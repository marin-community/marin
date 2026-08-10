# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from levanter.kernels.mixture_of_kittens.forward_ffi import MoKForwardConfig, schedule_capacity

from experiments.grug import dispatch as grug_dispatch
from experiments.grug.mixture_of_kittens import heuristic, launch, train
from experiments.grug.mixture_of_kittens.schedule import build_schedule


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _reference_schedule(
    top_experts: np.ndarray,
    *,
    num_local_experts: int,
    rank: int,
    expert_padding: int,
    schedule_capacity: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    world_size, _, top_k = top_experts.shape
    first_expert = rank * num_local_experts
    peer_rank = np.full(schedule_capacity, -1, dtype=np.int32)
    peer_token_idx = np.full(schedule_capacity, -1, dtype=np.int32)
    tokens_per_expert = np.zeros(num_local_experts, dtype=np.int32)
    output_offset = 0

    for local_expert in range(num_local_experts):
        global_expert = first_expert + local_expert
        peer_assignments: list[list[int]] = []
        for peer in range(world_size):
            assignments = [
                token * top_k + route
                for token in range(top_experts.shape[1])
                for route in range(top_k)
                if top_experts[peer, token, route] == global_expert
            ]
            peer_assignments.append(assignments)

        real_count = sum(len(assignments) for assignments in peer_assignments)
        padded_count = _round_up(real_count, expert_padding)
        tokens_per_expert[local_expert] = padded_count

        for peer_token_offset in range(max((len(assignments) for assignments in peer_assignments), default=0)):
            for peer, assignments in enumerate(peer_assignments):
                if peer_token_offset >= len(assignments):
                    continue
                destination = output_offset + sum(
                    min(len(other_assignments), peer_token_offset) for other_assignments in peer_assignments
                )
                destination += sum(
                    len(peer_assignments[earlier_peer]) > peer_token_offset for earlier_peer in range(peer)
                )
                if destination < schedule_capacity:
                    peer_rank[destination] = peer
                    peer_token_idx[destination] = assignments[peer_token_offset]

        output_offset += padded_count

    return peer_rank, peer_token_idx, output_offset, tokens_per_expert


@pytest.mark.parametrize(
    ("top_experts", "rank"),
    [
        (
            [
                [[0, 3], [1, 2], [0, 4]],
                [[1, 4], [0, 3], [1, 5]],
                [[0, 2], [0, 5], [1, 4]],
            ],
            0,
        ),
        (
            [
                [[0, 2], [3, 5], [2, 4]],
                [[2, 5], [4, 3], [2, 0]],
                [[5, 1], [3, 0], [4, 2]],
            ],
            1,
        ),
    ],
)
def test_build_schedule_matches_peer_interleaving_and_expert_padding(
    top_experts: Sequence[Sequence[Sequence[int]]], rank: int
):
    top_experts_array = np.asarray(top_experts, dtype=np.int32)
    expected = _reference_schedule(
        top_experts_array,
        num_local_experts=2,
        rank=rank,
        expert_padding=4,
        schedule_capacity=24,
    )

    actual = jax.jit(
        lambda routes, destination_rank: build_schedule(
            routes,
            num_local_experts=2,
            schedule_capacity=24,
            rank=destination_rank,
            expert_padding=4,
        )
    )(jnp.asarray(top_experts_array), jnp.asarray(rank, dtype=jnp.int32))

    np.testing.assert_array_equal(actual.peer_rank, expected[0])
    np.testing.assert_array_equal(actual.peer_token_idx, expected[1])
    assert int(actual.num_tokens) == expected[2]
    np.testing.assert_array_equal(actual.tokens_per_expert, expected[3])
    assert int(actual.dropped_assignments) == 0
    assert not bool(actual.overflow)


def test_build_schedule_reports_capacity_overflow():
    top_experts = jnp.zeros((2, 3, 1), dtype=jnp.int32)

    schedule = build_schedule(
        top_experts,
        num_local_experts=1,
        schedule_capacity=4,
        rank=jnp.asarray(0, dtype=jnp.int32),
        expert_padding=4,
    )

    assert int(schedule.num_tokens) == 4
    np.testing.assert_array_equal(schedule.tokens_per_expert, np.array([4], dtype=np.int32))
    assert int(schedule.dropped_assignments) == 2
    assert bool(schedule.overflow)


@pytest.mark.parametrize(
    ("implementation_name", "symmetric_collectives", "one_shot", "nccl_barrier", "device_kernel"),
    [
        ("ONE_SHOT", None, "true", "true", "false"),
        ("DEVICE", "raggedalltoall", "false", "false", "true"),
    ],
)
def test_runtime_environment_selects_one_ragged_all_to_all_implementation(
    implementation_name: str,
    symmetric_collectives: str | None,
    one_shot: str,
    nccl_barrier: str,
    device_kernel: str,
):
    implementation = getattr(train.RaggedAllToAllImplementation, implementation_name)
    existing = {
        "XLA_FLAGS": " ".join(
            [
                "--unrelated=true",
                "--xla_gpu_experimental_enable_nccl_symmetric_buffers=stale",
                "--xla_gpu_ragged_all_to_all_mode=stale",
                "--xla_enable_nccl_symmetric_buffers_for_collectives=stale",
                "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=stale",
                "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=stale",
                "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=stale",
                "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=stale",
            ]
        )
    }

    environment = train.runtime_environment(existing, implementation)

    flags = environment["XLA_FLAGS"].split()
    assert "--unrelated=true" in flags
    assert "--xla_gpu_experimental_enable_nccl_symmetric_buffers=false" in flags
    assert "--xla_gpu_ragged_all_to_all_mode=private" in flags
    assert f"--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel={one_shot}" in flags
    assert f"--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl={nccl_barrier}" in flags
    assert f"--xla_gpu_experimental_ragged_all_to_all_use_device_kernel={device_kernel}" in flags
    assert "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=false" in flags
    assert "--xla_gpu_enable_latency_hiding_scheduler=false" in flags
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=1" in flags
    targeted = [flag for flag in flags if flag.startswith("--xla_enable_nccl_symmetric_buffers_for_collectives=")]
    assert targeted == (
        []
        if symmetric_collectives is None
        else [f"--xla_enable_nccl_symmetric_buffers_for_collectives={symmetric_collectives}"]
    )
    assert sum(flag.startswith("--xla_gpu_experimental_enable_nccl_symmetric_buffers=") for flag in flags) == 1
    assert sum(flag.startswith("--xla_gpu_ragged_all_to_all_mode=") for flag in flags) == 1
    assert sum(flag.startswith("--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=") for flag in flags) == 1
    assert sum(flag.startswith("--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=") for flag in flags) == 1
    assert sum(flag.startswith("--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=") for flag in flags) == 1
    assert sum(flag.startswith("--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=") for flag in flags) == 1


def test_run_grug_pins_the_xla_nightly_for_training(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "")
    for name in train.MOK_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="nightly-run"),
            ragged_all_to_all_implementation=train.RaggedAllToAllImplementation.DEVICE,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=train.MOK_JAX_PACKAGES,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        train.run_grug(config)

    assert dispatch.call_args.kwargs["pip_packages"] == train.MOK_JAX_PACKAGES
    assert dispatch.call_args.kwargs["max_retries_failure"] == 0
    assert all("0.11.1.dev20260809" in package for package in train.MOK_JAX_PACKAGES)
    assert all("#sha256=" in package for package in train.MOK_JAX_PACKAGES)


def test_train_entrypoint_rejects_a_different_jax_build(monkeypatch):
    monkeypatch.setattr(train.importlib.metadata, "version", lambda _: "0.11.0")

    with pytest.raises(RuntimeError, match=r"requires JAX nightly 0\.11\.1\.dev20260809"):
        train._run_mok_local(object())


def test_grug_dispatch_adds_requested_pip_packages():
    client = MagicMock()
    client.submit.return_value.wait.return_value = None

    with patch.object(grug_dispatch, "current_client", return_value=client):
        grug_dispatch.dispatch_grug_training_run(
            run_id="pip-override",
            config=object(),
            local_entrypoint=lambda _: None,
            resources=ResourceConfig.with_cpu(),
            pip_packages=("jax@https://example.test/jax.whl#sha256=1234",),
        )

    request = client.submit.call_args.args[0]
    assert request.environment.pip_packages == ["jax@https://example.test/jax.whl#sha256=1234"]


def test_experiment_model_uses_ragged_all_to_all():
    assert heuristic.MOK_MODEL.moe_implementation == "ragged_all_to_all"
    assert heuristic.MOK_MODEL.ragged_all_to_all_splits_per_peer == 32


@pytest.mark.parametrize(("num_nodes", "expected_batch_size"), [(1, 64), (2, 128)])
def test_gate_keeps_the_per_gpu_batch_constant(num_nodes: int, expected_batch_size: int):
    step = launch.build_mok_run(
        run_id=f"batch-{num_nodes}",
        num_steps=1,
        execution=launch.MokExecution.XLA,
        implementation=train.RaggedAllToAllImplementation.ONE_SHOT,
        num_nodes=num_nodes,
        version="dev",
    )

    config = json.loads(step.fingerprint_payload())
    assert config["trainer"]["trainer"]["train_batch_size"] == expected_batch_size
    profiler = config["trainer"]["trainer"]["profiler"]
    assert profiler["start_step"] == 5
    assert profiler["num_steps"] == 5
    assert profiler["profile_options"]["enable_hlo_proto"] is True


def test_expert_bank_must_divide_requested_topology():
    with pytest.raises(ValueError, match="expert axis 4 must divide num_experts=10"):
        launch.build_mok_run(
            run_id="bad-bank",
            num_steps=1,
            execution=launch.MokExecution.FUSED,
            implementation=train.RaggedAllToAllImplementation.ONE_SHOT,
            num_nodes=1,
            num_experts=10,
            version="dev",
        )


def test_fused_gate_records_the_mixture_of_kittens_boundary():
    step = launch.build_mok_run(
        run_id="fused-boundary",
        num_steps=1,
        execution=launch.MokExecution.FUSED,
        implementation=train.RaggedAllToAllImplementation.DEVICE,
        num_nodes=1,
        version="dev",
    )

    config = json.loads(step.fingerprint_payload())
    fused = config["model"]["mixture_of_kittens"]
    assert config["model"]["remat_mode"] == "save_moe"
    assert fused["num_comm_sms"] == 40
    assert fused["bwd_num_comm_sms"] == 28
    assert fused["minibatch_size"] == 4096
    assert fused["macrobatch_size"] == 32768
    assert fused["schedule_capacity_factor"] == 1.1


def test_gate_can_log_optimizer_boundary_norms():
    step = launch.build_mok_run(
        run_id="watch-boundary",
        num_steps=2,
        execution=launch.MokExecution.FUSED,
        implementation=train.RaggedAllToAllImplementation.DEVICE,
        num_nodes=1,
        watch_interval=1,
        version="dev",
    )

    watch = json.loads(step.fingerprint_payload())["trainer"]["trainer"]["watch"]
    assert watch["watch_targets"] == ["grads", "updates", "params"]
    assert watch["interval"] == 1


def test_gate_can_reduce_the_layer_count():
    step = launch.build_mok_run(
        run_id="one-layer",
        num_steps=2,
        execution=launch.MokExecution.FUSED,
        implementation=train.RaggedAllToAllImplementation.DEVICE,
        num_nodes=1,
        num_layers=1,
        version="dev",
    )

    assert json.loads(step.fingerprint_payload())["model"]["num_layers"] == 1


def test_fused_schedule_capacity_adds_headroom_and_expert_padding():
    capacity = schedule_capacity(
        num_tokens=16 * 4096,
        top_k=4,
        num_local_experts=2,
        config=MoKForwardConfig(),
    )

    average_routes = 16 * 4096 * 4
    assert capacity >= average_routes * 1.1 + 2 * 255
    assert capacity % 4096 == 0


def test_fused_gate_rejects_more_than_one_worker():
    with pytest.raises(ValueError, match="requires one four-GPU worker"):
        launch.build_mok_run(
            run_id="fused-multinode",
            num_steps=1,
            execution=launch.MokExecution.FUSED,
            implementation=train.RaggedAllToAllImplementation.DEVICE,
            num_nodes=2,
            version="dev",
        )
