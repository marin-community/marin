# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
import os
import subprocess
import sys
import textwrap
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from click.testing import CliRunner
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, set_mesh, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks import profiler as profiler_lib
from levanter.callbacks.profiler import XprofUploadConfig
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import dev_run, grugmuon_hero, launch, launch_mok, model, train
from experiments.grug.moe_hero_ep import small_scale_abl_launch as abl
from experiments.grug.moe_hero_ep.expert_placement import (
    R9_HOT_COLD_EXPERT_PERMUTATIONS,
    hot_cold_expert_permutation,
)


def test_hero_run_without_shape_overrides_uses_the_selected_model():
    step = launch.build_hero_run(run_id="selected-default", dp_racks=1, num_steps=1, version="dev")
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert (
        config.model.hidden_dim,
        config.model.num_layers,
        config.model.num_experts,
        config.model.intermediate_dim,
        config.model.num_experts_per_token,
        config.model.latent_dim,
        config.model.capacity_factor,
        config.model.pooled_transport_capacity_factor,
        config.model.num_expert_waves,
        config.model.moe_implementation,
        config.trainer.trainer.train_batch_size,
        config.model.max_seq_len,
        config.processes_per_task,
        config.trainer.trainer.mp.param_dtype,
        config.trainer.trainer.mp.compute_dtype,
        config.trainer.master_param_mode,
    ) == (
        6144,
        48,
        384,
        3072,
        8,
        3072,
        1.15,
        1.15,
        3,
        "fixed_pooled_wave_all_to_all",
        1024,
        4096,
        1,
        jnp.bfloat16,
        jnp.bfloat16,
        train.MasterParamMode.FP32_PINNED_HOST,
    )


def test_full_bank_top_k_is_rejected_before_launch():
    # QB routing reads the (k+1)-th logit as its threshold, so a full-bank top-k asks `top_k` for
    # more entries than there are experts. Without this the job dies in the router, which is after
    # the 16-node gang is allocated.
    with pytest.raises(ValueError, match="must be < num_experts"):
        launch.build_hero_run(
            run_id="full-bank",
            dp_racks=1,
            num_steps=1,
            num_experts=128,
            num_experts_per_token=128,
            version="dev",
        )


def test_hot_cold_expert_permutation_balances_contiguous_rank_pairs():
    loads = (100, 90, 80, 70, 40, 30, 20, 10)

    permutation = hot_cold_expert_permutation(loads)
    pair_loads = tuple(loads[permutation[i]] + loads[permutation[i + 1]] for i in range(0, len(loads), 2))

    assert permutation == (0, 7, 1, 6, 2, 5, 3, 4)
    assert pair_loads == (110, 110, 110, 110)


def test_r9_hot_cold_placement_relabels_router_and_expert_weights_together():
    num_experts, hidden_dim, intermediate_dim = 128, 2, 3
    cfg = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_shared_experts=2,
        num_experts=num_experts,
        num_experts_per_token=2,
        num_layers=48,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=2,
        sliding_window=2,
        moe_implementation="mok",
        mok_expert_placement="r9_profile_hot_cold",
        mok_minibatch_size=256,
        mok_macrobatch_size=256,
    )
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    key = jax.random.key(0)
    permutation = jnp.asarray(R9_HOT_COLD_EXPERT_PERMUTATIONS[0], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        baseline = jax.jit(lambda init_key: model.MoEMLP.init(cfg, key=init_key))(key)
        placed = jax.jit(
            lambda init_key, expert_permutation: model.MoEMLP.init(
                cfg,
                key=init_key,
                expert_permutation=expert_permutation,
            )
        )(key, permutation)
    permutation_np = np.asarray(permutation)

    np.testing.assert_array_equal(
        np.asarray(placed.router),
        np.asarray(baseline.router)[:, permutation_np],
    )
    np.testing.assert_array_equal(
        np.asarray(placed.router_bias),
        np.asarray(baseline.router_bias)[permutation_np],
    )
    for actual, original in (
        (placed.expert_mlp.w_gate, baseline.expert_mlp.w_gate),
        (placed.expert_mlp.w_up, baseline.expert_mlp.w_up),
        (placed.expert_mlp.w_down, baseline.expert_mlp.w_down),
    ):
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.asarray(original)[permutation_np],
        )

    x = np.asarray(jax.random.normal(jax.random.key(2), (5, hidden_dim)))
    original_logits = x @ np.asarray(baseline.router) + np.asarray(baseline.router_bias)
    placed_logits = x @ np.asarray(placed.router) + np.asarray(placed.router_bias)
    original_top1 = np.argmax(original_logits, axis=-1)
    placed_top1 = np.argmax(placed_logits, axis=-1)
    mapped_top1 = permutation_np[placed_top1]
    np.testing.assert_array_equal(mapped_top1, original_top1)


def test_checkpoint_path_overrides_the_step_output_path():
    """A run that only exercises the checkpoint write sends it to disposable storage."""
    temp_path = "s3://marin-us-east-02a/tmp/ttl=1d/hero-ckpt-smoke"
    step = launch.build_hero_run(
        run_id="ckpt-elsewhere",
        dp_racks=1,
        num_steps=1,
        save_checkpoints=True,
        checkpoint_path=temp_path,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.trainer.checkpointer.base_path == temp_path


def test_checkpoint_path_defaults_under_the_step_output_path():
    step = launch.build_hero_run(run_id="ckpt-default", dp_racks=1, num_steps=1, version="dev")
    ctx = StepContext.for_fingerprint(step.runtime_args, step.deps)
    config = step.build_config(ctx)

    assert config.trainer.trainer.checkpointer.base_path == f"{ctx.output_path}/checkpoints"


def test_checkpoint_interval_must_be_positive():
    with pytest.raises(ValueError, match="checkpoint_interval must be positive"):
        launch.build_hero_run(
            run_id="bad-checkpoint-interval",
            dp_racks=1,
            num_steps=1,
            checkpoint_interval=timedelta(0),
            version="dev",
        )


@pytest.mark.parametrize(
    ("profile_steps", "profile_start_step"),
    [(-1, 0), (1, -1), (1, 3)],
)
def test_profile_window_must_fall_inside_the_run(profile_steps, profile_start_step):
    with pytest.raises(ValueError, match="profile"):
        launch.build_hero_run(
            run_id="bad-profile-window",
            dp_racks=1,
            num_steps=3,
            profile_steps=profile_steps,
            profile_start_step=profile_start_step,
            version="dev",
        )


def test_data_parallel_racks_keep_the_global_batch_explicit():
    step = launch.build_hero_run(run_id="two-racks", dp_racks=2, num_steps=1, version="dev")
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.replica_axis_size == 2
    assert config.trainer.trainer.train_batch_size == launch.HERO_EP_BATCH_SIZE
    assert step.runtime_args["train_resources"].replicas == 2 * launch.HERO_EP_NODES


def test_schedule_steps_do_not_extend_the_run():
    step = launch.build_hero_run(
        run_id="schedule-head",
        dp_racks=1,
        num_steps=5,
        schedule_steps=100,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.trainer.num_train_steps == 100
    assert config.stop_after_steps == 5


def test_expert_bank_override_must_be_divisible_by_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must be divisible by 64"):
        launch.build_hero_run(run_id="bad-bank", dp_racks=1, num_steps=1, num_experts=200, version="dev")


def test_expert_bank_override_must_support_three_waves():
    with pytest.raises(ValueError, match="local expert count=4 must be divisible by num_expert_waves=3"):
        launch.build_hero_run(run_id="bad-waves", dp_racks=1, num_steps=1, num_experts=256, version="dev")


def test_hero_comparison_launches_expose_legacy_and_matched_process_topologies():
    builders = (
        (launch.build_hero_run, {"dp_racks": 1}, 1, "fixed_pooled_wave_all_to_all", "capacity-1.15"),
        (launch.build_multiprocess_hero_run, {}, 4, "fixed_pooled_wave_all_to_all", "capacity-1.15"),
        (
            launch.build_mok_hero_run,
            {"mok_package": "mixture-of-kittens @ https://example.test/mok.whl"},
            4,
            "mok",
            "dropless",
        ),
    )
    for builder, extra_args, expected_processes, expected_backend, expected_semantics in builders:
        step = builder(run_id=builder.__name__, num_steps=1, version="dev", **extra_args)
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))
        assert config.processes_per_task == expected_processes
        assert config.comparison.backend == expected_backend
        assert config.comparison.routing_semantics == expected_semantics
        assert config.comparison.software_environment == "torch-cu130-cublas13.2"
        assert config.comparison.metric_contract == (
            "train/loss,throughput/tokens_per_second,throughput/mfu,moe/drop_fraction"
        )
        if expected_backend == "mok":
            assert config.runtime_pip_packages == ("mixture-of-kittens @ https://example.test/mok.whl",)
        else:
            assert config.runtime_pip_packages == ()


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--mok-fwd-num-comm-sms", "32"),
        ("--mok-bwd-num-comm-sms", "20"),
        ("--mok-minibatch-size", "8192"),
        ("--mok-macrobatch-size", "262144"),
        ("--mok-schedule-capacity-multiplier", "0.5"),
        ("--mok-all-gather-top-experts-chunk-bytes", "4096"),
        ("--mok-expert-placement", "r9_profile_hot_cold"),
    ),
)
def test_fixed_dev_run_rejects_mok_overrides(option, value):
    result = CliRunner().invoke(
        dev_run.main,
        [
            "--run-id",
            "fixed-mok-override",
            "--backend",
            "fixed",
            option,
            value,
        ],
    )

    assert result.exit_code == 2
    assert f"{option} is only valid with --backend mok" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    (
        (("--mok-fwd-num-comm-sms", "3"), "must be divisible by 2"),
        (("--mok-bwd-num-comm-sms", "3"), "must be divisible by 2"),
        (("--mok-minibatch-size", "1025"), "must be divisible by 256"),
        (("--mok-all-gather-top-experts-chunk-bytes", "17"), "must be divisible by 16"),
        (("--num-steps", "100", "--stop-after-steps", "101"), "cannot exceed --num-steps"),
        (("--profile-all-processes",), "requires --profile-steps"),
    ),
)
def test_dev_run_rejects_invalid_sweep_values_before_materializing(args, message):
    result = CliRunner().invoke(dev_run.main, ["--run-id", "invalid-sweep", *args])

    assert result.exit_code == 2
    assert message in result.output


def test_dev_run_rejects_macrobatch_incompatible_with_effective_minibatch(monkeypatch):
    def materialize_for_test(step, _prefix):
        return step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    monkeypatch.setattr(dev_run, "materialized_config", materialize_for_test)
    result = CliRunner().invoke(
        dev_run.main,
        [
            "--run-id",
            "invalid-batches",
            "--mok-minibatch-size",
            "8192",
            "--mok-macrobatch-size",
            "100000",
        ],
    )

    assert result.exit_code == 2
    assert "must be a multiple of the effective MoK minibatch size" in result.output


def test_mok_dev_run_applies_sweep_overrides_without_shortening_optimizer_horizon(monkeypatch):
    captured = {}

    def materialize_for_test(step, _prefix):
        return step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    def run_for_test(config, *, stop_after_steps):
        captured["config"] = config
        captured["stop_after_steps"] = stop_after_steps

    monkeypatch.setattr(dev_run, "materialized_config", materialize_for_test)
    monkeypatch.setattr(dev_run, "_apply_hero_ep_runtime_defaults", lambda **_: None)
    monkeypatch.setattr(dev_run, "_run_grug_local", run_for_test)

    result = CliRunner().invoke(
        dev_run.main,
        [
            "--run-id",
            "mok-sweep",
            "--num-steps",
            "100",
            "--stop-after-steps",
            "25",
            "--profile-start-step",
            "22",
            "--profile-steps",
            "3",
            "--profile-all-processes",
            "--mok-expert-placement",
            "r9_profile_hot_cold",
            "--mok-fwd-num-comm-sms",
            "32",
            "--mok-bwd-num-comm-sms",
            "20",
            "--mok-minibatch-size",
            "8192",
            "--mok-macrobatch-size",
            "262144",
            "--mok-schedule-capacity-multiplier",
            "1.0",
            "--mok-all-gather-top-experts-chunk-bytes",
            "4096",
        ],
    )

    assert result.exit_code == 0, result.exception
    config = captured["config"]
    assert config.trainer.trainer.num_train_steps == 100
    assert captured["stop_after_steps"] == 25
    assert config.trainer.trainer.profiler.process_index is None
    assert config.model.mok_expert_placement == "r9_profile_hot_cold"
    assert config.model.mok_fwd_num_comm_sms == 32
    assert config.model.mok_bwd_num_comm_sms == 20
    assert config.model.mok_minibatch_size == 8192
    assert config.model.mok_macrobatch_size == 262_144
    assert config.model.mok_schedule_capacity_multiplier == 1.0
    assert config.model.mok_all_gather_top_experts_chunk_bytes == 4096
    assert config.runtime_pip_packages == ()


@pytest.mark.parametrize(
    ("command", "args"),
    (
        (
            launch_mok.main,
            (
                "--run-id",
                "profile-cli",
                "--mok-package",
                "mixture-of-kittens",
                "--profile-start-step",
                "22",
                "--profile-steps",
                "3",
                "--version",
                "dev",
            ),
        ),
        (
            dev_run.main,
            (
                "--run-id",
                "profile-cli",
                "--profile-start-step",
                "22",
                "--profile-steps",
                "3",
            ),
        ),
    ),
)
def test_mok_profile_cli_options_parse_a_bounded_capture(command, args):
    ctx = command.make_context(command.name, list(args))

    assert ctx.params["profile_start_step"] == 22
    assert ctx.params["profile_steps"] == 3


def test_mok_profile_window_drives_a_single_process_host_trace(tmp_path, monkeypatch):
    step = launch.build_mok_hero_run(
        run_id="profile-window",
        num_steps=25,
        mok_package="mixture-of-kittens",
        profile_start_step=22,
        profile_steps=3,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))
    profile_config = dataclasses.replace(
        config.trainer.trainer.profiler,
        upload=XprofUploadConfig(enabled=False),
    )
    start_trace = patch.object(profiler_lib.jax.profiler, "start_trace")
    stop_trace = patch.object(profiler_lib.jax.profiler, "stop_trace")
    monkeypatch.setattr(profiler_lib.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_lib, "barrier_sync", lambda: None)

    with start_trace as mocked_start, stop_trace as mocked_stop:
        callback = profile_config.build(str(tmp_path), run_id="profile-window")
        callback(SimpleNamespace(step=20))
        assert not mocked_start.called

        callback(SimpleNamespace(step=21))
        assert mocked_start.call_count == 1
        options = mocked_start.call_args.kwargs["profiler_options"]
        assert options.enable_hlo_proto

        callback(SimpleNamespace(step=24))
        assert mocked_stop.call_count == 1


def test_mok_profiler_is_disabled_when_no_profile_window_is_requested():
    step = launch.build_mok_hero_run(
        run_id="profile-disabled",
        num_steps=1,
        mok_package="mixture-of-kittens",
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert not config.trainer.trainer.profiler.is_enabled


def test_mok_fuses_routed_and_two_shared_experts_in_one_dropless_call(monkeypatch):
    cfg = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=3,
        shared_expert_intermediate_dim=3,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=2,
        sliding_window=2,
        moe_implementation="mok",
        mok_minibatch_size=256,
        mok_macrobatch_size=256,
    )
    expert_mlp = model.MoEExpertMlp(
        w_gate=jnp.ones((4, 4, 3)),
        w_up=jnp.ones((4, 4, 3)) * 2,
        w_down=jnp.ones((4, 3, 4)) * 3,
        implementation="mok",
        activation=model.ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        expert_chunks=1,
    )
    mlp = model.MoEMLP(
        router=jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10,
        router_bias=jnp.zeros(4),
        expert_mlp=expert_mlp,
        w_latent_down=None,
        latent_norm=None,
        w_latent_up=None,
        cfg=cfg,
    )
    shared = (
        model.DenseMLP(jnp.ones((4, 3)) * 4, jnp.ones((4, 3)) * 5, jnp.ones((3, 4)) * 6),
        model.DenseMLP(jnp.ones((4, 3)) * 7, jnp.ones((4, 3)) * 8, jnp.ones((3, 4)) * 9),
    )
    calls = []

    def fake_mok(x, selected_experts, router_weights, *weights, **config):
        calls.append((selected_experts, router_weights, weights, config))
        return jnp.ones_like(x) * sum(weight.mean() for weight in weights)

    monkeypatch.setattr(model, "_mok_bf16", fake_mok)
    mesh = Mesh(
        np.asarray(jax.devices()).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        actual, router_stats = mlp(jnp.ones((1, 2, 4), dtype=jnp.bfloat16), shared_experts=shared)

    np.testing.assert_array_equal(np.asarray(actual), np.full((1, 2, 4), 45.0))
    assert int(router_stats["capacity_overflow"]) == 0
    assert len(calls) == 1
    _selected, router_weights, weights, adapter_config = calls[0]
    assert len(weights) == 9
    assert router_weights.dtype == jnp.float32
    assert not np.array_equal(
        np.asarray(router_weights),
        np.asarray(router_weights.astype(jnp.bfloat16).astype(jnp.float32)),
    )
    np.testing.assert_allclose(np.asarray(router_weights.sum(axis=-1)), 2.5, atol=1e-2)
    assert adapter_config["macrobatch_size"] == 256


def test_mok_with_latent_dispatches_the_compressed_token_and_leaves_shared_experts_to_the_block(monkeypatch):
    # The kernel routes and runs its fused shared experts at one token width. The hero's shared
    # experts read the full-width token, so with a latent projection they cannot ride along: the
    # call has to carry routed traffic only, at the latent width, and the block adds the shared
    # experts the same way it does for every all-to-all backend.
    hidden_dim, latent_dim, intermediate_dim = 512, 256, 3
    cfg = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=2,
        sliding_window=2,
        latent_dim=latent_dim,
        moe_implementation="mok",
        mok_minibatch_size=256,
        mok_macrobatch_size=256,
    )
    assert not cfg.fuses_shared_experts

    expert_mlp = model.MoEExpertMlp(
        w_gate=jnp.ones((4, latent_dim, intermediate_dim), dtype=jnp.bfloat16),
        w_up=jnp.ones((4, latent_dim, intermediate_dim), dtype=jnp.bfloat16),
        w_down=jnp.ones((4, intermediate_dim, latent_dim), dtype=jnp.bfloat16),
        implementation="mok",
        activation=model.ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        expert_chunks=1,
    )
    mlp = model.MoEMLP(
        router=jnp.zeros((hidden_dim, 4), dtype=jnp.float32),
        router_bias=jnp.zeros(4),
        expert_mlp=expert_mlp,
        w_latent_down=jnp.ones((hidden_dim, latent_dim), dtype=jnp.bfloat16),
        latent_norm=model.RMSNorm.init(latent_dim, cfg.layer_norm_eps),
        w_latent_up=jnp.full((latent_dim, hidden_dim), 2.0, dtype=jnp.bfloat16),
        cfg=cfg,
    )
    shared = (
        model.DenseMLP(
            jnp.ones((hidden_dim, intermediate_dim)),
            jnp.ones((hidden_dim, intermediate_dim)),
            jnp.ones((intermediate_dim, hidden_dim)),
        ),
    ) * 2
    captured = {}

    def fake_mok(x, selected_experts, router_weights, *weights, **config):
        captured["x"] = x
        captured["weights"] = weights
        return jnp.ones_like(x)

    monkeypatch.setattr(model, "_mok_bf16", fake_mok)
    mesh = Mesh(
        np.asarray(jax.devices()).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        actual, _router_stats = mlp(jnp.ones((1, 2, hidden_dim), dtype=jnp.bfloat16), shared_experts=shared)

    # Dispatch carries the compressed row, not the full-width token.
    assert captured["x"].shape == (2, latent_dim)
    # The fused shared slot is empty, and the routed weights still arrive.
    assert captured["weights"][:6] == (None,) * 6
    assert all(weight is not None for weight in captured["weights"][6:])
    # The combine is expanded back to hidden width before it leaves the layer.
    assert actual.shape == (1, 2, hidden_dim)
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), 2.0 * latent_dim, rtol=1e-2)


def test_mok_hero_arm_carries_the_hero_latent_width():
    step = launch.build_mok_hero_run(
        run_id="mok-latent",
        num_steps=1,
        mok_package="mixture-of-kittens",
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert config.model.moe_implementation == "mok"
    assert config.model.latent_dim == launch.HERO_MODEL.latent_dim
    assert not config.model.fuses_shared_experts
    assert config.comparison.fused_shared_experts == 0


def test_mok_hero_arm_without_latent_keeps_the_shared_experts_fused():
    step = launch.build_mok_hero_run(
        run_id="mok-no-latent",
        num_steps=1,
        mok_package="mixture-of-kittens",
        latent_dim=0,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert config.model.latent_dim is None
    assert config.model.fuses_shared_experts
    assert config.comparison.fused_shared_experts == config.model.num_shared_experts


def test_mok_and_pooled_wave_arms_differ_only_in_the_moe_backend():
    """The comparison is only worth running if the two arms are the same model."""
    mok = launch.build_mok_hero_run(
        run_id="mok-arm",
        num_steps=1,
        mok_package="mixture-of-kittens",
        version="dev",
    )
    wave = launch.build_multiprocess_hero_run(run_id="wave-arm", num_steps=1, version="dev")
    mok_model = mok.build_config(StepContext.for_fingerprint(mok.runtime_args.keys(), mok.deps)).model
    wave_model = wave.build_config(StepContext.for_fingerprint(wave.runtime_args.keys(), wave.deps)).model

    differing = {
        field.name
        for field in dataclasses.fields(mok_model)
        if getattr(mok_model, field.name) != getattr(wave_model, field.name)
    }
    assert differing == {"moe_implementation"}


def test_latent_width_must_suit_the_mok_workspace():
    with pytest.raises(ValueError, match="latent_dim divisible by 256"):
        model.GrugModelConfig(
            vocab_size=32,
            hidden_dim=512,
            intermediate_dim=3,
            shared_expert_intermediate_dim=3,
            num_shared_experts=2,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            max_seq_len=2,
            sliding_window=2,
            latent_dim=300,
            moe_implementation="mok",
            mok_minibatch_size=256,
            mok_macrobatch_size=256,
        )


def test_run_grug_applies_ep_xla_defaults_and_keeps_explicit_values(monkeypatch):
    explicit_overlap = "--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    monkeypatch.setenv("XLA_FLAGS", explicit_overlap)
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=1,
        runtime_pip_packages=(),
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert explicit_overlap in flags
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=4" not in flags
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" in flags
    assert train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG in flags
    assert os.environ["JAX_ENABLE_PGLE"] == "false"
    assert os.environ["XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"] == "192"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"


def test_run_grug_defaults_pgle_off_for_per_gpu_processes(monkeypatch):
    # Per-GPU processes cannot profile: the per-process CUPTI sessions collide with
    # each other and with the cluster's DCGM, and auto-PGLE's recompile path has
    # wedged per-node gangs (#7344). Per-GPU runs therefore default PGLE off, while
    # an explicit env setting still wins.
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=4,
        runtime_pip_packages=(),
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "false"

    monkeypatch.setenv("JAX_ENABLE_PGLE", "true")
    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)
    assert os.environ["JAX_ENABLE_PGLE"] == "true"


def test_run_grug_keeps_explicit_ep_runtime_values(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_PGLE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=1,
        runtime_pip_packages=(),
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


@pytest.mark.parametrize(
    ("watch_mode", "watch_interval", "expected_overlap_limit"),
    [
        (train.WatchMode.INLINE, 1, train.INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT),
        (train.WatchMode.DIAGNOSTIC, 1, train.DEFAULT_COLLECTIVE_OVERLAP_LIMIT),
        (train.WatchMode.INLINE, 0, train.DEFAULT_COLLECTIVE_OVERLAP_LIMIT),
    ],
)
def test_run_grug_reduces_collective_overlap_only_for_inline_watch(
    monkeypatch, watch_mode, watch_interval, expected_overlap_limit
):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=watch_interval)),
            watch_mode=watch_mode,
        ),
        resources=object(),
        processes_per_task=1,
        runtime_pip_packages=(),
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert f"{train.XLA_COLLECTIVE_OVERLAP_FLAG}={expected_overlap_limit}" in flags


def test_ep_newton_schulz_returns_to_expert_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    input_sharding = NamedSharding(mesh, P(None, "expert", None, None))
    x = jax.ShapeDtypeStruct((48, 256, 8, 4), jnp.float32, sharding=input_sharding)

    def apply_ns(y):
        path = (jax.tree_util.GetAttrKey("w_gate"),)
        return grugmuon_hero._newtonschulz_4d_distributed(
            path,
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
            use_syrk=False,
        )

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(apply_ns, x)

    assert output.sharding == NamedSharding(mesh, P(None, "expert", "data", "model"))


def test_ep_newton_schulz_matches_replicated_path():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from experiments.grug.moe_hero_ep.grugmuon_hero import (
            _newtonschulz_4d_distributed,
            _zeropower_via_newtonschulz_replicated,
        )

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 2, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        x = jax.random.normal(jax.random.key(0), (1, 2, 4, 2), dtype=jnp.float32)
        x_sharded = jax.device_put(x, NamedSharding(mesh, P(None, "expert", "data", "model")))
        path = (jax.tree_util.GetAttrKey("w_gate"),)
        expected = jax.vmap(
            jax.vmap(
                lambda matrix: _zeropower_via_newtonschulz_replicated(
                    matrix, steps=1, eps=1e-7, coefficient_type="quintic"
                )
            )
        )(x)

        apply_ns = jax.jit(
            lambda y: _newtonschulz_4d_distributed(
                path,
                y,
                steps=1,
                eps=1e-7,
                coefficient_type="quintic",
                use_syrk=False,
            )
        )
        with jax.set_mesh(mesh):
            actual = apply_ns(x_sharded)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ep_padded_newton_schulz_returns_to_parameter_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    parameter_sharding = NamedSharding(mesh, P(None, "expert", None))
    x = jax.ShapeDtypeStruct((48, 64, 4), jnp.float32, sharding=parameter_sharding)

    def apply_ns(y):
        return grugmuon_hero._newtonschulz_padded_stack_sharded(
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
            target_sharding=parameter_sharding,
        )

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(apply_ns, x)

    assert output.sharding == parameter_sharding


def test_dropless_local_transform_swaps_moe_backend_and_shares_weights():
    # The dropless eval transform must retarget only the static MoE backend fields and keep every
    # weight leaf shared by identity, so the eval scores the trained weights with no capacity drops.
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )
    with set_mesh(mesh):
        m = model.Transformer.init(cfg, key=jax.random.key(0))
    dropless = train._to_dropless_local(m)

    original = m.stacked_blocks.stacked.mlp.expert_mlp
    swapped = dropless.stacked_blocks.stacked.mlp.expert_mlp
    assert original.implementation == "fixed_all_to_all"  # input model left untouched
    assert swapped.implementation == "sonic_cute"
    assert swapped.expert_chunks == 1
    orig_leaves = jax.tree_util.tree_leaves(original)
    swapped_leaves = jax.tree_util.tree_leaves(swapped)
    assert len(orig_leaves) == len(swapped_leaves)
    assert all(a is b for a, b in zip(orig_leaves, swapped_leaves, strict=True))


def test_eval_every_adds_the_held_out_suites_as_dependencies():
    # Held-out sets are what make a run scoreable; a throughput-only run should not pay for them.
    off = launch.build_hero_run(run_id="eval-off", dp_racks=1, num_steps=1, version="dev")
    on = launch.build_hero_run(run_id="eval-on", dp_racks=1, num_steps=1, eval_every=50, version="dev")
    off_config = off.build_config(StepContext.for_fingerprint(off.runtime_args, off.deps))
    on_config = on.build_config(StepContext.for_fingerprint(on.runtime_args, on.deps))

    assert len(off.deps) == 1
    assert len(on.deps) > len(off.deps)
    assert off_config.eval is None
    assert on_config.eval is not None
    assert on_config.eval.steps_per_eval == 50


def test_ep_ablation_defaults_match_the_documented_arm_and_scale_per_rack():
    one = abl.build_small_run(run_id="d768", size="d768", flavor="ep", version="dev")
    cfg = one.build_config(StepContext.for_fingerprint(one.runtime_args, one.deps))
    m = cfg.model
    # The EP rung is a downsized hero: pooled-wave transport, 384 experts / top-8, hidden/2-wide experts
    # in a hidden/2 latent, receiver/sender capacity 1.15 with 3 waves, and top-k QB (the hero default).
    assert m.moe_implementation == "fixed_pooled_wave_all_to_all"
    assert (m.num_experts, m.num_experts_per_token) == (384, 8)
    assert m.intermediate_dim == m.hidden_dim // 2
    assert m.latent_dim == m.hidden_dim // 2
    assert m.capacity_factor == 1.15
    assert m.pooled_transport_capacity_factor == 1.15
    assert m.num_expert_waves == 3
    assert m.qb_estimator == model.QbEstimator.TOPK
    assert m.num_layers % 2 == 0  # even depth applied in the launcher, not GrugModelConfig
    # The histogram QB estimator is selectable on through the builder.
    hist = abl.build_small_run(run_id="d768-hist", size="d768", flavor="ep", qb_use_histogram=True, version="dev")
    assert hist.build_config(StepContext.for_fingerprint(hist.runtime_args, hist.deps)).model.qb_estimator == (
        model.QbEstimator.HIST
    )
    # The global batch scales with the rack count, holding the per-rack token load constant.
    four = abl.build_small_run(run_id="d2048", size="d2048", flavor="ep", dp_racks=4, version="dev")
    four_cfg = four.build_config(StepContext.for_fingerprint(four.runtime_args, four.deps))
    assert four_cfg.trainer.trainer.train_batch_size == cfg.trainer.trainer.train_batch_size * 4


def test_odd_depth_config_is_not_silently_rounded():
    # GrugModelConfig must preserve an odd depth so HF round-trips and odd configs stay faithful;
    # even-rounding is the launcher's job.
    cfg = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
    )
    assert cfg.num_layers == 3


def test_hybrid_kv_branches_agree_on_sharding_when_model_axis_is_wide():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = """
        import math

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, set_mesh

        from experiments.grug.moe_hero_ep import model

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 2, 2),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        cfg = model.GrugModelConfig(
            vocab_size=128,
            hidden_dim=32,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_shared_experts=1,
            num_experts=4,
            num_experts_per_token=1,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            local_kv_heads=2,
            global_kv_heads=1,
            head_dim=8,
            max_seq_len=8,
            sliding_window=4,
            global_every=2,
            capacity_factor=1.0,
            initializer_std=0.5 / math.sqrt(32),
            qk_mult=1.3,
            attention_implementation="reference",
            moe_implementation="fixed_all_to_all",
            report_capacity_overflow=True,
        )
        tokens = jax.ShapeDtypeStruct((2, 8), jnp.int32)
        with set_mesh(mesh):
            output = jax.eval_shape(
                lambda token_ids: model.Transformer.init(cfg, key=jax.random.key(0))(token_ids)[0],
                tokens,
            )
        assert output.shape == (2, 8, 32)
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _explicit_mesh(*axis_sizes):
    return Mesh(
        np.asarray(jax.devices()).reshape(*axis_sizes),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _latent_config(latent_dim=None):
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        latent_dim=latent_dim,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )


def test_latent_moe_shrinks_the_dispatched_width_but_not_the_token():
    # The point of LatentMoE is that the all-to-all payload narrows while the residual stream does
    # not, so the expert weights must be latent-wide and the layer output hidden-wide.
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=16)
    tokens = jax.ShapeDtypeStruct((1, 8), jnp.int32)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
        out = jax.eval_shape(lambda t: model.Transformer.init(cfg, key=jax.random.key(0))(t)[0], tokens)

    assert built.w_latent_down.shape == (cfg.hidden_dim, 16)
    # Normalizing the latent keeps the expert input at unit scale despite the down-projection.
    assert built.latent_norm.weight.shape == (16,)
    assert built.w_latent_up.shape == (16, cfg.hidden_dim)
    # Expert banks are latent-wide: this is what narrows the dispatch.
    assert built.expert_mlp.w_gate.shape[1] == 16
    assert built.expert_mlp.w_up.shape[1] == 16
    assert built.expert_mlp.w_down.shape[2] == 16
    # The residual stream is untouched.
    assert out.shape[-1] == cfg.hidden_dim


def test_latent_moe_is_absent_by_default():
    # A config without a latent width must keep the standard MoE layer.
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=None)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
    assert built.w_latent_down is None and built.w_latent_up is None
    assert built.latent_norm is None
    assert built.expert_mlp.w_gate.shape[1] == cfg.hidden_dim


def test_latent_moe_hf_config_roundtrip_preserves_the_architecture():
    cfg = _latent_config(latent_dim=16)

    hf_config = cfg.to_hf_config(cfg.vocab_size)
    roundtripped = model.GrugModelConfig.from_hf_config(hf_config)

    assert hf_config.to_dict()["latent_dim"] == 16
    assert hf_config.to_dict()[model.GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY] == 2
    assert roundtripped.latent_dim == 16


def test_latent_moe_state_dict_contains_the_projection_state():
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=16)
    with set_mesh(mesh):
        built = model.Transformer.init(cfg, key=jax.random.key(0))
        state_dict = built.to_state_dict()
    block = next(iter(built.stacked_blocks.unstacked()))
    assert block.mlp.w_latent_down is not None
    assert block.mlp.latent_norm is not None
    assert block.mlp.w_latent_up is not None

    expected = {
        "model.layers.0.mlp.latent_down_proj.weight": jnp.swapaxes(block.mlp.w_latent_down, -1, -2),
        "model.layers.0.mlp.latent_norm.weight": block.mlp.latent_norm.weight,
        "model.layers.0.mlp.latent_up_proj.weight": jnp.swapaxes(block.mlp.w_latent_up, -1, -2),
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(state_dict[name], value)


def test_latent_dim_above_hidden_is_rejected():
    # A latent wider than the hidden dim adds communication instead of removing it.
    with pytest.raises(ValueError, match="latent_dim must be in"):
        _latent_config(latent_dim=99999)


def test_latent_moe_flops_replace_routed_width_and_add_projections():
    latent_dim = 16
    full_width_config = _latent_config(latent_dim=None)
    latent_config = _latent_config(latent_dim=latent_dim)
    full_width_flops, _ = train._compute_flops(model_config=full_width_config)
    latent_flops, _ = train._compute_flops(model_config=latent_config)

    routed_delta = (
        2
        * 3
        * latent_config.intermediate_dim
        * latent_config.num_experts_per_token
        * (latent_dim - latent_config.hidden_dim)
    )
    projection_flops = 2 * 2 * latent_config.hidden_dim * latent_dim
    expected_delta = 3 * latent_config.max_seq_len * latent_config.num_layers * (routed_delta + projection_flops)

    assert latent_flops - full_width_flops == expected_delta


class _TinyWatchModel(eqx.Module):
    weight: jax.Array

    def next_token_loss(
        self,
        tokens,
        loss_weight,
        *,
        mask,
        reduction,
        logsumexp_weight,
        return_router_metrics,
    ):
        del mask, reduction, logsumexp_weight, return_router_metrics
        error = self.weight * tokens.astype(self.weight.dtype) - loss_weight
        return jnp.mean(error**2), {}


def test_diagnostic_watch_stats_match_direct_gradient_and_parameter_norms():
    params = _TinyWatchModel(weight=jnp.array(2.0))
    batch = SimpleNamespace(
        tokens=jnp.array([1, 3], dtype=jnp.int32),
        loss_weight=jnp.array([0.5, 1.0]),
        attn_mask=None,
    )
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    watch = WatchConfig(interval=1)

    actual = train._compute_diagnostic_watch_stats(params, batch, mp, None, watch)
    grads = jax.grad(
        lambda model: model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=None,
            return_router_metrics=True,
        )[0]
    )(params)
    expected = compute_watch_stats(
        watch_targets=watch.watch_targets,
        include_norms=watch.include_norms,
        include_per_parameter_norms=watch.include_per_parameter_norms,
        include_histogram=watch.include_histograms,
        split_scan_layers=watch.split_scan_layers,
        params=params,
        grads=grads,
        model_tree_type=type(params),
    )

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key])


def test_inline_watch_computes_stats_on_every_train_step(monkeypatch):
    params = _TinyWatchModel(weight=jnp.array(2.0))
    optimizer = optax.sgd(0.1)
    state = train.GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=None,
        opt_state=optimizer.init(params),
        ema_params=None,
        pending_qb_betas=jnp.zeros((1, 1)),
    )

    def loss_and_grads(current_params, batch, mp, z_loss):
        del batch, mp, z_loss
        loss = current_params.weight**2
        grads = _TinyWatchModel(weight=2 * current_params.weight)
        metrics = {"qb_beta_per_layer": jnp.zeros((1, 1))}
        return (loss, metrics), grads

    monkeypatch.setattr(train, "_apply_qb_betas", lambda model, qb_betas: model)
    monkeypatch.setattr(train, "_loss_and_grads", loss_and_grads)
    train_step = train._make_train_step(
        optimizer,
        jmp.get_policy("params=float32,compute=float32,output=float32"),
        z_loss_weight=0,
        ema_beta=None,
        watch_config=WatchConfig(interval=10),
    )

    state, _, step_zero_stats = train_step(state, jnp.array(0))
    state, _, step_one_stats = train_step(state, jnp.array(0))

    assert step_zero_stats is not None
    assert step_one_stats is not None
    np.testing.assert_allclose(step_zero_stats["grad/norm/total"], 4.0)
    np.testing.assert_allclose(step_one_stats["grad/norm/total"], 3.2)


def test_fp32_host_master_accumulates_updates_before_bfloat16_cast(monkeypatch):
    params = _TinyWatchModel(weight=jnp.array(1.0, dtype=jnp.bfloat16))
    master_params = _TinyWatchModel(weight=jnp.array(1.0, dtype=jnp.float32))
    optimizer = optax.sgd(0.1)
    state = train.GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=master_params,
        opt_state=optimizer.init(master_params),
        ema_params=None,
        pending_qb_betas=jnp.zeros((1, 1)),
    )

    def loss_and_grads(current_params, batch, mp, z_loss):
        del current_params, batch, mp, z_loss
        loss = jnp.array(0.0)
        grads = _TinyWatchModel(weight=jnp.array(0.01, dtype=jnp.bfloat16))
        metrics = {"qb_beta_per_layer": jnp.zeros((1, 1))}
        return (loss, metrics), grads

    monkeypatch.setattr(train, "_apply_qb_betas", lambda model, qb_betas: model)
    monkeypatch.setattr(train, "_loss_and_grads", loss_and_grads)
    train_step = train._make_train_step(
        optimizer,
        jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
        z_loss_weight=0,
        ema_beta=None,
        master_param_mode=train.MasterParamMode.FP32_PINNED_HOST,
    )

    for _ in range(10):
        state, _, _ = train_step(state, jnp.array(0))

    assert state.master_params is not None
    assert state.master_params.weight.dtype == jnp.float32
    assert state.params.weight.dtype == jnp.bfloat16
    expected_master = 1.0 - 10 * 0.1 * float(jnp.array(0.01, dtype=jnp.bfloat16))
    np.testing.assert_allclose(state.master_params.weight, expected_master, rtol=1e-6)
    np.testing.assert_allclose(state.params.weight, jnp.asarray(expected_master, dtype=jnp.bfloat16))


def test_fp32_host_master_preserves_float32_initialization(monkeypatch):
    config = _latent_config()
    key = jax.random.key(17)
    mesh = _explicit_mesh(1, 1, 1, 1)
    monkeypatch.setattr(train, "_tree_to_memory_kind", lambda tree, memory_kind: tree)

    with set_mesh(mesh):
        expected = model.Transformer.init(config, key=key)
        state = train.initial_state(
            config,
            optimizer=optax.sgd(0.1),
            mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
            key=key,
            ema_beta=None,
            master_param_mode=train.MasterParamMode.FP32_PINNED_HOST,
        )

    assert state.master_params is not None
    expected_leaves = jax.tree.leaves(expected)
    master_leaves = jax.tree.leaves(state.master_params)
    param_leaves = jax.tree.leaves(state.params)
    for expected_leaf, master_leaf, param_leaf in zip(expected_leaves, master_leaves, param_leaves, strict=True):
        np.testing.assert_array_equal(master_leaf, expected_leaf)
        np.testing.assert_array_equal(param_leaf, expected_leaf.astype(jnp.bfloat16))
    assert any(
        not np.array_equal(master_leaf, param_leaf.astype(jnp.float32))
        for master_leaf, param_leaf in zip(master_leaves, param_leaves, strict=True)
    )


def test_drop_metrics_reports_sender_and_receiver_fractions():
    metrics = train._drop_metrics(
        jnp.array(5, dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        batch_size=2,
        sequence_length=4,
        top_k=2,
        num_layers=1,
    )

    assert metrics == {
        "moe/dropped_assignments": 5,
        "moe/drop_fraction": 5 / 16,
        "moe/sender_dropped_assignments": 2,
        "moe/sender_drop_fraction": 2 / 16,
        "moe/receiver_dropped_assignments": 3,
        "moe/receiver_drop_fraction": 3 / 16,
        "moe/receiver_drop_fraction_of_received": 3 / 14,
    }


def test_drop_metrics_sums_per_layer_counts_in_int64_without_overflow():
    # Per-layer int32 counts whose 48-layer sum exceeds int32 (jax_enable_x64 is off, so an in-device
    # jnp.sum would wrap and break the total==sender+receiver check). The host sum must stay exact.
    num_layers = 48
    per_layer_sender = jnp.full((num_layers,), 40_000_000, dtype=jnp.int32)  # 48 * 40M = 1.92e9
    per_layer_receiver = jnp.full((num_layers,), 60_000_000, dtype=jnp.int32)  # 48 * 60M = 2.88e9 > int32
    per_layer_total = per_layer_sender + per_layer_receiver
    sender_total = 48 * 40_000_000
    receiver_total = 48 * 60_000_000

    metrics = train._drop_metrics(
        per_layer_total,
        per_layer_sender,
        per_layer_receiver,
        batch_size=4096,
        sequence_length=4096,
        top_k=8,
        num_layers=num_layers,
    )

    assert metrics["moe/dropped_assignments"] == sender_total + receiver_total  # no int32 wrap
    assert metrics["moe/sender_dropped_assignments"] == sender_total
    assert metrics["moe/receiver_dropped_assignments"] == receiver_total
