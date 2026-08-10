# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from click.testing import CliRunner
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks import profiler as profiler_lib
from levanter.callbacks.profiler import XprofUploadConfig
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import dev_run, grugmuon_hero, launch, launch_mok, model, train
from experiments.grug.moe_hero_ep.expert_placement import (
    R9_HOT_COLD_EXPERT_PERMUTATIONS,
    hot_cold_expert_permutation,
)


def test_full_bank_top_k_is_rejected_before_launch():
    # QB routing reads the (k+1)-th logit as its threshold, so a full-bank top-k asks `top_k` for
    # more entries than there are experts. Without this the job dies in the router, which is after
    # the 16-node gang is allocated.
    with pytest.raises(ValueError, match="must be < num_experts"):
        launch.build_hero_run(run_id="full-bank", num_steps=1, num_experts_per_token=128, version="dev")


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


def test_expert_bank_override_must_divide_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must divide the expert axis"):
        launch.build_hero_run(run_id="bad-bank", num_steps=1, num_experts=200, version="dev")


def test_hero_comparison_launches_expose_legacy_and_matched_process_topologies():
    builders = (
        (launch.build_hero_run, {}, 1, "fixed_all_to_all", "capacity-1"),
        (launch.build_multiprocess_hero_run, {}, 4, "fixed_all_to_all", "capacity-1"),
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
        (("--profile-all-processes",), "requires --profile-start-step"),
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
    monkeypatch.setattr(dev_run, "_apply_hero_ep_runtime_defaults", lambda: None)
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
            "--profile-num-steps",
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
                "--profile-num-steps",
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
                "--profile-num-steps",
                "3",
            ),
        ),
    ),
)
def test_mok_profile_cli_options_parse_a_bounded_capture(command, args):
    ctx = command.make_context(command.name, list(args))

    assert ctx.params["profile_start_step"] == 22
    assert ctx.params["profile_num_steps"] == 3


def test_mok_profile_window_drives_a_single_process_host_trace(tmp_path, monkeypatch):
    step = launch.build_mok_hero_run(
        run_id="profile-window",
        num_steps=25,
        mok_package="mixture-of-kittens",
        profile_start_step=22,
        profile_num_steps=3,
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
        assert options.host_tracer_level == 1
        assert options.python_tracer_level == 0
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


def test_cuda_profiler_range_brackets_only_the_requested_global_rank_zero_steps(monkeypatch):
    events = []

    class FakeCudaRuntime:
        def __init__(self):
            def start():
                events.append("start")
                return 0

            def stop():
                events.append("stop")
                return 0

            self.cudaProfilerStart = start
            self.cudaProfilerStop = stop

    monkeypatch.setattr(train, "_load_cuda_runtime", lambda: FakeCudaRuntime())
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.jax, "effects_barrier", lambda: events.append("effects_barrier"))

    with train._cuda_profiler_step_range(enabled=True, start_step=80, num_steps=4, current_step=80):
        events.append("step80")
    with train._cuda_profiler_step_range(enabled=True, start_step=80, num_steps=4, current_step=81):
        events.append("step81")
    with train._cuda_profiler_step_range(enabled=True, start_step=80, num_steps=4, current_step=83):
        events.append("step83")

    assert events == ["start", "step80", "step81", "step83", "effects_barrier", "stop"]


@pytest.mark.parametrize(("enabled", "process_index"), ((False, 0), (True, 1)))
def test_cuda_profiler_range_is_inert_when_disabled_or_off_process_zero(monkeypatch, enabled, process_index):
    monkeypatch.setattr(train.jax, "process_index", lambda: process_index)
    monkeypatch.setattr(
        train,
        "_load_cuda_runtime",
        lambda: pytest.fail("CUDA runtime must not load outside an enabled process-zero profile"),
    )

    with train._cuda_profiler_step_range(enabled=enabled, start_step=80, num_steps=4, current_step=80):
        pass


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


def test_run_grug_applies_ep_xla_defaults_and_keeps_explicit_values(monkeypatch):
    explicit_overlap = "--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    monkeypatch.setenv("XLA_FLAGS", explicit_overlap)
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
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
    for name, value in train.HERO_EP_RUNTIME_ENV.items():
        assert os.environ[name] == value


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
