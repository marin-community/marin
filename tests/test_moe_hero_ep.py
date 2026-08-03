# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from fray.cluster import ResourceConfig
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import grugmuon_hero, launch, train
from experiments.grug.moe_hero_ep.jax_wheel_setup import MoonEPJaxWheelBuild
from experiments.grug.moe_hero_ep.quantile_balancing import histogram_quantile_bias, quantile_balancing_routes


def _exact_required_bias_quantile(required_bias: np.ndarray, *, top_k: int) -> np.ndarray:
    target_rank = (required_bias.shape[0] * top_k + required_bias.shape[1] - 1) // required_bias.shape[1]
    target = np.sort(required_bias, axis=0)[target_rank - 1]
    return target - target.mean()


def test_qb_routing_uses_sigmoid_scores_for_bias_and_cutoff():
    router_logits = jnp.array([[10.0, 9.0, 0.0]], dtype=jnp.float32)
    current_bias = jnp.array([0.0, 0.5, -0.25], dtype=jnp.float32)

    router_scores, selected_experts, cutoff = quantile_balancing_routes(
        router_logits,
        current_bias,
        top_k=1,
    )

    np.testing.assert_array_equal(np.asarray(selected_experts), np.array([[1]], dtype=np.int32))
    raw_logit_route = jax.lax.top_k(router_logits + current_bias, 2)[1][:, :1]
    np.testing.assert_array_equal(np.asarray(raw_logit_route), np.array([[0]], dtype=np.int32))
    required_bias = np.asarray(cutoff - router_scores)
    assert np.all(required_bias >= float(jnp.min(current_bias)) - 1.0)
    assert np.all(required_bias <= float(jnp.max(current_bias)) + 1.0)


def test_histogram_qb_converges_from_strong_router_skew():
    num_tokens = 4096
    num_experts = 16
    top_k = 2
    target_load = num_tokens * top_k // num_experts
    router_logits = jax.random.normal(
        jax.random.key(0),
        (num_tokens, num_experts),
        dtype=jnp.float32,
    ) + jnp.linspace(-2.0, 2.0, num_experts)
    bias = jnp.zeros((num_experts,), dtype=jnp.float32)
    load_errors = []

    for _ in range(6):
        router_scores, selected_experts, cutoff = quantile_balancing_routes(
            router_logits,
            bias,
            top_k=top_k,
        )
        expert_loads = jnp.bincount(selected_experts.reshape(-1), length=num_experts)
        load_errors.append(int(jnp.max(jnp.abs(expert_loads - target_load))))
        bias = histogram_quantile_bias(
            cutoff - router_scores,
            bias,
            top_k=top_k,
            num_bins=1000,
            reduce_axes=(),
        )

    assert load_errors[0] > target_load
    assert load_errors[-1] < target_load // 10


def test_histogram_qb_matches_pooled_quantile_within_one_bin():
    required_bias = np.array(
        [
            [-0.85, -0.75, -0.65, -0.55],
            [-0.70, -0.45, -0.20, 0.05],
            [-0.55, -0.15, 0.25, 0.65],
            [-0.40, 0.15, 0.70, -0.80],
            [-0.25, 0.45, -0.75, -0.30],
            [-0.10, 0.75, -0.35, 0.20],
            [0.05, -0.80, 0.05, 0.70],
            [0.20, -0.50, 0.45, -0.60],
            [0.35, -0.20, 0.85, -0.10],
            [0.50, 0.10, -0.55, 0.40],
            [0.65, 0.40, -0.15, 0.90],
            [0.80, 0.70, 0.35, -0.70],
            [0.90, 0.90, 0.75, 0.10],
        ],
        dtype=np.float32,
    )
    current_bias = np.array([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
    num_bins = 32

    actual = histogram_quantile_bias(
        jnp.asarray(required_bias),
        jnp.asarray(current_bias),
        top_k=1,
        num_bins=num_bins,
        reduce_axes=(),
    )
    expected = _exact_required_bias_quantile(required_bias, top_k=1)
    bin_width = (current_bias.max() - current_bias.min() + 2.0) / num_bins

    np.testing.assert_allclose(np.asarray(actual), expected, atol=bin_width, rtol=0)


def test_histogram_qb_uses_pooled_distribution_instead_of_local_quantile_average():
    shards = np.array(
        [
            [[-0.9, -0.9], [-0.8, -0.8], [-0.7, -0.7], [0.9, -0.6]],
            [[-0.6, -0.5], [-0.5, -0.4], [-0.4, -0.3], [-0.3, -0.2]],
        ],
        dtype=np.float32,
    )
    pooled = shards.reshape(-1, 2)
    local_targets = np.stack([_exact_required_bias_quantile(shard, top_k=1) for shard in shards])
    averaged_local_target = local_targets.mean(axis=0)

    actual = histogram_quantile_bias(
        jnp.asarray(pooled),
        jnp.zeros((2,), dtype=jnp.float32),
        top_k=1,
        num_bins=200,
        reduce_axes=(),
    )
    expected = _exact_required_bias_quantile(pooled, top_k=1)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=0.011, rtol=0)
    assert np.max(np.abs(np.asarray(actual) - averaged_local_target)) > 0.02


def test_histogram_qb_reduces_shard_histograms_before_quantile():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from experiments.grug.moe_hero_ep.quantile_balancing import histogram_quantile_bias

        required_bias = jnp.array(
            [
                [-0.9, -0.9],
                [-0.8, -0.8],
                [-0.7, -0.7],
                [0.9, -0.6],
                [-0.6, -0.5],
                [-0.5, -0.4],
                [-0.4, -0.3],
                [-0.3, -0.2],
            ],
            dtype=jnp.float32,
        )
        mesh = Mesh(
            np.asarray(jax.devices()),
            ("data",),
            axis_types=(AxisType.Explicit,),
        )
        required_bias = jax.device_put(required_bias, NamedSharding(mesh, P("data", None)))
        current_bias = jax.device_put(jnp.zeros((2,), dtype=jnp.float32), NamedSharding(mesh, P(None)))

        target = jax.shard_map(
            lambda required, bias: histogram_quantile_bias(
                required,
                bias,
                top_k=1,
                num_bins=200,
                reduce_axes=("data",),
            ),
            mesh=mesh,
            in_specs=(P("data", None), P(None)),
            out_specs=P(None),
        )
        with jax.set_mesh(mesh):
            actual = jax.jit(target)(required_bias, current_bias)

        expected = np.array([0.0, 0.0], dtype=np.float32)
        averaged_local_target = np.array([-0.025, 0.025], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(actual), expected, atol=0.011, rtol=0)
        assert np.max(np.abs(np.asarray(actual) - averaged_local_target)) > 0.02
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_run_grug_applies_ep_xla_defaults_and_keeps_explicit_values(monkeypatch):
    explicit_overlap = "--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    explicit_slop = "--xla_gpu_memory_limit_slop_factor=108"
    monkeypatch.setenv("XLA_FLAGS", f"{explicit_overlap} {explicit_slop}")
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(moe_implementation="moonep_jax"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        moonep_jax_wheel_build=None,
        moonep_transport=train.MoonEPTransport.TWO_SLICE,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert explicit_overlap in flags
    assert (
        f"--xla_gpu_experimental_parallel_collective_overlap_limit="
        f"{train.MOONEP_TWO_SLICE_COLLECTIVE_OVERLAP_LIMIT}" not in flags
    )
    assert explicit_slop in flags
    assert "--xla_gpu_memory_limit_slop_factor=106" not in flags
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" in flags
    assert "--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true" in flags
    assert (
        f"--xla_gpu_unsupported_override_fast_interconnect_slice_size={train.MOONEP_FAST_INTERCONNECT_SLICE_SIZE}"
        in flags
    )
    assert "--xla_gpu_experimental_enable_nccl_symmetric_buffers=true" in flags
    assert "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true" in flags
    assert not any(flag.startswith("--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl") for flag in flags)
    assert train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG in flags
    for name, value in train.HERO_EP_RUNTIME_ENV.items():
        assert os.environ[name] == value
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.80"


def test_run_grug_selects_direct_device_transport(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(moe_implementation="moonep_jax"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="direct-device-test")),
        resources=object(),
        processes_per_task=1,
        moonep_jax_wheel_build=None,
        moonep_transport=train.MoonEPTransport.DIRECT_DEVICE,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert (
        f"--xla_gpu_experimental_parallel_collective_overlap_limit="
        f"{train.MOONEP_DIRECT_DEVICE_COLLECTIVE_OVERLAP_LIMIT}" in flags
    )
    assert "--xla_gpu_experimental_enable_nccl_symmetric_buffers=true" in flags
    assert "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true" in flags
    assert "--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true" not in flags
    assert not any(flag.startswith("--xla_gpu_unsupported_override_fast_interconnect_slice_size") for flag in flags)
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.84"


@pytest.mark.parametrize(
    ("moe_implementation", "moonep_transport", "expected_limit"),
    [
        ("fixed_all_to_all", train.MoonEPTransport.TWO_SLICE, train.HERO_EP_COLLECTIVE_OVERLAP_LIMIT),
        (
            "moonep_jax",
            train.MoonEPTransport.TWO_SLICE,
            train.MOONEP_TWO_SLICE_COLLECTIVE_OVERLAP_LIMIT,
        ),
        (
            "moonep_jax",
            train.MoonEPTransport.DIRECT_DEVICE,
            train.MOONEP_DIRECT_DEVICE_COLLECTIVE_OVERLAP_LIMIT,
        ),
    ],
)
def test_run_grug_selects_collective_overlap_limit(
    monkeypatch,
    moe_implementation: str,
    moonep_transport: train.MoonEPTransport,
    expected_limit: int,
):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(moe_implementation=moe_implementation),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        moonep_jax_wheel_build=None,
        moonep_transport=moonep_transport,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert (
        f"--xla_gpu_experimental_parallel_collective_overlap_limit={expected_limit}" in os.environ["XLA_FLAGS"].split()
    )


@pytest.mark.parametrize(
    ("wheel_build", "expected_prefix", "expected_pjrt_sha256", "expected_runtime_sha256"),
    [
        (
            MoonEPJaxWheelBuild.LSA_20260802,
            "jax-f9f6bbace-xla-5d53e1e-20260802",
            "fd2724cd9f128ea1a0d1f74029ce6fcdaf7915db1a351b088316cc821ac2408d",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-20260802",
            "a1bb00b9ed594e7d1b85251bce63660bb85c5f7a661d618af677cee481a4572a",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-resources-20260802",
            "ad8ee4dff204460f10bff5eb468957b332131203b628bf02ad2bcc0fdff73d0f",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_WEAK_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-weak-20260802",
            "c71148f3901030525093480bbdf6582d255d7b34af5564a636ac409b24de1ffa",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_CTA16_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-cta16-20260802",
            "6a87208443b820f2e37c6e4517d22d7b9d1f143b224b1c6d91550d9cae604b2e",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_WORLD_GIN_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-world-gin-20260802",
            "c2521e40e7cd87f445b42445ba5221771b89c754caf5fa81c92a7b47add6cb31",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_BARRIER_ONLY_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-barrier-only-20260802",
            "d3e391455196736d8793e4a983c63ed1644fe90e8ce87e9f56635fa43c83196c",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_LSA_BARRIER_ONLY_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-lsa-barrier-only-20260802",
            "2a9411320bbc9fc36ce21c60eb9a2825b3c54b2a6afcbb75cbfa0fb9ed3a1023",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_KERNEL_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-kernel-20260802",
            "458f3277349276d5a13a3c652d625339f34c8602ed5a230f9bea861e1005fa44",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_LSA_HOST_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-lsa-host-20260802",
            "c2b5461dfbfd53dfcebf76af16eb55736c02aa934020e81edc2477d59851f973",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_RUN_COLLECTIVE_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-run-collective-20260802",
            "ba8fb4ba686e18ec710f6495e7f4e8d407cadf5076e8786a06314d30443e6eb4",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_AFTER_DEVICE_COMM_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-device-comm-20260802",
            "4b1ddc5011aa44126ff711c4efe2fe889ef43d380e31609e560437cc6cbee0cd",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_BEFORE_DEVICE_COMM_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-before-device-comm-20260802",
            "b3c2d632c70628d026af8bc87f09d48c405b03a57dcba69880327b66ecd748d2",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_AFTER_BUFFER_CONVERSION_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-buffer-conversion-20260802",
            "3475c89c11683f1106290ab487f55087bd924b6de9c787153f83f9403ce9b2bf",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_AFTER_SYMMETRIC_LOOKUP_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-symmetric-lookup-20260802",
            "a5c642dd2476faf51287ad7cd5d4734b27d9ab5cc4fd1cb2353d575de8e536a8",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_REPORT_DEVICE_PREDICATE_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-report-device-predicate-20260802",
            "3a4cf444687179fea6545a25cec73ec930dd875ec1bd033e8495262c02430042",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_FULL_MNNVL_20260802,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-weak-20260802",
            "c71148f3901030525093480bbdf6582d255d7b34af5564a636ac409b24de1ffa",
            "e38471a61852b2ec56265a1d39b866a33d65b340498380c1ba2101c77e729b38",
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_SPARSE_GIN_20260803,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-sparse-gin-20260803",
            "c9def64fb448d0fd2ccf65e9aaca14c58e4d7b137c41d57dab52a9efa2b4e171",
            None,
        ),
        (
            MoonEPJaxWheelBuild.LSA_NCCL_2307_MULTICONTEXT_GIN_20260803,
            "jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-gin-20260803",
            "1c8884750de97f1620635b3e9a32cf8e0db00ad53d7de5e77bb9f4753f59a8f1",
            None,
        ),
    ],
)
def test_run_grug_adds_verified_jax_wheels_after_standard_gpu_setup(
    wheel_build: MoonEPJaxWheelBuild,
    expected_prefix: str,
    expected_pjrt_sha256: str,
    expected_runtime_sha256: str | None,
):
    config = SimpleNamespace(
        model=SimpleNamespace(moe_implementation="moonep_jax"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="fixed-jax-test")),
        resources=ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="256g", disk="256g"),
        processes_per_task=1,
        moonep_jax_wheel_build=wheel_build,
        moonep_transport=train.MoonEPTransport.TWO_SLICE,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        train.run_grug(config)

    scripts = dispatch.call_args.kwargs["setup_scripts"]

    assert dispatch.call_args.kwargs["max_retries_failure"] == 0
    assert scripts is not None
    assert len(scripts) == 3
    assert "uv sync" in scripts[0]
    assert "--extra gpu" in scripts[0]
    assert "staging CUDA toolchain" in scripts[1]
    assert "fsspec.core.url_to_fs" in scripts[2]
    assert "--no-deps --reinstall" in scripts[2]
    assert expected_prefix in scripts[2]
    assert expected_pjrt_sha256 in scripts[2]
    assert "40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d" in scripts[2]
    assert "03e838842547a66af13bc93a533ce1943dc0f2eb83026a94994eca7f47c072b4" in scripts[2]
    if expected_runtime_sha256 is not None:
        assert expected_runtime_sha256 in scripts[2]
        assert "nvidia/nccl/lib/libnccl.so.2" in scripts[2]


def test_profile_window_uses_one_process_and_complete_timeline():
    profiler = launch._hero_profiler_config(start_step=3, num_steps=2)
    assert profiler.enabled
    assert profiler.start_step == 3
    assert profiler.num_steps == 2
    assert profiler.process_index == 0
    assert profiler.barrier_timeout == 600
    assert profiler.profile_options.host_tracer_level == 1
    assert profiler.profile_options.python_tracer_level == 0
    assert profiler.profile_options.enable_hlo_proto
    assert profiler.profile_options.advanced_configuration == {
        "gpu_max_activity_api_events": 1_000_000,
        "gpu_max_callback_api_events": 1_000_000,
        "gpu_num_chips_to_profile_per_task": 1,
    }


def test_profile_window_rejects_steps_skipped_by_callback_runner():
    with pytest.raises(ValueError, match="profile_start_step must be at least 3"):
        launch.build_hero_run(
            run_id="early-profile-window",
            num_steps=5,
            profile_start_step=2,
        )


def test_hero_process_count_must_divide_local_gpus():
    with pytest.raises(ValueError, match="processes_per_task=3 must divide 4 GPUs per node"):
        launch.build_hero_run(
            run_id="invalid-process-count",
            num_steps=3,
            processes_per_task=3,
        )


def test_hero_process_count_reaches_training_config():
    step = launch.build_hero_run(
        run_id="one-process-per-gpu",
        num_steps=3,
        processes_per_task=4,
        version="dev",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.processes_per_task == 4


def test_hero_transport_reaches_training_config():
    step = launch.build_hero_run(
        run_id="direct-device-transport",
        num_steps=3,
        moe_implementation="moonep_jax",
        moonep_transport=train.MoonEPTransport.DIRECT_DEVICE,
        version="dev",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.moonep_transport == train.MoonEPTransport.DIRECT_DEVICE
    assert config.trainer.finite_diagnostics == train.FiniteDiagnostics.GRADS


def test_direct_transport_keeps_explicit_finite_diagnostics():
    step = launch.build_hero_run(
        run_id="direct-device-diagnostics",
        num_steps=3,
        moe_implementation="moonep_jax",
        moonep_transport=train.MoonEPTransport.DIRECT_DEVICE,
        finite_diagnostics=train.FiniteDiagnostics.ALL,
        version="dev",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.finite_diagnostics == train.FiniteDiagnostics.ALL


def test_hero_diagnostics_reach_training_config():
    step = launch.build_hero_run(
        run_id="finite-diagnostics",
        num_steps=3,
        finite_diagnostics=train.FiniteDiagnostics.ALL,
        version="dev",
    )

    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.finite_diagnostics == train.FiniteDiagnostics.ALL


@pytest.mark.parametrize(
    ("worker_cpu", "worker_ram_gb", "message"),
    [
        (0, launch.HERO_WORKER_RAM_GB, "worker_cpu must be positive, got 0"),
        (launch.HERO_WORKER_CPU, 0, "worker_ram_gb must be positive, got 0"),
    ],
)
def test_hero_worker_resources_must_be_positive(worker_cpu: int, worker_ram_gb: int, message: str):
    with pytest.raises(ValueError, match=message):
        launch.build_hero_run(
            run_id="invalid-worker-resources",
            num_steps=3,
            worker_cpu=worker_cpu,
            worker_ram_gb=worker_ram_gb,
        )


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
