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
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import grugmuon_hero, train
from experiments.grug.moe_hero_ep.quantile_balancing import histogram_quantile_bias


def _exact_required_bias_quantile(required_bias: np.ndarray, *, top_k: int) -> np.ndarray:
    target_rank = (required_bias.shape[0] * top_k + required_bias.shape[1] - 1) // required_bias.shape[1]
    target = np.sort(required_bias, axis=0)[target_rank - 1]
    return target - target.mean()


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
    monkeypatch.setenv("XLA_FLAGS", explicit_overlap)
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
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
