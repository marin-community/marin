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
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import grugmuon_hero, launch, train


def test_full_bank_top_k_is_rejected_before_launch():
    # QB routing reads the (k+1)-th logit as its threshold, so a full-bank top-k asks `top_k` for
    # more entries than there are experts. Without this the job dies in the router, which is after
    # the 16-node gang is allocated.
    with pytest.raises(ValueError, match="must be < num_experts"):
        launch.build_hero_run(run_id="full-bank", num_steps=1, num_experts_per_token=128, version="dev")


def test_expert_bank_override_must_divide_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must divide the expert axis"):
        launch.build_hero_run(run_id="bad-bank", num_steps=1, num_experts=200, version="dev")


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


def _metrics(dropped, counts, entropy):
    return {
        "moe/dropped_assignments": jnp.asarray(dropped, dtype=jnp.float32),
        "train/router/routing_counts_per_layer": jnp.asarray(counts, dtype=jnp.float32),
        "train/router/routing_entropy_mean": jnp.asarray(entropy, dtype=jnp.float32),
        "qb_beta_per_layer": None,
    }


def test_fold_metrics_sums_drop_counts_and_averages_rates():
    # `_drop_metrics` divides dropped assignments by the FULL batch's assignment total, so a mean
    # fold here would understate the drop rate by exactly the microbatch count -- silently, and on
    # the metric the capacity sweep is measuring.
    folded = train._fold_metrics(
        [
            _metrics(100.0, [[6.0, 2.0]], 0.5),
            _metrics(300.0, [[4.0, 8.0]], 1.5),
        ]
    )

    assert float(folded["moe/dropped_assignments"]) == 400.0
    assert folded["train/router/routing_counts_per_layer"].tolist() == [[10.0, 10.0]]
    assert float(folded["train/router/routing_entropy_mean"]) == pytest.approx(1.0)
    assert folded["qb_beta_per_layer"] is None


def test_fold_metrics_rebuilds_routing_histogram_from_summed_counts():
    folded = train._fold_metrics(
        [
            {**_metrics(0.0, [[6.0, 2.0]], 0.5), "train/router/layer_0/routing_hist": object()},
            {**_metrics(0.0, [[4.0, 8.0]], 0.5), "train/router/layer_0/routing_hist": object()},
        ]
    )

    # Summed counts are [10, 10], so the histogram's mean expert id is 0.5, not either input's.
    assert float(folded["train/router/layer_0/routing_hist"].mean) == pytest.approx(0.5)


def test_fold_metrics_rejects_an_unclassifiable_metric():
    # A metric added later must not quietly default into the averaging bucket.
    with pytest.raises(TypeError, match="unfoldable type"):
        train._fold_metrics([{**_metrics(0.0, [[1.0]], 0.0), "train/new": "surprise"}] * 2)
