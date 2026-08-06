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
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, set_mesh, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import grugmuon_hero, launch, model, small_scale_abl_launch, train


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


def test_capacity_factor_is_rejected_for_a_flavor_that_never_drops():
    # `scatter` computes every assignment, so a capacity factor would be silently inert and a sweep
    # over it would produce identical runs under different names.
    with pytest.raises(ValueError, match="never drops"):
        launch.build_hero_run(run_id="nodrop-cf", num_steps=1, flavor="fsdp-nodrop", capacity_factor=1.5, version="dev")


def test_eval_every_adds_the_held_out_suites_as_dependencies():
    # Held-out sets are what make a run scoreable; a throughput-only run should not pay for them.
    off = launch.build_hero_run(run_id="eval-off", num_steps=1, version="dev")
    on = launch.build_hero_run(run_id="eval-on", num_steps=1, eval_every=50, version="dev")

    assert len(off.deps) == 1
    assert len(on.deps) > len(off.deps)


@pytest.mark.parametrize("size", ["d768", "d1024", "d1280"])
def test_hybrid_kv_branches_agree_on_sharding_when_model_axis_is_wide(size):
    # `lax.cond` compares full types, not just shapes. The pass-through branch kept the projection's
    # `model`-sharded head axis while the align branch sliced to one head and broadcast back, so any
    # shape with local_kv_heads != global_kv_heads failed at trace time on a mesh whose model axis is
    # wider than one. d768 masked it by setting both counts to 1; d1024 and d1280 set 2 and 1.
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 8, 4, 2),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    shape = small_scale_abl_launch.SMALL_SHAPES[size]
    cfg = small_scale_abl_launch._small_model(shape, 1.0, "reference", "fixed_all_to_all", 1, 128)
    tokens = jax.ShapeDtypeStruct((64, 128), jnp.int32)
    with set_mesh(mesh):
        jax.eval_shape(lambda t: model.Transformer.init(cfg, key=jax.random.key(0))(t)[0], tokens)
