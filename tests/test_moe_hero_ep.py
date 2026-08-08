# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
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
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, set_mesh, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import grugmuon_hero, launch, model, small_scale_abl_launch, train


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
        config.trainer.trainer.train_batch_size,
        config.model.max_seq_len,
    ) == (6144, 48, 192, 6272, 4, 3072, 1.33, 1024, 4096)


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


def test_checkpoint_interval_must_be_positive():
    with pytest.raises(ValueError, match="checkpoint_interval must be positive"):
        launch.build_hero_run(
            run_id="bad-checkpoint-interval",
            dp_racks=1,
            num_steps=1,
            checkpoint_interval=timedelta(0),
            version="dev",
        )


def test_expert_bank_override_must_divide_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must divide the expert axis"):
        launch.build_hero_run(run_id="bad-bank", dp_racks=1, num_steps=1, num_experts=200, version="dev")


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


def test_run_grug_keeps_explicit_ep_runtime_values(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_PGLE", "true")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=1,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "true"
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


def test_capacity_factor_is_rejected_for_a_flavor_that_never_drops():
    # `scatter` computes every assignment, so a capacity factor would be silently inert and a sweep
    # over it would produce identical runs under different names.
    with pytest.raises(ValueError, match="never drops"):
        launch.build_hero_run(
            run_id="nodrop-cf", dp_racks=1, num_steps=1, flavor="fsdp-nodrop", capacity_factor=1.5, version="dev"
        )


def test_eval_every_adds_the_held_out_suites_as_dependencies():
    # Held-out sets are what make a run scoreable; a throughput-only run should not pay for them.
    off = launch.build_hero_run(run_id="eval-off", dp_racks=1, num_steps=1, version="dev")
    on = launch.build_hero_run(run_id="eval-on", dp_racks=1, num_steps=1, eval_every=50, version="dev")

    assert len(off.deps) == 1
    assert len(on.deps) > len(off.deps)


@pytest.mark.parametrize("size", ["d768", "d1024", "d1280"])
def test_hybrid_kv_branches_agree_on_sharding_when_model_axis_is_wide(size):
    # `lax.cond` compares full types, not just shapes. The pass-through branch kept the projection's
    # `model`-sharded head axis while the align branch sliced to one head and broadcast back, so any
    # shape with local_kv_heads != global_kv_heads failed at trace time on a mesh whose model axis is
    # wider than one. d768 masked it by setting both counts to 1; d1024 and d1280 set 2 and 1.
    mesh = _explicit_mesh(1, 8, 4, 2)
    shape = small_scale_abl_launch.SMALL_SHAPES[size]
    cfg = small_scale_abl_launch._small_model(shape, 1.0, "reference", "fixed_all_to_all", 1, 128)
    tokens = jax.ShapeDtypeStruct((64, 128), jnp.int32)
    with set_mesh(mesh):
        jax.eval_shape(lambda t: model.Transformer.init(cfg, key=jax.random.key(0))(t)[0], tokens)


def _explicit_mesh(*axis_sizes):
    return Mesh(
        np.asarray(jax.devices()).reshape(*axis_sizes),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _latent_config(size="d768", latent_dim=None):
    shape = small_scale_abl_launch.SMALL_SHAPES[size]
    cfg = small_scale_abl_launch._small_model(shape, 1.0, "reference", "fixed_all_to_all", 1, 128)
    return dataclasses.replace(cfg, latent_dim=latent_dim)


def test_latent_moe_shrinks_the_dispatched_width_but_not_the_token():
    # The point of LatentMoE is that the all-to-all payload narrows while the residual stream does
    # not, so the expert weights must be latent-wide and the layer output hidden-wide.
    mesh = _explicit_mesh(1, 1, 64, 1)
    cfg = _latent_config(latent_dim=192)
    tokens = jax.ShapeDtypeStruct((64, 128), jnp.int32)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
        out = jax.eval_shape(lambda t: model.Transformer.init(cfg, key=jax.random.key(0))(t)[0], tokens)

    assert built.w_latent_down.shape == (cfg.hidden_dim, 192)
    # Normalizing the latent keeps the expert input at unit scale despite the down-projection.
    assert built.latent_norm.weight.shape == (192,)
    assert built.w_latent_up.shape == (192, cfg.hidden_dim)
    # Expert banks are latent-wide: this is what narrows the dispatch.
    assert built.expert_mlp.w_gate.shape[1] == 192
    assert built.expert_mlp.w_up.shape[1] == 192
    assert built.expert_mlp.w_down.shape[2] == 192
    # The residual stream is untouched.
    assert out.shape[-1] == cfg.hidden_dim


def test_latent_moe_is_absent_by_default():
    # Every recorded run predates this feature, so the default must build the identical layer.
    mesh = _explicit_mesh(1, 1, 64, 1)
    cfg = _latent_config(latent_dim=None)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
    assert built.w_latent_down is None and built.w_latent_up is None
    assert built.latent_norm is None
    assert built.expert_mlp.w_gate.shape[1] == cfg.hidden_dim


def test_latent_dim_above_hidden_is_rejected():
    # A latent wider than the hidden dim adds communication instead of removing it.
    with pytest.raises(ValueError, match="latent_dim must be in"):
        _latent_config(latent_dim=99999)


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


def test_inline_watch_uses_one_watched_train_step_between_log_intervals(monkeypatch):
    params = _TinyWatchModel(weight=jnp.array(2.0))
    optimizer = optax.sgd(0.1)
    state = train.GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
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
