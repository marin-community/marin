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
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, set_mesh, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.grug._moe import rms_gated_norm as rgn
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import grugmuon_hero, launch, model, train
from experiments.grug.moe_hero_ep import small_scale_abl_launch as abl


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
    ) == (6144, 48, 192, 6144, 4, 3072, 1.33, 1024, 4096)


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
    assert os.environ["JAX_ENABLE_PGLE"] == "true"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"


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
    # The EP rung is a downsized hero: latent = hidden/2, capacity 1.33, top-k QB (the hero default).
    assert m.latent_dim == m.hidden_dim // 2
    assert m.capacity_factor == 1.33
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


_RMS_GATED_NORM_EPS = 1e-5


def _rms_gated_norm_mesh():
    return Mesh(
        np.asarray([jax.devices("cpu")[0]]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _rms_gated_norm_inputs(dtype):
    keys = jax.random.split(jax.random.key(19), 5)
    return SimpleNamespace(
        x=jax.random.normal(keys[0], (2, 3, 16), dtype=dtype),
        norm_weight=(1 + 0.1 * jax.random.normal(keys[1], (16,), dtype=jnp.float32)).astype(dtype),
        w_down=(0.1 * jax.random.normal(keys[2], (16, model._GATED_NORM_RANK), dtype=jnp.float32)).astype(dtype),
        w_up=(0.1 * jax.random.normal(keys[3], (model._GATED_NORM_RANK, 16), dtype=jnp.float32)).astype(dtype),
        cotangent=jax.random.normal(keys[4], (2, 3, 16), dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rms_gated_norm_matches_the_stock_modules(dtype):
    """The shared boundary must reproduce this variant's own GatedNorm(RMSNorm(x)) exactly."""
    with set_mesh(_rms_gated_norm_mesh()):
        inputs = _rms_gated_norm_inputs(dtype)
        rms = model.RMSNorm(weight=inputs.norm_weight, eps=_RMS_GATED_NORM_EPS)
        gated = model.GatedNorm(w_down=inputs.w_down, w_up=inputs.w_up)

        expected = gated(rms(inputs.x))
        actual = model._apply_rms_gated_norm(inputs.x, rms, gated, "xla")

        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(jnp.float32, 2e-6), (jnp.bfloat16, 3e-2)],
)
def test_fused_reverse_matches_stock_autodiff(monkeypatch, dtype, tolerance):
    """Pin the EP wiring of the fused custom VJP against differentiating the stock path.

    The SM100 kernels are swapped for their pure-JAX references so the composition and the
    batch-axis reductions are exercised on CPU.
    """

    monkeypatch.setattr(
        rgn,
        "_backward_kernel",
        lambda: rgn.exact_rms_backward_fused_reference,
    )
    with set_mesh(_rms_gated_norm_mesh()):
        inputs = _rms_gated_norm_inputs(dtype)
        rms = model.RMSNorm(weight=inputs.norm_weight, eps=_RMS_GATED_NORM_EPS)
        gated = model.GatedNorm(w_down=inputs.w_down, w_up=inputs.w_up)
        spec = P(model._BATCH_AXES)
        x = jax.sharding.reshard(inputs.x, spec)
        cotangent = jax.sharding.reshard(inputs.cotangent, spec)

        def stock(x, norm_weight, w_down, w_up):
            return model.GatedNorm(w_down=w_down, w_up=w_up)(
                model.RMSNorm(weight=norm_weight, eps=_RMS_GATED_NORM_EPS)(x)
            )

        def fused(x, norm_weight, w_down, w_up):
            return model._apply_rms_gated_norm(
                x,
                model.RMSNorm(weight=norm_weight, eps=_RMS_GATED_NORM_EPS),
                model.GatedNorm(w_down=w_down, w_up=w_up),
                "quack_coda_backward",
            )

        primals = (x, rms.weight, gated.w_down, gated.w_up)
        expected_out, expected_pullback = jax.vjp(stock, *primals)
        actual_out, actual_pullback = jax.vjp(fused, *primals)
        np.testing.assert_array_equal(actual_out, expected_out)

        for name, actual, expected in zip(
            ("input", "norm_weight", "w_down", "w_up"),
            actual_pullback(cotangent),
            expected_pullback(cotangent),
            strict=True,
        ):
            error = float(jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))))
            assert error <= tolerance, f"{dtype} {name} gradient: max abs error {error:.6g} > {tolerance:.6g}"


def test_runtime_rms_implementation_is_not_serialized_in_hf_config():
    cfg = _latent_config()
    cfg = dataclasses.replace(cfg, rms_gated_norm_implementation="quack_coda_backward")

    serialized = cfg.to_hf_config(cfg.vocab_size).to_dict()
    restored = model.GrugModelConfig.from_hf_config(model.GrugMoeHfConfig.from_dict(serialized))

    assert "rms_gated_norm_implementation" not in serialized
    assert restored.rms_gated_norm_implementation == "xla"
