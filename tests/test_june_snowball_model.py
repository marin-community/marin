# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import collections
import contextlib
import dataclasses
import io
import os
import subprocess
import sys
import textwrap
from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, use_abstract_mesh
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.models.snowball import SnowballConfig

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, MoEMLP, RMSNorm, Transformer, rms_norm
from experiments.june_tpu_67b_a2b.moe.rl_model import JuneSnowballConfig, split_june_pipeline_model


def test_bf16_rms_norm_scoring_matches_differentiated_forward():
    # This size exercises reduction fusion that differs when AD retains intermediates.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        inputs = jax.random.normal(jax.random.PRNGKey(10), (2, 2048, 2560)).astype(jnp.bfloat16)
        weight = (jax.random.normal(jax.random.PRNGKey(11), (2560,)) * 0.3 + 1).astype(jnp.bfloat16)
        norm = RMSNorm(weight, eps=1e-6)

        def forward(module, values):
            return module(values)

        def objective(module, values):
            output = forward(module, values)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = jax.jit(forward)(norm, inputs)
        (_, trained), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(norm, inputs)
        jax.block_until_ready(gradients)
        np.testing.assert_array_equal(scored, trained)


# The custom-VJP RMS norm (rms_norm_impl="custom_vjp") computes the barrier path's float32 forward expression, but XLA
# may fuse its reduction with the users the barrier keeps apart, which rounds differently: a few float32 ulps, at most
# one ulp after the cast to bfloat16. Both paths are compared with a float64 reference, to which they are equally close.
_RMS_SHAPES = [(4, 16), (2, 8, 128), (2, 3, 4, 128), (3, 7, 2560)]
# Largest error of each path against the float64 reference, in units of the output dtype's epsilon times the
# reference's largest magnitude.
_RMS_EPS_MULTIPLE = {jnp.float32: 64, jnp.bfloat16: 2}
_RMS_FORWARD_RTOL = {
    jnp.float32: 4 * float(jnp.finfo(jnp.float32).eps),
    jnp.bfloat16: float(jnp.finfo(jnp.bfloat16).eps),
}


def _rms_inputs(shape, dtype, seed=0):
    k_x, k_g, k_w = jax.random.split(jax.random.key(seed), 3)
    # Rows of different scales, so eps matters for some of them.
    scale = jnp.exp(jax.random.uniform(k_w, (*shape[:-1], 1), minval=-6.0, maxval=3.0))
    x = (jax.random.normal(k_x, shape) * scale).astype(dtype)
    g = jax.random.normal(k_g, shape, dtype=jnp.float32)
    return x, g


def _rms_reference(x, g, eps, weight=None):
    # float64 value and VJP of x * rsqrt(mean(x^2) + eps) * weight.
    x = np.asarray(x, np.float64)
    g = np.asarray(g, np.float64)
    w = np.ones(x.shape[-1]) if weight is None else np.asarray(weight, np.float64)
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    gw = g * w
    dx = r * gw - x * r**3 * np.mean(gw * x, axis=-1, keepdims=True)
    dw = np.sum((g * x * r).reshape(-1, x.shape[-1]), axis=0)
    return x * r * w, dx, dw


def _max_scaled_error(actual, expected):
    actual = np.asarray(actual, np.float64)
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("shape", _RMS_SHAPES)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_matches_barrier_rms_norm(dtype, shape, eps):
    x, g = _rms_inputs(shape, dtype)

    @jax.jit
    def run(x, g):
        out = {}
        for impl in ("barrier", "custom_vjp"):
            y, vjp = jax.vjp(lambda v, i=impl: rms_norm(v, eps, impl=i), x)
            out[impl] = (y, vjp(g.astype(y.dtype))[0])
        return out

    out = run(x, g)
    (y_barrier, dx_barrier), (y_custom, dx_custom) = out["barrier"], out["custom_vjp"]
    assert y_custom.dtype == dx_custom.dtype == dtype
    np.testing.assert_allclose(y_custom, y_barrier, rtol=_RMS_FORWARD_RTOL[dtype], atol=0)
    tolerance = _RMS_EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    y_ref, dx_ref, _ = _rms_reference(x, g, eps)
    for name, dx in (("barrier", dx_barrier), ("custom_vjp", dx_custom)):
        assert _max_scaled_error(dx, dx_ref) <= tolerance, name
    assert _max_scaled_error(y_custom, y_ref) <= tolerance
    diff = np.max(np.abs(np.asarray(dx_custom, np.float64) - np.asarray(dx_barrier, np.float64)))
    assert diff <= tolerance * np.max(np.abs(dx_ref))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_layer_matches_reference(dtype):
    # The weight multiplies the normalized activations outside the custom VJP, so its gradient comes from AD.
    eps = 1e-5
    x, g = _rms_inputs((2, 64, 256), dtype, seed=3)
    weight = jax.random.normal(jax.random.key(4), (256,)) * 0.3 + 1
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):

        @jax.jit
        def run(w, v, g):
            y, vjp = jax.vjp(lambda w, v: RMSNorm(w, eps=eps, impl="custom_vjp")(v), w, v)
            dw, dx = vjp(g.astype(y.dtype))
            return y, dx, dw

        y, dx, dw = run(weight, x, g)
    y_ref, dx_ref, dw_ref = _rms_reference(x, np.asarray(g.astype(dtype), np.float64), eps, weight)
    tolerance = _RMS_EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    assert _max_scaled_error(y, y_ref) <= tolerance
    assert _max_scaled_error(dx, dx_ref) <= tolerance
    assert _max_scaled_error(dw, dw_ref) <= 64 * float(jnp.finfo(jnp.float32).eps)


def test_bf16_custom_vjp_rms_norm_scoring_is_within_one_ulp_of_differentiated_forward():
    # Unlike the barrier path (test_bf16_rms_norm_scoring_matches_differentiated_forward), the scoring forward does not
    # output the inverse RMS, so XLA may fuse the reduction with its users there and not in the differentiated forward.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        inputs = jax.random.normal(jax.random.PRNGKey(10), (2, 2048, 2560)).astype(jnp.bfloat16)
        weight = (jax.random.normal(jax.random.PRNGKey(11), (2560,)) * 0.3 + 1).astype(jnp.bfloat16)
        norm = RMSNorm(weight, eps=1e-6, impl="custom_vjp")

        def objective(module, values):
            output = module(values)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = jax.jit(lambda module, values: module(values))(norm, inputs)
        (_, trained), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(norm, inputs)
        jax.block_until_ready(gradients)
        np.testing.assert_allclose(scored, trained, rtol=float(jnp.finfo(jnp.bfloat16).eps), atol=0)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe"])
def test_custom_vjp_rms_norm_saves_fewer_residuals(remat_mode, stacked):
    # The custom VJP's residuals are unnamed, so the block checkpoints recompute them like the barrier path's: the
    # blocks save the same residuals at any depth. Outside the checkpoints, the embedding and final norms each save one
    # float32 activation and one float32 statistic fewer than AD of the barrier path.
    small = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        capacity_factor=8.0,
        attention_implementation="reference",
        disable_pko=True,
    )
    tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, small.vocab_size)

    def saved(num_layers, impl):
        cfg = dataclasses.replace(
            small, num_layers=num_layers, remat_mode=remat_mode, use_array_stacked_blocks=stacked, rms_norm_impl=impl
        )
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            params, static = eqx.partition(Transformer.init(cfg, key=jax.random.key(0)), eqx.is_array)

            def loss(p):
                hidden, _ = eqx.combine(p, static)(tokens)
                return jnp.mean(jnp.square(hidden))

            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                jax.ad_checkpoint.print_saved_residuals(loss, params)
        # One line per residual: its type, then where it comes from.
        return collections.Counter(line.split()[0] for line in report.getvalue().splitlines())

    for num_layers in (2, 4):
        barrier, custom = saved(num_layers, "barrier"), saved(num_layers, "custom_vjp")
        assert custom <= barrier, num_layers
        # Rows: a batch of 2 sharded over the batch axes.
        activation, statistic = "f32[2@(replica_dcn,data,expert),8,16]", "f32[2@(replica_dcn,data,expert),8,1]"
        assert barrier - custom == collections.Counter({activation: 2, statistic: 2}), num_layers


def test_bf16_embedding_gather_scoring_matches_differentiated_forward():
    # The gather barrier keeps scoring and AD from fusing the squared-sum reduction
    # into different layouts (wiki/363 embedding boundary). Production dims keep the
    # 8x320 shard layout that produced the original divergence.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        config = dataclasses.replace(
            _config(),
            vocab_size=512,
            hidden_dim=2560,
            intermediate_dim=1280,
            shared_expert_intermediate_dim=1280,
            num_heads=20,
            num_kv_heads=5,
            head_dim=128,
            max_seq_len=2048,
            sliding_window=512,
        )
        model = config.build(hax.Axis("vocab", 512), key=jax.random.PRNGKey(7))
        model = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_array(leaf) else leaf, model)
        stage = split_june_pipeline_model(model, 2)[0]
        tokens = jax.random.randint(jax.random.PRNGKey(5), (8, 512), 0, 512)

        def forward(part):
            return part.embed(tokens)

        def objective(part):
            output = forward(part)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = eqx.filter_jit(forward)(stage)
        (_, trained), gradients = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))(stage)
        jax.block_until_ready(gradients)
        np.testing.assert_array_equal(scored, trained)


def test_bf16_router_sigmoid_scoring_matches_differentiated_forward():
    # The sigmoid barrier keeps the expert-combine renormalization from fusing into
    # the sigmoid's reciprocal (wiki/363 router boundary). The trainable weights use
    # BF16 compute while the fixed QB router bias stays FP32, matching the cast
    # boundary described in snowball.md.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        mlp = MoEMLP.init(
            dataclasses.replace(
                _config().training_config(),
                hidden_dim=2560,
                intermediate_dim=1280,
                shared_expert_intermediate_dim=1280,
            ),
            key=jax.random.PRNGKey(0),
        )
        mlp = eqx.tree_at(
            lambda module: (
                module.router,
                module.expert_mlp.w_gate,
                module.expert_mlp.w_up,
                module.expert_mlp.w_down,
            ),
            mlp,
            replace=tuple(
                value.astype(jnp.bfloat16)
                for value in (mlp.router, mlp.expert_mlp.w_gate, mlp.expert_mlp.w_up, mlp.expert_mlp.w_down)
            ),
        )
        inputs = jax.random.normal(jax.random.PRNGKey(1), (8, 512, 2560)).astype(jnp.bfloat16)
        token_valid = jnp.ones((8, 512), dtype=jnp.bool_)

        def forward(module):
            return module(inputs, token_valid)[0]

        def objective(module):
            output = forward(module)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = eqx.filter_jit(forward)(mlp)
        (_, trained), gradients = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))(mlp)
        jax.block_until_ready(gradients)
        np.testing.assert_array_equal(scored, trained)


def _config():
    return JuneSnowballConfig(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.02,
        attention_implementation="reference",
        moe_implementation="ring",
    )


def test_june_adapter_uses_june_capacity_default_from_hf_config():
    assert JuneSnowballConfig().capacity_factor == 1.0
    decoded = JuneSnowballConfig.from_hf_config(_config().to_hf_config(24))
    assert decoded.capacity_factor == 1.0


def test_june_stage_telemetry_rejects_oversized_expert_parallel_microbatches():
    # With more than one expert shard, routing counters ride through floating auxiliary
    # transport, so a microbatch above 2**24 total assignments cannot represent exact
    # counts and must be rejected before tracing.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = dataclasses.replace(_config(), num_layers=26).build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(21))
        stage = split_june_pipeline_model(model, 2)[0]
    # 400_000 tokens x 2 experts x 26 layers exceeds 2**24 total assignments.
    segments = jnp.zeros((1, 400_000), dtype=jnp.int32)
    positions = jnp.zeros((1, 400_000), dtype=jnp.int32)
    hidden = jnp.zeros((1, 400_000, 12))
    expert_parallel = AbstractMesh(
        axis_sizes=(1, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with use_abstract_mesh(expert_parallel):
        with pytest.raises(ValueError, match=r"2\*\*24 assignments"):
            stage.run_blocks_with_stats(hidden, segments, positions)
    # A single expert shard keeps exact integer transport, so the same microbatch must trace.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        jax.eval_shape(
            stage.run_blocks_with_stats,
            jax.ShapeDtypeStruct((1, 400_000, 12), jnp.float32),
            jax.ShapeDtypeStruct((1, 400_000), jnp.int32),
            jax.ShapeDtypeStruct((1, 400_000), jnp.int32),
        )


class ModelInputs(NamedTuple):
    tokens: hax.NamedArray
    segments: hax.NamedArray
    positions: hax.NamedArray


def _inputs() -> ModelInputs:
    Batch, Position = hax.Axis("batch", jax.device_count()), hax.Axis("position", 6)
    tokens = hax.named(jnp.broadcast_to(jnp.array([0, 0, 2, 3, 4, 5]), (Batch.size, 6)), (Batch, Position))
    segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 1, 1, 1, 1]), (Batch.size, 6)), (Batch, Position))
    positions = hax.named(jnp.broadcast_to(jnp.array([0, 0, 0, 2, 4, 6]), (Batch.size, 6)), (Batch, Position))
    return ModelInputs(tokens, segments, positions)


def _arrays(tree):
    return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


@pytest.mark.timeout(180)
def test_june_hf_mapping_roundtrip_preserves_weights_and_forward():
    config = _config()
    decoded = JuneSnowballConfig.from_hf_config(config.to_hf_config(config.vocab_size))
    assert isinstance(decoded, JuneSnowballConfig)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = config.build(hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(5))
        tensors = model.to_state_dict()
        # A non-square Q projection catches transpose errors independently of roundtrip symmetry.
        np.testing.assert_array_equal(
            tensors["model.layers.0.self_attn.q_proj.weight"], model.transformer.blocks[0].attn.w_q.T
        )
        loaded = config.build(model.Vocab, key=jax.random.PRNGKey(9)).from_state_dict(tensors)
        for source, restored in zip(_arrays(model), _arrays(loaded), strict=True):
            np.testing.assert_array_equal(source, restored)
        serving_config = dataclasses.replace(
            SnowballConfig.from_hf_config(config.to_hf_config(config.vocab_size)),
            attention_implementation="reference",
        )
        serving = serving_config.build(model.Vocab, key=jax.random.PRNGKey(3)).from_state_dict(tensors)
        tokens, _, _ = _inputs()
        # Unpadded inputs exercise shared HF semantics without the serving snapshot's mask limitation.
        tokens = tokens["position", hax.ds(2, 4)]
        actual = hax.named_jit(lambda m, t: m.activations(t))(loaded, tokens)
        reference = hax.named_jit(lambda m, t: m.activations(t))(serving, tokens)
        np.testing.assert_allclose(actual.array, reference.array, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(180)
def test_june_adapter_preserves_mask_positions_and_transformer_gradient():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _config().build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(7))
        tokens, segments, positions = _inputs()
        mask = AttentionMask.causal().with_segment_ids(segments)

        def wrapped(m):
            return m.activations(tokens, mask, pos_ids=positions).array[:, 2:].sum()

        def direct(m):
            hidden, _ = m.transformer(
                tokens.array, GrugAttentionMask.causal().with_segment_ids(segments.array), position_ids=positions.array
            )
            return hidden[:, 2:].sum()

        value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(wrapped))(model)
        expected, expected_gradient = eqx.filter_jit(eqx.filter_value_and_grad(direct))(model)
        np.testing.assert_array_equal(value, expected)
        for actual, reference in zip(_arrays(gradient), _arrays(expected_gradient), strict=True):
            np.testing.assert_array_equal(actual, reference)
        padded = hax.named_jit(lambda m, t, a, p: m.activations(t, a, pos_ids=p))(model, tokens, mask, positions)
        unpadded_tokens = tokens["position", hax.ds(2, 4)]
        unpadded_positions = positions["position", hax.ds(2, 4)]
        unpadded = hax.named_jit(lambda m, t, p: m.activations(t, pos_ids=p))(model, unpadded_tokens, unpadded_positions)
        np.testing.assert_allclose(padded.array[:, 2:], unpadded.array, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_layers", [2, 5])
def test_june_pipeline_stage_split_preserves_masked_forward(num_layers):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = dataclasses.replace(_config(), num_layers=num_layers).build(
            hax.Axis("vocab", 24), key=jax.random.PRNGKey(21)
        )
        tokens, segments, positions = _inputs()
        stages = split_june_pipeline_model(model, 2)

        def pipeline(parts):
            hidden = parts[0].embed(tokens.array)
            counts = []
            for stage in parts:
                hidden, stats = stage.run_blocks_with_stats(hidden, segments.array, positions.array)
                counts.append(stats)
            hidden = parts[-1].finish(hidden)
            return hidden @ parts[-1].get_lm_head(), counts

        actual, counts = eqx.filter_jit(pipeline)(stages)
        for stage, stats in zip(stages, counts, strict=True):
            # Padding is routed too and consumes capacity; count the complete input width.
            assert int(stats["routing_assignments"]) == tokens.array.size * 2 * len(stage.blocks)
            assert int(stats["routing_sender_drops"]) == 0
            assert int(stats["routing_receiver_drops"]) == 0
            assert int(stats["routing_max_layer_drops"]) == 0
            assert int(stats["routing_drop_layer"]) == -1
        expected = hax.named_jit(
            lambda m: m.activations(
                tokens,
                AttentionMask.causal().with_segment_ids(segments),
                pos_ids=positions,
            ).array
            @ m.get_lm_head().array
        )(model)
        np.testing.assert_array_equal(actual, expected)
        # Every parameter is owned by exactly one stage, including both boundary modules.
        assert sum(array.size for array in _arrays(stages)) == sum(array.size for array in _arrays(model))


@pytest.mark.timeout(180)
def test_june_adapter_mask_axis_order_does_not_change_attention():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _config().build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(23))
        tokens, segments, positions = _inputs()
        Batch = hax.Axis("batch", 2 * jax.device_count())
        tokens, segments, positions = (hax.concatenate(Batch, [value, value]) for value in (tokens, segments, positions))
        canonical = AttentionMask.causal().with_segment_ids(segments)
        transposed = AttentionMask.causal().with_segment_ids(segments.rearrange(("position", "batch")))
        score = hax.named_jit(lambda mask: model.activations(tokens, mask, pos_ids=positions))
        expected = score(canonical)
        actual = score(transposed)
        np.testing.assert_array_equal(actual.array, expected.array)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_june_stage_hf_mapping_loads_only_owned_tensors(num_stages):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = dataclasses.replace(_config(), num_layers=5).build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(31))
        full = model.to_state_dict(prefix="policy")
        stage_tensors = {}
        for stage in split_june_pipeline_model(model, num_stages):
            owned = stage.to_state_dict(prefix="policy")
            template = eqx.filter_eval_shape(lambda stage: stage, stage)
            assert not (stage_tensors.keys() & owned.keys())
            stage_tensors.update(owned)
            # The loader only receives this stage's key subset and abstract shape leaves.
            restored = template.from_state_dict(owned, prefix="policy")
            for expected, actual in zip(_arrays(stage), _arrays(restored), strict=True):
                np.testing.assert_array_equal(expected, actual)
        assert stage_tensors.keys() == full.keys()
        for key, value in full.items():
            np.testing.assert_array_equal(stage_tensors[key], value)


@pytest.mark.timeout(180)
def test_june_pipeline_stages_match_model_under_expert_parallelism():
    """Stages must match the full model exactly when the expert axis has size 2.

    Runs in a fresh interpreter with two forced CPU devices because the XLA device
    count is process-global; the ring MoE backend then executes expert-parallel
    collectives through shard_map, exercising the path a size-1 expert axis skips.
    Capacity factor 0.5 clips assignments structurally, so the per-stage drop
    counters observe real clipped work. Gradients stay pinned so the equality holds
    at the training boundary, not only in the forward.
    """
    script = textwrap.dedent(
        """
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.layers.attention import AttentionMask

        from experiments.june_tpu_67b_a2b.moe.rl_model import JuneSnowballConfig, split_june_pipeline_model

        assert jax.device_count() == 2, jax.device_count()

        Batch, Position = hax.Axis("batch", 2), hax.Axis("position", 6)
        tokens = hax.named(jnp.broadcast_to(jnp.array([0, 0, 2, 3, 4, 5]), (2, 6)), (Batch, Position))
        segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 1, 1, 1, 1]), (2, 6)), (Batch, Position))
        positions = hax.named(jnp.broadcast_to(jnp.array([0, 0, 0, 2, 4, 6]), (2, 6)), (Batch, Position))
        mask = AttentionMask.causal().with_segment_ids(segments)

        config = JuneSnowballConfig(
            vocab_size=24, hidden_dim=12, intermediate_dim=16, shared_expert_intermediate_dim=12,
            num_experts=4, num_experts_per_token=2, num_layers=5, num_heads=2, num_kv_heads=1, head_dim=8,
            max_seq_len=8, sliding_window=4, initializer_std=0.02,
            attention_implementation="reference", moe_implementation="ring", capacity_factor=0.5,
        )

        with jax.set_mesh(compact_grug_mesh(expert_axis_size=2)):
            model = config.build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(21))
            stages = split_june_pipeline_model(model, 2)

            def pipeline(parts):
                hidden = parts[0].embed(tokens.array)
                counts = []
                for stage in parts:
                    hidden, stats = stage.run_blocks_with_stats(hidden, segments.array, positions.array)
                    counts.append(stats)
                hidden = parts[-1].finish(hidden)
                return hidden @ parts[-1].get_lm_head(), counts

            actual, counts = eqx.filter_jit(pipeline)(stages)
            expected = eqx.filter_jit(
                lambda m: m.activations(tokens, mask, pos_ids=positions).array @ m.get_lm_head().array
            )(model)
            assert np.array_equal(np.asarray(actual), np.asarray(expected))

            # Each shard accepts at most 6 of each layer's 24 assignments, so every layer
            # must drop at least 12 and the counters must see that clipping.
            for stage, stats in zip(stages, counts, strict=True):
                assert int(stats["routing_assignments"]) == tokens.array.size * 2 * len(stage.blocks)
                drops = int(stats["routing_sender_drops"]) + int(stats["routing_receiver_drops"])
                assert drops >= 12 * len(stage.blocks)
                assert int(stats["routing_max_layer_drops"]) > 0
                assert 0 <= int(stats["routing_drop_layer"]) < config.num_layers

            def pipeline_loss(parts):
                hidden = parts[0].embed(tokens.array)
                for stage in parts:
                    hidden, _ = stage.run_blocks_with_stats(hidden, segments.array, positions.array)
                return jnp.sum(jnp.sin(parts[-1].finish(hidden).astype(jnp.float32)))

            def model_loss(m):
                return jnp.sum(jnp.sin(m.activations(tokens, mask, pos_ids=positions).array.astype(jnp.float32)))

            _, stage_gradients = eqx.filter_jit(eqx.filter_value_and_grad(pipeline_loss))(stages)
            _, model_gradients = eqx.filter_jit(eqx.filter_value_and_grad(model_loss))(model)
            stage_tensors = {}
            for stage, gradients in zip(stages, stage_gradients, strict=True):
                owned = gradients.to_state_dict(prefix="policy")
                assert not (stage_tensors.keys() & owned.keys())
                stage_tensors.update(owned)
            model_tensors = model_gradients.to_state_dict(prefix="policy")
            assert stage_tensors.keys() == model_tensors.keys()
            for key, value in model_tensors.items():
                assert np.array_equal(np.asarray(stage_tensors[key]), np.asarray(value)), key
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
