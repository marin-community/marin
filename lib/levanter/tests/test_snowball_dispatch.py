# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.testing.devices import use_test_devices


def test_stats_forward_preserves_observer_and_packed_activations():
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
    )
    batch = hax.Axis("batch", 2 * jax.device_count())
    position = hax.Axis("position", 8)
    ids = hax.named(jnp.broadcast_to(jnp.arange(8), (batch.size, 8)), (batch, position))
    valid = hax.named(jnp.ones(ids.array.shape, dtype=jnp.bool_), ids.axes)
    with (
        jax.set_mesh(compact_grug_mesh()),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
        hidden, routes = eqx.filter_jit(model.activations_with_routes)(ids)
        measured, stats = eqx.filter_jit(model.activations_with_stats)(ids, valid)
        np.testing.assert_array_equal(measured.array, hidden.array)
        np.testing.assert_array_equal(stats.routes, routes)
        np.testing.assert_array_equal(stats.dropped, np.zeros(2))
        segments = hax.named(jnp.broadcast_to(jnp.arange(8) // 4, ids.array.shape), ids.axes)
        mask = AttentionMask.causal().with_segment_ids(segments, segments)
        measured, counts = eqx.filter_jit(lambda m: m.activations_with_stats(ids, valid, attn_mask=mask))(model)
        reference = eqx.filter_jit(lambda m: m.activations(ids, mask))(model)
        np.testing.assert_allclose(measured.array, reference.array, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(counts.dropped, np.zeros(2))


@pytest.mark.multidevice
def test_dropless_bos_routes_ignore_companions_and_padding():
    with use_test_devices(8) as devices:

        mesh = Mesh(
            np.array(devices).reshape(1, 2, 4, 1, 1, 1),
            ("replica_dcn", "data", "expert", "context", "replica", "model"),
            axis_types=(AxisType.Explicit,) * 6,
        )
        config = SnowballConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_experts=8,
            num_experts_per_token=2,
            num_layers=3,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=32,
            sliding_window=4,
            attention_implementation="reference",
            moe_implementation="ring_gather_combine",
        )
        with jax.set_mesh(mesh):
            model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
            blocks = tuple(
                replace(block, mlp=replace(block.mlp, expert_mlp=replace(block.mlp.expert_mlp, capacity_factor=4.0)))
                for block in model.transformer.blocks
            )
            model = replace(model, transformer=replace(model.transformer, blocks=blocks))
            forward = eqx.filter_jit(model.activations_with_stats)
            baseline = None
            for sequence, offset in [(8, 0), (16, 7)]:
                tokens = (np.arange(8 * sequence).reshape(8, sequence) + offset) % 32
                tokens[:, 0] = 1
                valid = np.arange(sequence)[None, :] < np.array([5, 4, 3, 6, 2, 7, 4, 5])[:, None]
                tokens[~valid] = 2
                axes = (hax.Axis("batch", 8), hax.Axis("position", sequence))
                _, stats = forward(hax.named(jnp.asarray(tokens), axes), hax.named(jnp.asarray(valid), axes))
                np.testing.assert_array_equal(stats.dropped, np.zeros(3))
                bos = np.asarray(stats.routes)[:, :, 0, :]
                np.testing.assert_array_equal(bos, np.broadcast_to(bos[:, :1, :], bos.shape))
                if baseline is not None:
                    np.testing.assert_array_equal(bos, baseline)
                baseline = bos
            capped_blocks = tuple(
                replace(
                    block,
                    mlp=replace(
                        block.mlp,
                        router=jnp.zeros_like(block.mlp.router),
                        router_bias=jnp.arange(8, 0, -1, dtype=jnp.float32),
                        expert_mlp=replace(block.mlp.expert_mlp, capacity_factor=1.0),
                    ),
                )
                for block in model.transformer.blocks
            )
            capped = replace(model, transformer=replace(model.transformer, blocks=capped_blocks))
            axes = (hax.Axis("batch", 8), hax.Axis("position", 8))
            tokens = hax.named(jnp.ones((8, 8), jnp.int32), axes)
            valid = hax.named(jnp.broadcast_to(jnp.arange(8) < 4, (8, 8)), axes)
            segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 1, 1, -1, -1, -1, -1]), (8, 8)), axes)
            mask = AttentionMask.causal().with_segment_ids(segments, segments.rename({"position": "key_position"}))
            _, counts = eqx.filter_jit(lambda m: m.activations_with_stats(tokens, valid, attn_mask=mask))(capped)
            # 32 valid tokens choose two experts; each layer admits 16 of the
            # 64 assignments and drops 48, all at receivers.
            np.testing.assert_array_equal(counts.sender_dropped, np.zeros(3))
            np.testing.assert_array_equal(counts.receiver_dropped, np.full(3, 48))
            np.testing.assert_array_equal(counts.dropped, np.full(3, 48))


@pytest.mark.multidevice
def test_training_forward_keeps_padding_out_of_lossy_expert_capacity():
    """Padding (segment id -1) takes no expert slots in the training forward, so it cannot push real tokens out."""
    batch, sequence, real_length = 4, 8, 3

    def config(capacity):
        return SnowballConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=1,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=16,
            sliding_window=4,
            attention_implementation="reference",
            moe_implementation="ring",
            moe_capacity_factor=capacity,
        )

    axes = (hax.Axis("batch", batch), hax.Axis("position", sequence))
    # Left padding: a row's padding comes before its real tokens in dispatch order.
    real = np.broadcast_to(np.arange(sequence) >= sequence - real_length, (batch, sequence))
    tokens = (np.arange(batch * sequence).reshape(batch, sequence) * 7 + 2) % 32
    tokens[~real] = 0
    ids = hax.named(jnp.asarray(tokens), axes)

    def mask(padding_segment):
        segments = hax.named(jnp.asarray(np.where(real, 0, padding_segment)), axes)
        return AttentionMask.causal().with_segment_ids(segments, segments.rename({"position": "key_position"}))

    with (
        use_test_devices(2),
        jax.set_mesh(compact_grug_mesh(expert_axis_size=2)),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert")}),
    ):
        dropless = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config(None), key=jax.random.PRNGKey(2))
        lossy = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config(1.0), key=jax.random.PRNGKey(2))
        # At capacity 1.0 the real assignments fit, but counting the padding pushes some out.
        drops = eqx.filter_jit(lambda m, valid: m.activations_with_stats(ids, valid, attn_mask=mask(1))[1].dropped)
        assert int(drops(lossy, hax.named(jnp.asarray(real), axes)).sum()) == 0
        assert int(drops(lossy, hax.named(jnp.ones((batch, sequence), bool), axes)).sum()) > 0

        forward = eqx.filter_jit(lambda m, attn_mask: m.activations(ids, attn_mask).array)
        expected = np.asarray(forward(dropless, mask(-1)))[real]
        np.testing.assert_allclose(np.asarray(forward(lossy, mask(-1)))[real], expected, rtol=1e-5, atol=1e-5)
        # The same padding as its own (valid) document competes for capacity and drops real tokens.
        assert not np.allclose(np.asarray(forward(lossy, mask(1)))[real], expected, rtol=1e-5, atol=1e-5)

        # The training drop metric counts only real tokens' assignments, and padding is not a drop.
        def drop_fraction(attn_mask):
            example = LmExample(tokens=ids, loss_weight=hax.ones(axes), attn_mask=attn_mask)
            metrics = eqx.filter_jit(lambda m: m.next_token_loss_and_metrics(example)[1])(lossy)
            return float(metrics["moe/dropped_assignment_fraction"])

        assert drop_fraction(mask(-1)) == 0.0
        assert drop_fraction(mask(1)) > 0.0
