# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import haliax as hax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh

from levanter.grug.attention import AttentionMask
from levanter.grug.model import GrugModelConfig
from levanter.grug.model import activations, init_parameters, loss_fn
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmExample


def _make_grug_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(1, 1, 1, len(devices))
    return Mesh(
        mesh_devices,
        axis_names=("replica_dcn", "replica", "data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _full_next_token_loss(logits: jax.Array, token_ids: jax.Array, loss_weight: jax.Array) -> jax.Array:
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    nll = nll * loss_weight
    denom = jnp.sum(loss_weight)
    return jnp.sum(nll) / jnp.maximum(denom, jnp.array(1.0, dtype=nll.dtype))


def test_grug_model_loss_fn_matches_full_logits():
    cfg = GrugModelConfig(
        vocab_size=23,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
        cross_entropy_block_size=8,
    )

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        params = init_parameters(cfg, key=jax.random.key(0))
        token_ids = jax.random.randint(jax.random.key(1), (2, 9), 0, cfg.vocab_size, dtype=jnp.int32)
        loss_weight = jnp.ones((2, 9), dtype=jnp.float32).at[:, -1].set(0.0)

        hidden = activations(params, token_ids, cfg, mask=AttentionMask.causal())
        logits = hidden @ params.output_proj

        ref = _full_next_token_loss(logits, token_ids, loss_weight)
        got = loss_fn(params, token_ids, loss_weight, cfg, mask=AttentionMask.causal(), reduction="mean")

        assert jnp.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_grug_wrapper_compute_next_token_loss_uses_grug_loss_fn():
    cfg = GrugModelConfig(
        vocab_size=29,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
        cross_entropy_block_size=8,
    )

    mesh = _make_grug_mesh()
    with jax.set_mesh(mesh):
        Vocab = hax.Axis("vocab", cfg.vocab_size)
        model = GrugWrapper.init(Vocab, cfg, key=jax.random.key(0))

        Batch = hax.Axis("batch", 2)
        Pos = hax.Axis("position", 9)
        token_ids = hax.random.randint(jax.random.key(1), (Batch, Pos), 0, cfg.vocab_size, dtype=jnp.int32)
        loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32).at[Pos, Pos.size - 1].set(0.0)
        example = LmExample(tokens=token_ids, loss_weight=loss_weight, attn_mask=LevanterAttentionMask.causal())

        per_pos = model.compute_next_token_loss(example, reduction=None, reduction_axis=())
        assert isinstance(per_pos, hax.NamedArray)
        assert per_pos.axes == token_ids.axes

        expected = loss_fn(
            model.params, token_ids.array, loss_weight.array, cfg, mask=AttentionMask.causal(), reduction="none"
        )
        assert jnp.allclose(per_pos.array, expected, atol=1e-4, rtol=1e-4)
