# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``pointer_mixture``: the sorted-link candidates against a brute-force scan, the mixture loss against a
dense full-vocabulary mixture, and a tiny model that starts as the control and trains through the head."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track import pointer
from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_K = pointer.POINTER_SLOTS


def _brute_force_candidates(tokens: np.ndarray, segments: np.ndarray) -> np.ndarray:
    out = np.full((*tokens.shape, _K), -1)
    for row in range(tokens.shape[0]):
        x, seg = tokens[row], segments[row]
        for t in range(len(x)):
            for order, n in enumerate((1, 2)):
                if t - n + 1 < 0 or seg[t - n + 1] != seg[t]:
                    continue
                found = [
                    x[s + 1]
                    for s in range(t - 1, n - 2, -1)
                    if seg[s - n + 1] == seg[t] and list(x[s - n + 1 : s + 1]) == list(x[t - n + 1 : t + 1])
                ][: pointer.MATCHES_PER_ORDER]
                start = order * pointer.MATCHES_PER_ORDER
                out[row, t, start : start + len(found)] = found
    return out


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def test_candidates_match_a_brute_force_scan_within_documents():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 3, size=(3, 40))
    segments = np.stack([np.zeros(40, int), np.arange(40) // 13, np.repeat([0, 1, 2, 3], [5, 1, 20, 14])])
    with jax.set_mesh(_mesh()):
        got = pointer.pointer_candidates(jnp.asarray(tokens), jnp.asarray(segments), ("replica_dcn", "data", "expert"))
    np.testing.assert_array_equal(np.asarray(got), _brute_force_candidates(tokens, segments))


def _dense_distribution(head, h, output, candidates):
    """The full next-token mixture, written out over the vocabulary."""
    p_vocab = jax.nn.softmax(h @ output, -1)
    valid = candidates >= 0
    agreement = jnp.sum(valid[..., None, :] & (candidates[..., :, None] == candidates[..., None, :]), -1) - 1
    logits = h @ head.pointer_w + 8 * head.pointer_b[jnp.arange(_K), jnp.maximum(agreement, 0)] - 2
    weights = jax.nn.softmax(jnp.concatenate([jnp.zeros_like(logits[..., :1]), jnp.where(valid, logits, -jnp.inf)], -1))
    pi = weights[..., 1:]
    open_gate = jnp.clip(8 * head.pointer_open, 0, 1)
    copied = jnp.einsum("bsk,bskv->bsv", pi, jax.nn.one_hot(candidates, output.shape[-1]))
    return (1 - open_gate * pi.sum(-1))[..., None] * p_vocab + open_gate * copied


def test_mixture_loss_matches_the_dense_mixture_values_and_gradients():
    vocab, dim = 6, 5
    keys = iter(jax.random.split(jax.random.key(3), 8))
    head = pointer.PointerHead(
        pointer_w=jax.random.normal(next(keys), (dim, _K)),
        pointer_b=0.3 * jax.random.normal(next(keys), (_K, _K)),
        pointer_open=jnp.asarray(0.07),
    )
    h = jax.random.normal(next(keys), (2, 16, dim))
    output = jax.random.normal(next(keys), (dim, vocab))
    tokens = jax.random.randint(next(keys), (2, 16), 0, vocab)
    labels = jnp.roll(tokens, -1, 1)
    segments = jnp.broadcast_to(jnp.arange(16) // 9, tokens.shape)
    with jax.set_mesh(_mesh()):
        candidates = pointer.pointer_candidates(tokens, segments, ("replica_dcn", "data", "expert"))
    # Unsharded, to meet the reference's off-mesh arrays.
    candidates = jnp.asarray(np.asarray(candidates))
    assert np.any(np.asarray(candidates) == np.asarray(labels)[..., None])

    def actual(head, h):
        logits = h @ output
        ce = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
        return pointer.pointer_mixture_nll(head, h @ head.pointer_w, candidates, labels, ce)[0]

    def expected(head, h):
        p = _dense_distribution(head, h, output, candidates)
        return -jnp.log(jnp.take_along_axis(p, labels[..., None], -1)[..., 0])

    np.testing.assert_allclose(_dense_distribution(head, h, output, candidates).sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(actual(head, h), expected(head, h), rtol=1e-5, atol=1e-6)
    got = jax.tree.leaves(jax.grad(lambda *a: actual(*a).mean(), argnums=(0, 1))(head, h))
    want = jax.tree.leaves(jax.grad(lambda *a: expected(*a).mean(), argnums=(0, 1))(head, h))
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-4, atol=1e-6)


def test_tiny_model_starts_as_the_control_and_trains_through_the_head():
    hidden, seq, vocab = 64, 128, 128
    config = GrugModelConfig(
        vocab_size=vocab,
        hidden_dim=hidden,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden // 2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=seq,
        sliding_window=seq,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(hidden),
        qk_norm=False,
        sconv_sites=("attn", "mlp"),
        mla=True,
        mla_kv_latent_dim=32,
        attn_res=True,
        attn_res_num_blocks=2,
        local_mixer=LocalMixer.KDA,
        pointer_mixture=True,
    )
    # Repeated spans give the pointer matches to find.
    tokens = jnp.tile(jax.random.randint(jax.random.key(1), (2, seq // 4), 0, vocab), (1, 4))
    weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, pointer_mixture=False), key=jax.random.key(0))

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4)

        def per_token_fn(m, tokens):
            return m.next_token_loss(tokens, weight, reduction="none", logsumexp_weight=1e-4)

        start_loss, start_grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        plain_loss = eqx.filter_jit(loss_fn)(plain)
        keys = jax.random.split(jax.random.key(2), 2)
        opened = eqx.tree_at(
            lambda m: (m.pointer_head.pointer_w, m.pointer_head.pointer_b, m.pointer_head.pointer_open),
            model,
            (
                0.1 * jax.random.normal(keys[0], (hidden, _K)),
                0.1 * jax.random.normal(keys[1], (_K, _K)),
                jnp.asarray(0.05),
            ),
        )
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(opened)
        per_token = eqx.filter_jit(per_token_fn)(opened, tokens)
        changed = eqx.filter_jit(per_token_fn)(opened, tokens.at[:, 64:].set((tokens[:, 64:] + 1) % vocab))
        mask = GrugMoeMuonHConfig().create_mask(model)

    # Zero-initialised, the head leaves the loss at the control's; only the opening scalar has a gradient.
    np.testing.assert_allclose(float(start_loss), float(plain_loss), rtol=1e-6)
    assert float(start_grads.pointer_head.pointer_open) != 0
    np.testing.assert_allclose(float(jnp.sum(per_token)) / float(jnp.sum(weight)), float(loss), rtol=1e-5)
    for name in ("pointer_w", "pointer_b", "pointer_open"):
        assert float(jnp.linalg.norm(getattr(grads.pointer_head, name))) > 0, name
        assert getattr(mask.pointer_head, name) == "adam"
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
    # Changing tokens from position 64 on moves no loss before position 63 (whose label is token 64).
    np.testing.assert_allclose(changed[:, :63], per_token[:, :63], atol=1e-5)
