# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``ngram_cache_mixture``: the sorted-rank counts against a brute-force scan, the mixture loss against a
dense full-vocabulary mixture, and a tiny model that starts as the control and trains through the head."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track import ngram_cache
from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_BATCH_AXES = ("replica_dcn", "data", "expert")


def _brute_force_counts(tokens: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """``[4, B, S]``: c1, n1, c2, n2 by scanning every earlier position of the same document."""
    out = np.zeros((4, *tokens.shape), int)
    for row in range(tokens.shape[0]):
        x, seg = tokens[row], segments[row]
        for t in range(len(x)):
            for order, n in enumerate((1, 2)):
                if t - n + 1 < 0 or seg[t - n + 1] != seg[t]:
                    continue
                matches = [
                    s
                    for s in range(n - 1, t)
                    if seg[s - n + 1] == seg[t] and list(x[s - n + 1 : s + 1]) == list(x[t - n + 1 : t + 1])
                ]
                out[2 * order, row, t] = len(matches)
                if t + 1 < len(x) and seg[t + 1] == seg[t]:
                    out[2 * order + 1, row, t] = sum(x[s + 1] == x[t + 1] for s in matches)
    return out


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def test_counts_match_a_brute_force_scan_within_documents():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 3, size=(3, 40))
    segments = np.stack([np.zeros(40, int), np.arange(40) // 13, np.repeat([0, 1, 2, 3], [5, 1, 20, 14])])
    with jax.set_mesh(_mesh()):
        got = ngram_cache.document_ngram_counts(jnp.asarray(tokens), jnp.asarray(segments), 3, _BATCH_AXES)
    np.testing.assert_array_equal(np.stack([np.asarray(c) for c in got]), _brute_force_counts(tokens, segments))


def _dense_distribution(head, h, output, tokens, segments):
    """The full next-token mixture, with q1 and q2 written out over the vocabulary by scanning the document."""
    vocab = output.shape[-1]
    x, seg = np.asarray(tokens), np.asarray(segments)
    q = np.zeros((*x.shape, 2, vocab), np.float32)
    context = np.zeros((*x.shape, 2), np.float32)
    for row in range(x.shape[0]):
        for t in range(x.shape[1]):
            for order, n in enumerate((1, 2)):
                if t - n + 1 < 0 or seg[row, t - n + 1] != seg[row, t]:
                    continue
                for s in range(n - 1, t):
                    if seg[row, s - n + 1] == seg[row, t] and list(x[row, s - n + 1 : s + 1]) == list(
                        x[row, t - n + 1 : t + 1]
                    ):
                        context[row, t, order] += 1
                        q[row, t, order, x[row, s + 1]] += 1
    q = q / np.maximum(context, 1)[..., None]
    p_vocab = jax.nn.softmax(h @ output, -1)
    valid = context > 0
    logits = h @ head.ngram_cache_w + 8 * head.ngram_cache_b * jnp.log2(1 + context) / 12 - 2
    weights = jax.nn.softmax(jnp.concatenate([jnp.zeros_like(logits[..., :1]), jnp.where(valid, logits, -jnp.inf)], -1))
    pi = weights[..., 1:]
    open_gate = jnp.clip(8 * head.ngram_cache_open, 0, 1)
    cached = jnp.einsum("bsk,bskv->bsv", pi, q)
    return (1 - open_gate * pi.sum(-1))[..., None] * p_vocab + open_gate * cached


def test_mixture_loss_matches_the_dense_mixture_values_and_gradients():
    vocab, dim = 4, 5
    keys = iter(jax.random.split(jax.random.key(3), 8))
    head = ngram_cache.NgramCacheHead(
        ngram_cache_w=jax.random.normal(next(keys), (dim, 2)),
        ngram_cache_b=0.3 * jax.random.normal(next(keys), (2,)),
        ngram_cache_open=jnp.asarray(0.07),
    )
    h = jax.random.normal(next(keys), (2, 24, dim))
    output = jax.random.normal(next(keys), (dim, vocab))
    tokens = jax.random.randint(next(keys), (2, 24), 0, vocab)
    labels = jnp.roll(tokens, -1, 1)
    segments = jnp.broadcast_to(jnp.arange(24) // 13, tokens.shape)
    with jax.set_mesh(_mesh()):
        counts = ngram_cache.document_ngram_counts(tokens, segments, vocab, _BATCH_AXES)
    # Unsharded, to meet the reference's off-mesh arrays.
    counts = ngram_cache.NgramCounts(*(jnp.asarray(np.asarray(c)) for c in counts))
    assert np.any(np.asarray(counts.n2) > 0)
    # The last position of each document has a label from the next document (or none): score the in-document ones.
    scored = np.asarray((jnp.roll(segments, -1, 1) == segments).at[:, -1].set(False))

    def actual(head, h):
        logits = h @ output
        ce = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
        return ngram_cache.ngram_cache_mixture_nll(head, h @ head.ngram_cache_w, counts, ce)[0][scored]

    def expected(head, h):
        p = _dense_distribution(head, h, output, tokens, segments)
        return -jnp.log(jnp.take_along_axis(p, labels[..., None], -1)[..., 0])[scored]

    np.testing.assert_allclose(_dense_distribution(head, h, output, tokens, segments).sum(-1), 1, atol=1e-6)
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
        ngram_cache_mixture=True,
    )
    # Repeated spans give the cache counts to find.
    tokens = jnp.tile(jax.random.randint(jax.random.key(1), (2, seq // 4), 0, vocab), (1, 4))
    weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, ngram_cache_mixture=False), key=jax.random.key(0))
        counts = eqx.filter_jit(lambda m: m.ngram_cache_counts(tokens, None))(model)

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4, ngram_counts=counts)

        def per_token_fn(m, tokens):
            return m.next_token_loss(tokens, weight, reduction="none", logsumexp_weight=1e-4)

        start_loss, start_grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        plain_loss = eqx.filter_jit(loss_fn)(plain)
        keys = jax.random.split(jax.random.key(2), 2)
        opened = eqx.tree_at(
            lambda m: (m.ngram_cache_head.ngram_cache_w, m.ngram_cache_head.ngram_cache_b),
            model,
            (0.1 * jax.random.normal(keys[0], (hidden, 2)), 0.1 * jax.random.normal(keys[1], (2,))),
        )
        opened = eqx.tree_at(lambda m: m.ngram_cache_head.ngram_cache_open, opened, jnp.asarray(0.05))
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(opened)
        per_token = eqx.filter_jit(per_token_fn)(opened, tokens)
        changed = eqx.filter_jit(per_token_fn)(opened, tokens.at[:, 64:].set((tokens[:, 64:] + 1) % vocab))
        vocab_only_loss = eqx.filter_jit(loss_fn)(dataclasses.replace(opened, ngram_cache_head=None))
        mask = GrugMoeMuonHConfig().create_mask(model)

    # Zero-initialised, the head leaves the loss at the control's; only the opening scalar has a gradient.
    np.testing.assert_allclose(float(start_loss), float(plain_loss), rtol=1e-6)
    assert float(start_grads.ngram_cache_head.ngram_cache_open) != 0
    # Precomputed counts (the train step) and counts computed in the loss (evals) give the same loss.
    np.testing.assert_allclose(float(jnp.sum(per_token)) / float(jnp.sum(weight)), float(loss), rtol=1e-5)
    # The repeated spans make the open cache predict the label: the mixture beats the vocabulary head alone.
    assert float(loss) < float(vocab_only_loss)
    for name in ("ngram_cache_w", "ngram_cache_b", "ngram_cache_open"):
        assert float(jnp.linalg.norm(getattr(grads.ngram_cache_head, name))) > 0, name
        assert getattr(mask.ngram_cache_head, name) == "adam"
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
    # Changing tokens from position 64 on moves no loss before position 63 (whose label is token 64).
    np.testing.assert_allclose(changed[:, :63], per_token[:, :63], atol=1e-5)
