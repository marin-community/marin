# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``kda_local_attn_window``: the blocked window attention against a direct per-query softmax with its
offset bias and document masking, and a tiny model that starts as the baseline, trains the gain, and routes the new
leaves to their groups."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer, _window_attention
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

WINDOW = 4


def test_window_attention_matches_direct_softmax_within_documents():
    batch, seq, heads, width = 2, 11, 2, 3  # seq is not a multiple of the window
    segments = jnp.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    q, k, v = (jax.random.normal(jax.random.key(i), (batch, seq, heads, width)) for i in range(3))
    bias = jax.random.normal(jax.random.key(3), (heads, WINDOW))
    with jax.set_mesh(tiny_mesh()):
        got = _window_attention(q, k, v, bias, {"segment_ids": segments}, window=WINDOW)

    q, k, v, bias, seg = (np.asarray(t, np.float64) for t in (q, k, v, bias, segments))
    want = np.zeros_like(q)
    for b in range(batch):
        for t in range(seq):
            keys = [s for s in range(max(0, t - WINDOW + 1), t + 1) if seg[b, s] == seg[b, t]]
            for n in range(heads):
                logits = np.array([q[b, t, n] @ k[b, s, n] + bias[n, t - s] for s in keys])
                probs = np.exp(logits - logits.max())
                want[b, t, n] = (probs / probs.sum()) @ v[b, keys, n]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_tiny_model_starts_as_baseline_and_trains_the_gain():
    config = dataclasses.replace(tiny_config(), kda_local_attn_window=WINDOW)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, kda_local_attn_window=0), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised output gain: the branch adds nothing at step 0.
        np.testing.assert_allclose(forward(model, tokens), forward(plain, tokens), rtol=1e-6, atol=1e-6)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)

    assert np.isfinite(float(loss))
    assert float(jnp.linalg.norm(grads.kda_blocks.stacked.attn.local_attn.out_gain)) > 0


def test_local_attn_leaves_route_to_muonh_and_adam():
    config = dataclasses.replace(tiny_config(), kda_local_attn_window=WINDOW)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    labels = {jax.tree_util.keystr(path): label for path, label in jax.tree_util.tree_flatten_with_path(mask)[0]}

    def labels_ending(leaf: str) -> set[str]:
        found = {label for name, label in labels.items() if name.endswith(f".local_attn.{leaf}")}
        assert found, leaf
        return found

    for leaf in ("w_q", "w_k", "w_v", "w_o"):
        assert labels_ending(leaf) == {"muonh"}
    for leaf in ("logit_scale", "offset_bias", "out_gain"):
        assert labels_ending(leaf) == {"adam"}
