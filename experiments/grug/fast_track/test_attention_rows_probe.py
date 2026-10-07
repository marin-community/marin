# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``attention_rows_probe``: full softmax rows for traced queries match the static ``forward_probe`` rows."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import (
    ATTN_PROBE_KEYS,
    ATTN_PROBE_STAT,
    ATTN_ROWS_STAT,
    ForwardProbe,
    attention_rows_probe,
    forward_probe,
)


def test_traced_rows_match_the_static_probe_with_inkling_and_documents():
    seq = 128
    mesh, model = t._model(ngram_stat_rows=0, mla=True, inkling_relpos=True, max_seq_len=seq, sliding_window=seq)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, seq), 0, t._VOCAB)
    segments = jnp.zeros((2, seq), jnp.int32).at[1, 50:].set(1)
    mask = AttentionMask.causal().with_segment_ids(segments)
    queries = ((0, 100), (1, 70), (1, 30))
    with jax.set_mesh(mesh):
        with forward_probe(ForwardProbe(attn_queries=queries, spot=None, spot_token_ids=())):
            _, static = eqx.filter_jit(lambda m: m(tokens, mask=mask))(model)

        def traced(m, rows, positions):
            with attention_rows_probe(rows, positions):
                return m(tokens, mask=mask)[1]

        rows = jnp.asarray([r for r, _ in queries])
        positions = jnp.asarray([p for _, p in queries])
        dynamic = eqx.filter_jit(traced)(model, rows, positions)
    static_key = next(k for k in static if k.startswith(ATTN_PROBE_STAT))
    layer = static_key.rsplit("_L", 1)[1]
    full = np.asarray(dynamic[f"{ATTN_ROWS_STAT}_L{layer}"])  # [Q, H, S]
    window = np.asarray(static[static_key])  # [Q, H, ATTN_PROBE_KEYS], keys position-K+1 .. position
    for i, (_, position) in enumerate(queries):
        lo = position - ATTN_PROBE_KEYS + 1
        expected = window[i][:, max(0, -lo) :]
        np.testing.assert_allclose(full[i][:, max(0, lo) : position + 1], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(full[i].sum(-1), 1.0, rtol=1e-5)
    assert np.all(full[1][:, :50] == 0)  # query (1, 70) is in document 1
