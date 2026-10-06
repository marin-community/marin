# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``logit_decomp_stats``: per-token split of the logits into the stream head and the head-only extra slice."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.logit_decomp import LOGIT_DECOMP_FIELDS, logit_decomp_stats


def test_full_loss_matches_the_model_and_removing_the_extra_term_changes_it():
    mesh, model = t._model(ngram_stat_rows=0, mla=True, num_layers=4, latent_out_full_layers=(3,), lm_head_extra_dim=16)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    segments = jnp.zeros(tokens.shape, jnp.int32).at[1, 8:].set(1)
    unigram = jnp.log(jnp.full((t._VOCAB,), 1.0 / t._VOCAB))
    with jax.set_mesh(mesh):
        stats = np.asarray(eqx.filter_jit(logit_decomp_stats)(model, tokens, segments, unigram))
        mask = AttentionMask.causal().with_segment_ids(segments)
        valid = np.asarray(stats[..., LOGIT_DECOMP_FIELDS.index("valid")]) > 0
        ref = np.asarray(
            eqx.filter_jit(
                lambda m: m.next_token_loss(tokens, jnp.asarray(valid, jnp.float32), mask=mask, reduction="none")
            )(model)
        )
    field = {name: stats[..., i] for i, name in enumerate(LOGIT_DECOMP_FIELDS)}
    assert stats.shape == (2, t._SEQ, len(LOGIT_DECOMP_FIELDS))
    assert not valid[1, 7] and not valid[0, -1] and valid[0, 0]
    np.testing.assert_allclose(field["loss_full"][valid], ref[valid], rtol=1e-3, atol=1e-3)
    assert np.abs(field["loss_full"] - field["loss_main"])[valid].max() > 0
    assert np.all(np.isfinite(stats))
