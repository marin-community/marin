# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``spelling_ngram_embeds``: a tiny model whose input rows add each token's n-gram, space and case rows,
summed by one segment-sum over the vocabulary's (token, n-gram) pairs."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track import spelling_ngrams
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.spelling_bytes import spelling_tables


def test_pairs_list_each_tokens_kept_ngrams():
    tables = spelling_ngrams.ngram_tables(512)
    pairs = spelling_ngrams.ngram_pairs(512)
    assert np.all(np.diff(pairs.token) >= 0)
    for i in range(512):
        expected = tables.ngram_ids[i, tables.ngram_present[i] > 0]
        np.testing.assert_array_equal(pairs.ngram[pairs.token == i], expected)


def test_tiny_model_adds_and_trains_the_ngram_rows():
    # The first 128 ids are single bytes; 512 includes leading-space words such as " the".
    config = dataclasses.replace(tiny_config(), vocab_size=512, spelling_ngram_embeds=True)
    spelling = spelling_tables(config.vocab_size)
    ngrams = spelling_ngrams.ngram_tables(config.vocab_size)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, spelling_ngram_embeds=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised rows: the model starts as the baseline.
        np.testing.assert_array_equal(forward(model, tokens), forward(plain, tokens))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)
        rows = (model.token_embed_add_ngrams, model.token_embed_add_space, model.token_embed_add_case)
        keys = jax.random.split(jax.random.key(2), len(rows))
        random_rows = tuple(jax.random.normal(k, x.shape) for k, x in zip(keys, rows, strict=True))
        filled = eqx.tree_at(
            lambda m: (m.token_embed_add_ngrams, m.token_embed_add_space, m.token_embed_add_case), model, random_rows
        )
        table = filled.input_embedding_table()

    assert np.isfinite(float(loss))
    for name in ("token_embed_add_ngrams", "token_embed_add_space", "token_embed_add_case"):
        assert float(jnp.linalg.norm(getattr(grads, name))) > 0, name
    token = int(np.argmax(ngrams.ngram_present.sum(1)))
    expected = filled.token_embed[token] + filled.token_embed_add_case[spelling.case[token]]
    expected = expected + spelling.leading_space[token] * filled.token_embed_add_space
    for slot in range(int(ngrams.ngram_present[token].sum())):
        expected = expected + filled.token_embed_add_ngrams[ngrams.ngram_ids[token, slot]]
    np.testing.assert_allclose(table[token], expected, rtol=1e-5, atol=1e-5)
