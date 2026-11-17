# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``spelling_ngram_source``: the character n-gram tables of the stored vocabulary, and a tiny model that
reads the n-gram feature as its own AttnRes source."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track import spelling_ngrams
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.spelling_bytes import spelling_tables


def test_ngrams_are_normalised_and_boundary_marked():
    start, end = spelling_ngrams._WORD_START, spelling_ngrams._WORD_END
    the = spelling_ngrams.spelling_ngrams(b" The")
    assert the == spelling_ngrams.spelling_ngrams(b"the")
    t, h, e = map(ord, "the")
    assert {(start,), (t,), (start, t, h, e), (t, h, e, end), (e, end)} <= the
    assert (start, t, h, e, end) not in the  # longer than MAX_NGRAM
    assert spelling_ngrams.spelling_ngrams(b"   ") == frozenset()


def test_tables_keep_only_ngrams_shared_by_two_tokens():
    surfaces = [b" the", b"then", b"xq", b""]
    tables = spelling_ngrams.ngram_tables_from_surfaces(surfaces)
    shared = spelling_ngrams.spelling_ngrams(b"the") & spelling_ngrams.spelling_ngrams(b"then")
    # "xq"'s only n-grams shared with another token are the two boundary markers.
    assert tables.num_ngrams == len(shared)
    np.testing.assert_array_equal(tables.ngram_present.sum(1), [len(shared), len(shared), 2, 0])


def test_tiny_model_reads_the_ngram_source():
    # The first 128 ids are single bytes; 512 includes leading-space words such as " the".
    config = dataclasses.replace(tiny_config(), vocab_size=512, spelling_ngram_source=True)
    spelling = spelling_tables(config.vocab_size)
    ngrams = spelling_ngrams.ngram_tables(config.vocab_size)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)
        hidden, stats = eqx.filter_jit(lambda m, t: m(t))(model, tokens)
        table = model.spelling_ngram_table()

    assert np.isfinite(float(loss))
    for name in ("token_embed_ngrams", "token_embed_ngram_space", "token_embed_ngram_case"):
        assert float(jnp.linalg.norm(getattr(grads, name))) > 0, name
    assert float(jnp.linalg.norm(grads.spelling_ngram_norm.weight)) > 0
    # The final gate (8 in the 4-layer model) reads the n-gram source, the embedding and the layer-2 block.
    assert "attn_res_w_g08_b02" in stats
    assert hidden.shape == (2, config.max_seq_len, config.hidden_dim)

    token = int(np.argmax(ngrams.ngram_present.sum(1)))
    expected = model.token_embed_ngram_case[spelling.case[token]]
    expected = expected + spelling.leading_space[token] * model.token_embed_ngram_space
    for slot in range(int(ngrams.ngram_present[token].sum())):
        expected = expected + model.token_embed_ngrams[ngrams.ngram_ids[token, slot]]
    np.testing.assert_allclose(table[token], expected, rtol=1e-5, atol=1e-6)
