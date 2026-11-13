# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``spelling_byte_embeds``: the space/case-normalised byte tables of the stored vocabulary, and a tiny
model whose input rows add the byte, space and case rows."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track import spelling_bytes
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.spelling_bytes import SpellingCase


def test_spelling_variants_share_bytes_and_keep_space_and_case():
    surfaces = [b" The", b"the", b"THE", b" McDonald", b"A", b"\xc3\x89COLE", b"\xe2\x80", b"", b"internationally"]
    tables = spelling_bytes.spelling_tables_from_surfaces(surfaces)
    np.testing.assert_array_equal(tables.byte_ids[:3], [[116, 104, 101, 0, 0, 0, 0, 0]] * 3)
    np.testing.assert_array_equal(tables.byte_present[:3], [[1, 1, 1, 0, 0, 0, 0, 0]] * 3)
    np.testing.assert_array_equal(tables.leading_space, [1, 0, 0, 1, 0, 0, 0, 0, 0])
    expected_case = ["CAPITALISED", "LOWER", "ALL_CAPS", "MIXED", "CAPITALISED", "ALL_CAPS", "LOWER", "LOWER", "LOWER"]
    np.testing.assert_array_equal(tables.case, [SpellingCase[name] for name in expected_case])
    # Lowercasing is Unicode-aware for whole characters; a partial character keeps its bytes.
    np.testing.assert_array_equal(tables.byte_ids[5, :6], list("école".encode()))
    np.testing.assert_array_equal(tables.byte_ids[6, :2], [0xE2, 0x80])
    assert tables.byte_present[7].sum() == 0
    np.testing.assert_array_equal(tables.byte_ids[8], list(b"internat"))


def test_stored_vocabulary_groups_spelling_variants():
    surfaces = spelling_bytes.stored_surfaces()
    tables = spelling_bytes.spelling_tables(len(surfaces))
    ids = [surfaces.index(s) for s in (b" The", b"the", b"The", b" the")]
    for i in ids[1:]:
        np.testing.assert_array_equal(tables.byte_ids[i], tables.byte_ids[ids[0]])
    np.testing.assert_array_equal(tables.leading_space[ids], [1, 0, 0, 1])


def test_tiny_model_adds_and_trains_the_byte_rows():
    # The first 128 ids are single bytes; 512 includes leading-space words such as " the".
    config = dataclasses.replace(tiny_config(), vocab_size=512, spelling_byte_embeds=True)
    tables = spelling_bytes.spelling_tables(config.vocab_size)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, spelling_byte_embeds=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised rows: the model starts as the baseline.
        np.testing.assert_array_equal(forward(model, tokens), forward(plain, tokens))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)
        rows = (model.token_embed_bytes, model.token_embed_space, model.token_embed_case)
        keys = jax.random.split(jax.random.key(2), len(rows))
        random_rows = tuple(jax.random.normal(k, x.shape) for k, x in zip(keys, rows, strict=True))
        filled = eqx.tree_at(
            lambda m: (m.token_embed_bytes, m.token_embed_space, m.token_embed_case), model, random_rows
        )
        table = filled.input_embedding_table()

    assert np.isfinite(float(loss))
    for name in ("token_embed_bytes", "token_embed_space", "token_embed_case"):
        assert float(jnp.linalg.norm(getattr(grads, name))) > 0, name
    token = int(np.argmax(tables.byte_present.sum(1)))
    expected = filled.token_embed[token] + filled.token_embed_case[tables.case[token]]
    expected = expected + tables.leading_space[token] * filled.token_embed_space
    for position in range(int(tables.byte_present[token].sum())):
        expected = expected + filled.token_embed_bytes[position, tables.byte_ids[token, position]]
    np.testing.assert_allclose(table[token], expected, rtol=1e-6, atol=1e-6)
