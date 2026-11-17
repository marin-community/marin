# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``spelling_byte_source``: a tiny model that reads the spelling byte rows as their own AttnRes source,
leaving the input embedding table unchanged."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.spelling_bytes import spelling_tables


def test_tiny_model_reads_the_byte_source():
    # The first 128 ids are single bytes; 512 includes leading-space words such as " the".
    config = dataclasses.replace(tiny_config(), vocab_size=512, spelling_byte_source=True)
    tables = spelling_tables(config.vocab_size)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)
        _, stats = eqx.filter_jit(lambda m, t: m(t))(model, tokens)
        table = model.spelling_byte_table()

    assert np.isfinite(float(loss))
    for name in ("token_embed_source_bytes", "token_embed_source_space", "token_embed_source_case"):
        assert float(jnp.linalg.norm(getattr(grads, name))) > 0, name
    assert float(jnp.linalg.norm(grads.spelling_byte_norm.weight)) > 0
    # The input embedding stays the plain token table.
    assert model.input_embedding_table() is model.token_embed
    # The final gate (8 in the 4-layer model) reads the byte source, the embedding and the layer-2 block.
    assert "attn_res_w_g08_b02" in stats

    token = int(np.argmax(tables.byte_present.sum(1)))
    expected = model.token_embed_source_case[tables.case[token]]
    expected = expected + tables.leading_space[token] * model.token_embed_source_space
    for position in range(int(tables.byte_present[token].sum())):
        expected = expected + model.token_embed_source_bytes[position, tables.byte_ids[token, position]]
    np.testing.assert_allclose(table[token], expected, rtol=1e-5, atol=1e-6)
