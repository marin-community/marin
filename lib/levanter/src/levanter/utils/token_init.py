# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for (re)initializing token-specific parameters such as embeddings and
LM heads when expanding or adapting vocabularies.
"""

from typing import Iterable

import jax.random as jrandom
from lenses import lens

import haliax as hax

from levanter.utils.hf_utils import HfTokenizer


def reinitialize_some_tokens(
    model,
    tokenizer: HfTokenizer,
    tokens_to_reinit: list[str],
    key,
    donate: bool = False,
    reinit_lm_head: bool = True,
    reinit_embeddings: bool = True,
):
    """
    Reinitialize embeddings (and optionally LM head rows) for a subset of tokens.

    Args:
        model: Model with `embeddings.token_embeddings.weight` and `lm_head.weight`.
        tokenizer: HF tokenizer aligned with the model vocabulary.
        tokens_to_reinit: Tokens whose parameters should be reinitialized.
        key: PRNGKey used for sampling new parameters.
        donate: Whether to donate JAX buffers when jitting.
        reinit_lm_head: Reinitialize corresponding LM head rows when True.
        reinit_embeddings: Reinitialize embedding rows when True.

    Raises:
        ValueError: If `tokens_to_reinit` is empty or contains unknown tokens.
    """
    ids_to_reinit = [tokenizer.convert_tokens_to_ids(token) for token in tokens_to_reinit]
    if len(ids_to_reinit) == 0:
        raise ValueError("No tokens to reinitialize")
    # convert_tokens_to_ids may return the unknown token id; verify exact round-trip.
    elif any(
        token is None or tokenizer.convert_ids_to_tokens(id) != token
        for id, token in zip(ids_to_reinit, tokens_to_reinit)
    ):
        raise ValueError("One or more tokens are not in the tokenizer vocabulary")

    @hax.named_jit(donate_args=(donate,))
    def _reinit_tokens(model):
        Embed = model.embeddings.Embed
        new_vocab = model.Vocab.resize(len(ids_to_reinit))

        emb_key, lm_key = jrandom.split(key, 2)

        embeddings_matrix = model.embeddings.token_embeddings.weight

        if reinit_embeddings:
            new_embeddings = _reinit_embed_vectors(Embed, new_vocab, embeddings_matrix, ids_to_reinit, emb_key)
            model = lens.embeddings.token_embeddings.weight.set(new_embeddings)(model)

        if reinit_lm_head:
            new_lm_head = _reinit_embed_vectors(Embed, new_vocab, model.lm_head.weight, ids_to_reinit, lm_key)
            model = lens.lm_head.weight.set(new_lm_head)(model)

        return model

    return _reinit_tokens(model)


def _reinit_embed_vectors(Embed, new_vocab, embeddings_matrix, ids_to_reinit: Iterable[int], key):
    # Match the existing embedding statistics to avoid abrupt scale shifts.
    mu = hax.mean(embeddings_matrix, axis="vocab")
    std = hax.std(embeddings_matrix, axis="vocab")
    reinited = hax.random.truncated_normal(key, (new_vocab, Embed), -3, 3) * std + mu
    new_weight = embeddings_matrix.at["vocab", ids_to_reinit].set(reinited)
    return new_weight
