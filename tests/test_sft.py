# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox
import jax
import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.main.sft import reinitialize_some_tokens
from levanter.models.llama import LlamaEmbedding


class TestModel(equinox.Module):
    __test__ = False
    Vocab: hax.Axis
    embeddings: LlamaEmbedding
    lm_head: hax.nn.Linear


def test_reinitialize_some_tokens(local_gpt2_tokenizer):
    # Setup test data
    embed_dim = 32
    tokens_to_reinit = ["<|test_token|>", "<|another_token|>"]

    # Create a simple tokenizer with our test tokens
    tokenizer = local_gpt2_tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_reinit})
    vocab_size = len(tokenizer)

    # Create axes
    Vocab = hax.Axis("vocab", vocab_size)
    Embed = hax.Axis("embed", embed_dim)

    # Create a simple model with random embeddings
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    embeddings = hax.random.normal(key1, (Vocab, Embed))
    lm_head = hax.nn.Linear.init(In=Embed, Out=Vocab, key=key3)

    # Create a mock model with LlamaEmbedding
    model = TestModel(
        Vocab=Vocab,
        embeddings=LlamaEmbedding(token_embeddings=hax.nn.Embedding(weight=embeddings, Vocab=Vocab, Embed=Embed)),
        lm_head=lm_head,
    )

    # Get token IDs
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens_to_reinit]

    # Call the function with a new key
    new_model = reinitialize_some_tokens(model, tokenizer, tokens_to_reinit, key2)

    # Verify the results
    new_embeddings = new_model.embeddings.token_embeddings.weight

    # Check that the embeddings for our tokens have been updated
    for token_id in token_ids:
        old_embed = embeddings.array[token_id]
        new_embed = new_embeddings.array[token_id]
        assert not jnp.allclose(
            old_embed, new_embed
        ), f"Embedding for token {token_id} was not updated: {old_embed} == {new_embed}"

        # check that the lm head got updated
        assert not jnp.allclose(
            lm_head.weight.array[:, token_id], new_model.lm_head.weight.array[:, token_id]
        ), f"LM head for token {token_id} was not updated"

    # Check that other embeddings remain unchanged
    for i in range(vocab_size):
        if i not in token_ids:
            assert jnp.allclose(
                embeddings.array[i], new_embeddings.array[i]
            ), f"Embedding for token {i} was changed when it shouldn't have been"


def test_reinitialize_some_tokens_invalid_tokens(local_gpt2_tokenizer):
    # Test with tokens not in vocabulary
    tokenizer = local_gpt2_tokenizer
    model = None

    with pytest.raises(ValueError, match="One or more tokens are not in the tokenizer vocabulary"):
        reinitialize_some_tokens(model, tokenizer, ["<mklamnfljkaf>"], jax.random.PRNGKey(0))


def test_reinitialize_some_tokens_empty_list(local_gpt2_tokenizer):
    tokenizer = local_gpt2_tokenizer
    model = None

    with pytest.raises(ValueError, match="No tokens to reinitialize"):
        reinitialize_some_tokens(model, tokenizer, [], jax.random.PRNGKey(0))
