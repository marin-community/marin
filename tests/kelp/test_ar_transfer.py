# Copyright 2026 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Kelp AR transfer utilities."""

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jrandom

from experiments.kelp.ar_transfer import (
    TransferConfig,
    compute_mean_pooled_embeddings,
    create_random_projection,
    project_embeddings,
)
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab


class TestProjection:
    def test_random_projection_shape(self):
        key = jrandom.PRNGKey(0)
        projection = create_random_projection(4096, 256, key)

        assert projection.shape == (4096, 256)

    def test_random_projection_variance(self):
        # Random projection should approximately preserve variance
        key = jrandom.PRNGKey(0)
        projection = create_random_projection(4096, 256, key)

        # Check that the projection has reasonable variance
        assert jnp.std(projection) > 0.01
        assert jnp.std(projection) < 0.1

    def test_project_embeddings_truncate(self):
        embeddings = np.random.randn(100, 4096).astype(np.float32)
        projected = project_embeddings(embeddings, 256, method="truncate")

        assert projected.shape == (100, 256)
        # Truncate should just take first 256 dims
        np.testing.assert_array_almost_equal(
            projected[:, :10], embeddings[:, :10], decimal=5
        )

    def test_project_embeddings_random(self):
        embeddings = np.random.randn(100, 4096).astype(np.float32)
        key = jrandom.PRNGKey(42)
        projected = project_embeddings(embeddings, 256, method="random", key=key)

        assert projected.shape == (100, 256)

    def test_project_embeddings_identity(self):
        # If source_dim == target_dim, should return unchanged
        embeddings = np.random.randn(100, 256).astype(np.float32)
        projected = project_embeddings(embeddings, 256, method="random")

        assert projected.shape == (100, 256)
        np.testing.assert_array_almost_equal(projected, embeddings, decimal=5)

    def test_project_embeddings_padding(self):
        # If source_dim < target_dim, should pad with zeros
        embeddings = np.random.randn(100, 128).astype(np.float32)
        projected = project_embeddings(embeddings, 256, method="random")

        assert projected.shape == (100, 256)
        # First 128 dims should be original
        np.testing.assert_array_almost_equal(projected[:, :128], embeddings, decimal=5)
        # Rest should be zeros
        np.testing.assert_array_almost_equal(
            projected[:, 128:], np.zeros((100, 128)), decimal=5
        )


class TestMeanPooling:
    def test_mean_pooled_embeddings(self):
        # Create mock embeddings
        embeddings = np.arange(1000).reshape(100, 10).astype(np.float32)

        # Create token mapping
        token_mapping = {
            "FunctionDef": [0, 1, 2],  # Should average rows 0, 1, 2
            "Return": [50],  # Should be row 50
            "Empty": [],  # Should be zeros
        }

        pooled = compute_mean_pooled_embeddings(token_mapping, embeddings)

        assert pooled.shape == (3, 10)

        # Check FunctionDef (average of rows 0, 1, 2)
        expected_func = np.mean(embeddings[[0, 1, 2]], axis=0)
        np.testing.assert_array_almost_equal(pooled[0], expected_func, decimal=5)

        # Check Return (just row 50)
        np.testing.assert_array_almost_equal(pooled[1], embeddings[50], decimal=5)

        # Check Empty (zeros)
        np.testing.assert_array_almost_equal(pooled[2], np.zeros(10), decimal=5)


class TestTransferConfig:
    def test_default_config(self):
        config = TransferConfig()

        assert config.source_model_id == "marin-community/marin-8b-base"
        assert config.strategy == "projection"
        assert config.target_hidden_dim == 256
        assert config.transfer_node_embeddings is True
        assert config.transfer_value_embeddings is True

    def test_custom_config(self):
        config = TransferConfig(
            target_hidden_dim=512,
            projection_method="pca",
            seed=123,
        )

        assert config.target_hidden_dim == 512
        assert config.projection_method == "pca"
        assert config.seed == 123


class TestVocabularies:
    def test_node_vocab_for_transfer(self):
        vocab = PythonNodeVocab()

        # Check we have expected node types for tokenization
        assert "FunctionDef" in vocab.node_types
        assert "Return" in vocab.node_types
        assert "BinOp" in vocab.node_types
        assert vocab.vocab_size > 0

    def test_value_vocab_for_transfer(self):
        vocab = PythonValueVocab()

        # Check we have expected value tokens
        assert vocab.vocab_size > 0


# Skip tests that require transformers/model download
@pytest.mark.skip(reason="Requires transformers and model download")
class TestMarinLoading:
    def test_load_marin_embeddings(self):
        from experiments.kelp.ar_transfer import load_marin_embeddings

        embeddings, tokenizer = load_marin_embeddings()

        # Marin 8B has vocab_size ~128k and hidden_dim 4096
        assert embeddings.shape[1] == 4096
        assert embeddings.shape[0] > 100000

    def test_tokenize_ast_node_types(self):
        from experiments.kelp.ar_transfer import (
            load_marin_embeddings,
            tokenize_ast_node_types,
        )

        _, tokenizer = load_marin_embeddings()
        node_vocab = PythonNodeVocab()

        token_mapping = tokenize_ast_node_types(node_vocab, tokenizer)

        # Should have entry for each node type
        assert len(token_mapping) == node_vocab.vocab_size

        # Each entry should be a list of token IDs
        for node_type, tokens in token_mapping.items():
            assert isinstance(tokens, list)
            assert all(isinstance(t, int) for t in tokens)

    def test_full_transfer_pipeline(self):
        from experiments.kelp.ar_transfer import initialize_tree_embeddings_from_marin

        node_vocab = PythonNodeVocab()
        value_vocab = PythonValueVocab()

        node_emb, value_emb = initialize_tree_embeddings_from_marin(
            node_vocab,
            value_vocab,
            target_hidden_dim=256,
        )

        assert node_emb.shape == (node_vocab.vocab_size, 256)
        assert value_emb.shape == (value_vocab.vocab_size, 256)
