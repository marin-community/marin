#!/usr/bin/env python3
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

"""
AR-to-Tree-Diffusion Transfer for Kelp.

This module provides utilities to initialize tree diffusion models from
pretrained autoregressive LLMs (specifically Marin 8B).

Key insight: AST node types (e.g., "FunctionDef", "Return", "BinOp") can be
mapped to token sequences in the LLM vocabulary. We tokenize each node type
name and use the mean-pooled token embeddings as initialization.

Transfer strategies:
1. Embedding transfer: Initialize node/value embeddings from LLM token embeddings
2. Transformer transfer: Copy transformer layer weights (requires matching dims)
3. Projection transfer: Project high-dim LLM embeddings to lower-dim tree model

References:
- Dream: Diffusion Rectification and Estimation-Adaptive Models
- DiffuGPT: AR-to-diffusion transfer for language models
- Berkeley Tree Diffusion: arXiv:2405.20519
"""

import logging
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax import Axis

from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logger = logging.getLogger(__name__)

# Marin 8B configuration (LLaMA 3 architecture)
MARIN_8B_CONFIG = {
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "mlp_dim": 14336,
    "vocab_size": 128256,  # LLaMA 3 vocab size
    "model_id": "marin-community/marin-8b-base",
}


@dataclass
class TransferConfig:
    """Configuration for AR-to-tree-diffusion transfer."""

    # Source model
    source_model_id: str = MARIN_8B_CONFIG["model_id"]
    source_revision: str = "main"  # or "deeper-starling"

    # Transfer strategy
    strategy: Literal["embedding", "projection", "transformer"] = "projection"

    # For projection strategy: target dimension
    target_hidden_dim: int = 256

    # Which components to transfer
    transfer_node_embeddings: bool = True
    transfer_value_embeddings: bool = True
    transfer_transformer_layers: bool = False  # Requires matching dims

    # Projection method
    projection_method: Literal["pca", "random", "learned"] = "random"

    # Random projection seed
    seed: int = 42


def load_marin_embeddings(
    model_id: str = MARIN_8B_CONFIG["model_id"],
    revision: str = "main",
    device: str = "cpu",
) -> tuple[np.ndarray, "AutoTokenizer"]:  # noqa: F821
    """Load Marin 8B token embeddings.

    Returns:
        embeddings: numpy array of shape (vocab_size, hidden_dim)
        tokenizer: HuggingFace tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    logger.info(f"Loading Marin embeddings from {model_id} (revision: {revision})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # Load model (only need embeddings, so we can use low memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype="float32",
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Extract embedding weights
    embeddings = model.model.embed_tokens.weight.detach().cpu().numpy()
    logger.info(f"Loaded embeddings with shape {embeddings.shape}")

    # Clean up model to free memory
    del model

    return embeddings, tokenizer


def tokenize_ast_node_types(
    node_vocab: PythonNodeVocab,
    tokenizer: "AutoTokenizer",  # noqa: F821
) -> dict[str, list[int]]:
    """Tokenize AST node type names using the LLM tokenizer.

    For example:
        "FunctionDef" -> [Function, Def] -> [token_id_1, token_id_2]

    Returns:
        Dictionary mapping node type names to token ID lists
    """
    node_to_tokens = {}

    for node_type in node_vocab.node_types:
        # Tokenize the node type name
        # Add spaces to help tokenizer split compound words
        spaced_name = "".join(
            f" {c}" if c.isupper() and i > 0 else c for i, c in enumerate(node_type)
        ).strip()

        tokens = tokenizer.encode(spaced_name, add_special_tokens=False)
        node_to_tokens[node_type] = tokens

    return node_to_tokens


def tokenize_value_tokens(
    value_vocab: PythonValueVocab,
    tokenizer: "AutoTokenizer",  # noqa: F821
) -> dict[str, list[int]]:
    """Tokenize value vocabulary items using the LLM tokenizer."""
    value_to_tokens = {}

    for value in value_vocab.vocab:
        tokens = tokenizer.encode(str(value), add_special_tokens=False)
        value_to_tokens[value] = tokens if tokens else [tokenizer.unk_token_id or 0]

    return value_to_tokens


def compute_mean_pooled_embeddings(
    token_mapping: dict[str, list[int]],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute mean-pooled embeddings for each vocabulary item.

    Args:
        token_mapping: Dictionary mapping names to token ID lists
        embeddings: Full embedding matrix (vocab_size, hidden_dim)

    Returns:
        Mean-pooled embeddings (len(token_mapping), hidden_dim)
    """
    pooled = []
    for name, token_ids in token_mapping.items():
        if not token_ids:
            # Use zero vector for empty mappings
            pooled.append(np.zeros(embeddings.shape[1]))
        else:
            # Mean pool over tokens
            token_embeds = embeddings[token_ids]
            pooled.append(np.mean(token_embeds, axis=0))

    return np.stack(pooled)


def create_random_projection(
    source_dim: int,
    target_dim: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Create a random projection matrix.

    Uses Johnson-Lindenstrauss lemma: random projections approximately
    preserve distances with high probability.

    Args:
        source_dim: Input dimension (e.g., 4096 for Marin 8B)
        target_dim: Output dimension (e.g., 256 for tree model)
        key: JAX random key

    Returns:
        Projection matrix of shape (source_dim, target_dim)
    """
    # Random Gaussian projection, scaled for variance preservation
    projection = jrandom.normal(key, (source_dim, target_dim))
    projection = projection / jnp.sqrt(source_dim)
    return projection


def project_embeddings(
    embeddings: np.ndarray,
    target_dim: int,
    method: str = "random",
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Project high-dimensional embeddings to target dimension.

    Args:
        embeddings: Source embeddings (vocab_size, source_dim)
        target_dim: Target dimension
        method: Projection method ("random", "pca", or "truncate")
        key: Random key for random projection

    Returns:
        Projected embeddings (vocab_size, target_dim)
    """
    source_dim = embeddings.shape[1]

    if source_dim == target_dim:
        return jnp.array(embeddings)

    if source_dim < target_dim:
        # Pad with zeros
        padding = np.zeros((embeddings.shape[0], target_dim - source_dim))
        return jnp.array(np.concatenate([embeddings, padding], axis=1))

    if method == "truncate":
        # Simple truncation (not recommended)
        return jnp.array(embeddings[:, :target_dim])

    elif method == "pca":
        # PCA projection
        from sklearn.decomposition import PCA

        pca = PCA(n_components=target_dim)
        projected = pca.fit_transform(embeddings)
        logger.info(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        return jnp.array(projected)

    elif method == "random":
        # Random projection (Johnson-Lindenstrauss)
        if key is None:
            key = jrandom.PRNGKey(42)
        projection = create_random_projection(source_dim, target_dim, key)
        return jnp.array(embeddings) @ projection

    else:
        raise ValueError(f"Unknown projection method: {method}")


def initialize_tree_embeddings_from_marin(
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    target_hidden_dim: int,
    config: TransferConfig | None = None,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize tree diffusion embeddings from Marin 8B.

    This is the main entry point for embedding transfer.

    Args:
        node_vocab: Python AST node vocabulary
        value_vocab: Python value vocabulary
        target_hidden_dim: Target embedding dimension for tree model
        config: Transfer configuration
        key: Random key

    Returns:
        node_embeddings: (num_node_types, target_hidden_dim)
        value_embeddings: (num_value_tokens, target_hidden_dim)
    """
    if config is None:
        config = TransferConfig(target_hidden_dim=target_hidden_dim)

    if key is None:
        key = jrandom.PRNGKey(config.seed)

    # Load Marin embeddings
    marin_embeddings, tokenizer = load_marin_embeddings(
        config.source_model_id,
        config.source_revision,
    )

    # Tokenize AST vocabulary
    logger.info("Tokenizing AST node types...")
    node_token_mapping = tokenize_ast_node_types(node_vocab, tokenizer)

    logger.info("Tokenizing value vocabulary...")
    value_token_mapping = tokenize_value_tokens(value_vocab, tokenizer)

    # Compute mean-pooled embeddings
    logger.info("Computing mean-pooled embeddings...")
    node_pooled = compute_mean_pooled_embeddings(node_token_mapping, marin_embeddings)
    value_pooled = compute_mean_pooled_embeddings(value_token_mapping, marin_embeddings)

    # Project to target dimension
    logger.info(f"Projecting to {target_hidden_dim} dimensions using {config.projection_method}...")
    key1, key2 = jrandom.split(key)

    node_embeddings = project_embeddings(
        node_pooled,
        target_hidden_dim,
        method=config.projection_method,
        key=key1,
    )

    value_embeddings = project_embeddings(
        value_pooled,
        target_hidden_dim,
        method=config.projection_method,
        key=key2,
    )

    logger.info(f"Node embeddings shape: {node_embeddings.shape}")
    logger.info(f"Value embeddings shape: {value_embeddings.shape}")

    return node_embeddings, value_embeddings


def initialize_model_from_marin(
    tree_config: TreeDiffusionConfig,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    transfer_config: TransferConfig | None = None,
    key: jax.Array | None = None,
) -> TreeDiffusionModel:
    """Initialize a full TreeDiffusionModel with Marin 8B transfer.

    This creates a new model and replaces the embedding weights with
    transferred embeddings from Marin 8B.

    Args:
        tree_config: Tree diffusion model configuration
        node_vocab: Python AST node vocabulary
        value_vocab: Python value vocabulary
        transfer_config: Transfer configuration
        key: Random key for model initialization

    Returns:
        TreeDiffusionModel with transferred embeddings
    """
    if transfer_config is None:
        transfer_config = TransferConfig(target_hidden_dim=tree_config.hidden_dim)

    if key is None:
        key = jrandom.PRNGKey(transfer_config.seed)

    key, model_key, transfer_key = jrandom.split(key, 3)

    # Initialize base model randomly
    logger.info("Initializing base tree diffusion model...")
    model = TreeDiffusionModel.init(tree_config, key=model_key)

    # Get transferred embeddings
    logger.info("Transferring embeddings from Marin 8B...")
    node_embeddings, value_embeddings = initialize_tree_embeddings_from_marin(
        node_vocab,
        value_vocab,
        tree_config.hidden_dim,
        transfer_config,
        transfer_key,
    )

    # Replace embedding weights in the model
    # The TreeEmbedding has node_embed and value_embed as hax.nn.Embedding modules
    def replace_embeddings(model):
        # Create new embedding arrays with correct Haliax axes
        new_node_weight = hax.named(
            node_embeddings,
            (tree_config.NodeVocab, tree_config.Embed),
        )
        new_value_weight = hax.named(
            value_embeddings,
            (tree_config.ValueVocab, tree_config.Embed),
        )

        # Replace the embedding weights
        new_node_embed = eqx.tree_at(
            lambda e: e.weight,
            model.encoder.embedding.node_embed,
            new_node_weight,
        )
        new_value_embed = eqx.tree_at(
            lambda e: e.weight,
            model.encoder.embedding.value_embed,
            new_value_weight,
        )

        # Update the embedding module
        new_embedding = eqx.tree_at(
            lambda e: (e.node_embed, e.value_embed),
            model.encoder.embedding,
            (new_node_embed, new_value_embed),
        )

        # Update the encoder
        new_encoder = eqx.tree_at(
            lambda e: e.embedding,
            model.encoder,
            new_embedding,
        )

        # Update the model
        return eqx.tree_at(
            lambda m: m.encoder,
            model,
            new_encoder,
        )

    model = replace_embeddings(model)
    logger.info("Embedding transfer complete!")

    return model


# Convenience function for quick experiments
def create_marin_initialized_model(
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    max_nodes: int = 128,
    projection_method: str = "random",
    seed: int = 42,
) -> TreeDiffusionModel:
    """Create a tree diffusion model initialized from Marin 8B.

    This is a convenience function for quick experiments.

    Args:
        hidden_dim: Hidden dimension for tree model
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_nodes: Maximum nodes in tree
        projection_method: How to project Marin embeddings ("random", "pca")
        seed: Random seed

    Returns:
        Initialized TreeDiffusionModel
    """
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    tree_config = TreeDiffusionConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=hidden_dim * 4,
        max_nodes=max_nodes,
        node_vocab_size=node_vocab.vocab_size,
        value_vocab_size=value_vocab.vocab_size,
    )

    transfer_config = TransferConfig(
        target_hidden_dim=hidden_dim,
        projection_method=projection_method,
        seed=seed,
    )

    key = jrandom.PRNGKey(seed)

    return initialize_model_from_marin(
        tree_config,
        node_vocab,
        value_vocab,
        transfer_config,
        key,
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing AR transfer from Marin 8B...")

    # Test embedding loading and projection
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    print(f"Node vocab size: {node_vocab.vocab_size}")
    print(f"Value vocab size: {value_vocab.vocab_size}")

    # This requires transformers to be installed and model access
    try:
        node_emb, value_emb = initialize_tree_embeddings_from_marin(
            node_vocab,
            value_vocab,
            target_hidden_dim=256,
        )
        print(f"Node embeddings: {node_emb.shape}")
        print(f"Value embeddings: {value_emb.shape}")

        # Test full model initialization
        model = create_marin_initialized_model(hidden_dim=256)
        num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        print(f"Model parameters: {num_params:,}")

    except ImportError as e:
        print(f"Skipping test (missing dependency): {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
