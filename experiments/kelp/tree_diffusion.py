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
Core tree diffusion model for Python program synthesis.

This module implements the tree diffusion model architecture following
Berkeley's approach (arXiv:2405.20519), adapted for JAX/Equinox and
designed to be compatible with Levanter's training infrastructure.

Architecture:
- TreeEncoder: Encodes AST structure using tree-aware attention
- EditPredictor: Predicts edit location and replacement tokens
- TreeDiffusionModel: Full model combining encoder and predictor
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call


@dataclass(frozen=True)
class TreeDiffusionConfig:
    """Configuration for the tree diffusion model.

    Follows Berkeley's hyperparameters where applicable.
    """

    # Model dimensions (Berkeley: 256 hidden, 8 layers, 16 heads)
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 16
    mlp_dim: int = 1024  # Typically 4x hidden_dim
    dropout: float = 0.1

    # Tree structure limits
    max_nodes: int = 256
    max_depth: int = 32
    max_children: int = 16
    max_value_len: int = 32

    # Vocabulary sizes
    node_vocab_size: int = 128  # Python AST node types
    value_vocab_size: int = 256  # Value tokens

    # Conditioning (signature + docstring)
    use_conditioning: bool = False
    condition_vocab_size: int = 128  # Character-level vocab
    max_condition_len: int = 128

    # Diffusion parameters (Berkeley: sigma_small=2, s_max=5)
    sigma_small: int = 2
    s_max: int = 5

    # Training
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

    # Axis definitions
    @property
    def Nodes(self) -> Axis:
        return Axis("nodes", self.max_nodes)

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Heads(self) -> Axis:
        return Axis("heads", self.num_heads)

    @property
    def HeadDim(self) -> Axis:
        return Axis("head_dim", self.hidden_dim // self.num_heads)

    @property
    def Mlp(self) -> Axis:
        return Axis("mlp", self.mlp_dim)

    @property
    def Depth(self) -> Axis:
        return Axis("depth", self.max_depth)

    @property
    def Children(self) -> Axis:
        return Axis("children", self.max_children)

    @property
    def ValueLen(self) -> Axis:
        return Axis("value_len", self.max_value_len)

    @property
    def NodeVocab(self) -> Axis:
        return Axis("node_vocab", self.node_vocab_size)

    @property
    def ValueVocab(self) -> Axis:
        return Axis("value_vocab", self.value_vocab_size)

    @property
    def Layers(self) -> Axis:
        return Axis("layers", self.num_layers)

    @property
    def CondVocab(self) -> Axis:
        return Axis("cond_vocab", self.condition_vocab_size)

    @property
    def CondLen(self) -> Axis:
        return Axis("cond_len", self.max_condition_len)


class TreeEmbedding(eqx.Module):
    """Embedding layer for tree structure.

    Combines:
    - Node type embeddings
    - Node value embeddings (pooled across value tokens)
    - Depth embeddings (positional encoding for tree depth)

    We use simple addition to combine embeddings rather than concatenation
    to avoid dimension blowup and linear projection issues.
    """

    node_embed: hnn.Embedding
    value_embed: hnn.Embedding
    depth_embed: hnn.Embedding
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeEmbedding":
        k1, k2, k3 = jrandom.split(key, 3)

        node_embed = hnn.Embedding.init(config.NodeVocab, config.Embed, key=k1)
        value_embed = hnn.Embedding.init(config.ValueVocab, config.Embed, key=k2)
        depth_embed = hnn.Embedding.init(config.Depth, config.Embed, key=k3)

        return TreeEmbedding(node_embed, value_embed, depth_embed, config)

    @named_call
    def __call__(
        self,
        node_types: NamedArray,  # (Nodes,) int
        node_values: NamedArray,  # (Nodes, ValueLen) int
        depth: NamedArray,  # (Nodes,) int
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Embed tree structure.

        Args:
            node_types: Node type IDs
            node_values: Value token IDs
            depth: Node depths in tree

        Returns:
            Node embeddings of shape (Nodes, Embed)
        """
        # Node type embeddings: (Nodes, Embed)
        node_emb = self.node_embed(node_types)

        # Value embeddings: (Nodes, ValueLen, Embed) -> mean pool -> (Nodes, Embed)
        value_emb = self.value_embed(node_values)
        value_emb = hax.mean(value_emb, axis=self.config.ValueLen)

        # Depth embeddings: (Nodes, Embed)
        # Clamp depth to valid range
        depth_clamped = hax.clip(depth, 0, self.config.max_depth - 1)
        depth_emb = self.depth_embed(depth_clamped)

        # Combine via addition (like standard transformer positional encoding)
        output = node_emb + value_emb + depth_emb

        return output


class TreeAttention(eqx.Module):
    """Tree-aware multi-head attention.

    Standard multi-head attention with optional tree structure masking.
    The tree structure is encoded through the depth embeddings and
    can optionally use parent-child masking.
    """

    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeAttention":
        k1, k2, k3, k4 = jrandom.split(key, 4)

        # Combined dimension for all heads
        HeadEmbed = Axis("head_embed", config.num_heads * (config.hidden_dim // config.num_heads))

        q_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k1, use_bias=False)
        k_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k2, use_bias=False)
        v_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k3, use_bias=False)
        o_proj = hnn.Linear.init(In=HeadEmbed, Out=config.Embed, key=k4, use_bias=False)

        return TreeAttention(q_proj, k_proj, v_proj, o_proj, config)

    @named_call
    def __call__(
        self,
        x: NamedArray,  # (Nodes, Embed)
        mask: NamedArray | None = None,  # (Nodes,) bool - valid node mask
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Apply tree-aware attention.

        Args:
            x: Input embeddings
            mask: Optional mask for valid nodes

        Returns:
            Output embeddings of same shape as input
        """
        Nodes = self.config.Nodes
        Heads = self.config.Heads
        HeadDim = self.config.HeadDim

        # Project to Q, K, V
        q = self.q_proj(x)  # (Nodes, HeadEmbed)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (Nodes, Heads, HeadDim)
        HeadEmbed = Axis("head_embed", self.config.num_heads * (self.config.hidden_dim // self.config.num_heads))
        q = q.unflatten_axis(HeadEmbed, (Heads, HeadDim))
        k = k.unflatten_axis(HeadEmbed, (Heads, HeadDim))
        v = v.unflatten_axis(HeadEmbed, (Heads, HeadDim))

        # Compute attention scores
        # (Nodes, Heads, HeadDim) @ (Nodes, Heads, HeadDim).T -> (Nodes, Heads, Nodes)
        KeyNodes = Nodes.alias("key_nodes")
        k_t = k.rename({Nodes: KeyNodes})
        v_t = v.rename({Nodes: KeyNodes})

        scale = 1.0 / jnp.sqrt(self.config.hidden_dim // self.config.num_heads)
        scores = hax.dot(q, k_t, axis=HeadDim) * scale  # (Nodes, Heads, KeyNodes)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for key dimension
            key_mask = mask.rename({Nodes: KeyNodes})
            # Large negative value for masked positions
            scores = hax.where(key_mask, scores, -1e9)

        # Softmax over keys
        attn_weights = hax.nn.softmax(scores, axis=KeyNodes)

        # Apply attention to values
        output = hax.dot(attn_weights, v_t, axis=KeyNodes)  # (Nodes, Heads, HeadDim)

        # Reshape back to (Nodes, HeadEmbed)
        output = output.flatten_axes((Heads, HeadDim), HeadEmbed)

        # Output projection
        output = self.o_proj(output)

        return output


class ConditionEncoder(eqx.Module):
    """Encoder for conditioning text (signature + docstring).

    Uses character-level embeddings and a small transformer to encode
    the conditioning text. The encoded conditioning is then used for
    cross-attention in the tree transformer.
    """

    token_embed: hnn.Embedding
    position_embed: hnn.Embedding
    layers: list["ConditionTransformerLayer"]
    final_norm: hnn.LayerNorm
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "ConditionEncoder":
        keys = jrandom.split(key, 4)

        token_embed = hnn.Embedding.init(config.CondVocab, config.Embed, key=keys[0])
        position_embed = hnn.Embedding.init(config.CondLen, config.Embed, key=keys[1])

        # Use fewer layers for conditioning encoder (simpler task)
        num_cond_layers = max(1, config.num_layers // 2)
        layer_keys = jrandom.split(keys[2], num_cond_layers)
        layers = [ConditionTransformerLayer.init(config, key=k) for k in layer_keys]

        final_norm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        return ConditionEncoder(token_embed, position_embed, layers, final_norm, config)

    @named_call
    def __call__(
        self,
        condition_tokens: NamedArray,  # (CondLen,) int
        condition_mask: NamedArray | None = None,  # (CondLen,) bool
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Encode conditioning text.

        Args:
            condition_tokens: Character token IDs
            condition_mask: Valid token mask (1 for valid, 0 for padding)

        Returns:
            (CondLen, Embed) encoded conditioning
        """
        # Token embeddings
        x = self.token_embed(condition_tokens)

        # Position embeddings
        positions = hax.arange(self.config.CondLen)
        pos_emb = self.position_embed(positions)
        x = x + pos_emb

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=condition_mask, key=key)

        x = self.final_norm(x)
        return x


class ConditionTransformerLayer(eqx.Module):
    """Transformer layer for conditioning encoder (self-attention only)."""

    attn: "ConditionSelfAttention"
    mlp: "TreeMLP"
    ln1: hnn.LayerNorm
    ln2: hnn.LayerNorm
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "ConditionTransformerLayer":
        k1, k2 = jrandom.split(key, 2)

        attn = ConditionSelfAttention.init(config, key=k1)
        mlp = TreeMLP.init(config, key=k2)
        ln1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        ln2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        return ConditionTransformerLayer(attn, mlp, ln1, ln2, config)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        h = self.ln1(x)
        h = self.attn(h, mask=mask, key=key)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h, key=key)
        x = x + h

        return x


class ConditionSelfAttention(eqx.Module):
    """Self-attention for conditioning sequence."""

    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "ConditionSelfAttention":
        k1, k2, k3, k4 = jrandom.split(key, 4)

        HeadEmbed = Axis("head_embed", config.num_heads * (config.hidden_dim // config.num_heads))

        q_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k1, use_bias=False)
        k_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k2, use_bias=False)
        v_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k3, use_bias=False)
        o_proj = hnn.Linear.init(In=HeadEmbed, Out=config.Embed, key=k4, use_bias=False)

        return ConditionSelfAttention(q_proj, k_proj, v_proj, o_proj, config)

    @named_call
    def __call__(
        self,
        x: NamedArray,  # (CondLen, Embed)
        mask: NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        CondLen = self.config.CondLen
        Heads = self.config.Heads
        HeadDim = self.config.HeadDim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        HeadEmbed = Axis("head_embed", self.config.num_heads * (self.config.hidden_dim // self.config.num_heads))
        q = q.unflatten_axis(HeadEmbed, (Heads, HeadDim))
        k = k.unflatten_axis(HeadEmbed, (Heads, HeadDim))
        v = v.unflatten_axis(HeadEmbed, (Heads, HeadDim))

        KeyCondLen = CondLen.alias("key_cond_len")
        k_t = k.rename({CondLen: KeyCondLen})
        v_t = v.rename({CondLen: KeyCondLen})

        scale = 1.0 / jnp.sqrt(self.config.hidden_dim // self.config.num_heads)
        scores = hax.dot(q, k_t, axis=HeadDim) * scale

        if mask is not None:
            key_mask = mask.rename({CondLen: KeyCondLen})
            scores = hax.where(key_mask, scores, -1e9)

        attn_weights = hax.nn.softmax(scores, axis=KeyCondLen)
        output = hax.dot(attn_weights, v_t, axis=KeyCondLen)

        output = output.flatten_axes((Heads, HeadDim), HeadEmbed)
        output = self.o_proj(output)

        return output


class CrossAttention(eqx.Module):
    """Cross-attention from tree nodes to conditioning sequence.

    Allows tree nodes to attend to the conditioning (signature + docstring)
    to guide generation.
    """

    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "CrossAttention":
        k1, k2, k3, k4 = jrandom.split(key, 4)

        HeadEmbed = Axis("head_embed", config.num_heads * (config.hidden_dim // config.num_heads))

        q_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k1, use_bias=False)
        k_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k2, use_bias=False)
        v_proj = hnn.Linear.init(In=config.Embed, Out=HeadEmbed, key=k3, use_bias=False)
        o_proj = hnn.Linear.init(In=HeadEmbed, Out=config.Embed, key=k4, use_bias=False)

        return CrossAttention(q_proj, k_proj, v_proj, o_proj, config)

    @named_call
    def __call__(
        self,
        x: NamedArray,  # (Nodes, Embed) - queries from tree
        condition: NamedArray,  # (CondLen, Embed) - keys/values from conditioning
        condition_mask: NamedArray | None = None,  # (CondLen,) - valid conditioning mask
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Apply cross-attention from tree to conditioning.

        Args:
            x: Tree node embeddings (queries)
            condition: Conditioning embeddings (keys and values)
            condition_mask: Mask for valid conditioning tokens

        Returns:
            Output embeddings of same shape as x
        """
        Nodes = self.config.Nodes
        CondLen = self.config.CondLen
        Heads = self.config.Heads
        HeadDim = self.config.HeadDim

        # Project queries from tree, keys/values from conditioning
        q = self.q_proj(x)  # (Nodes, HeadEmbed)
        k = self.k_proj(condition)  # (CondLen, HeadEmbed)
        v = self.v_proj(condition)  # (CondLen, HeadEmbed)

        # Reshape to multi-head
        HeadEmbed = Axis("head_embed", self.config.num_heads * (self.config.hidden_dim // self.config.num_heads))
        q = q.unflatten_axis(HeadEmbed, (Heads, HeadDim))  # (Nodes, Heads, HeadDim)
        k = k.unflatten_axis(HeadEmbed, (Heads, HeadDim))  # (CondLen, Heads, HeadDim)
        v = v.unflatten_axis(HeadEmbed, (Heads, HeadDim))  # (CondLen, Heads, HeadDim)

        # Compute attention: (Nodes, Heads, HeadDim) @ (CondLen, Heads, HeadDim) -> (Nodes, Heads, CondLen)
        scale = 1.0 / jnp.sqrt(self.config.hidden_dim // self.config.num_heads)
        scores = hax.dot(q, k, axis=HeadDim) * scale  # (Nodes, Heads, CondLen)

        # Apply conditioning mask
        if condition_mask is not None:
            scores = hax.where(condition_mask, scores, -1e9)

        # Softmax over conditioning positions
        attn_weights = hax.nn.softmax(scores, axis=CondLen)

        # Apply attention to values
        output = hax.dot(attn_weights, v, axis=CondLen)  # (Nodes, Heads, HeadDim)

        # Reshape and project
        output = output.flatten_axes((Heads, HeadDim), HeadEmbed)
        output = self.o_proj(output)

        return output


class TreeMLP(eqx.Module):
    """Feed-forward MLP for tree transformer."""

    up_proj: hnn.Linear
    down_proj: hnn.Linear
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeMLP":
        k1, k2 = jrandom.split(key, 2)

        up_proj = hnn.Linear.init(In=config.Embed, Out=config.Mlp, key=k1, use_bias=True)
        down_proj = hnn.Linear.init(In=config.Mlp, Out=config.Embed, key=k2, use_bias=True)

        return TreeMLP(up_proj, down_proj, config)

    @named_call
    def __call__(self, x: NamedArray, *, key: PRNGKeyArray | None = None) -> NamedArray:
        """Apply MLP with GELU activation."""
        h = self.up_proj(x)
        h = hax.nn.gelu(h)
        output = self.down_proj(h)
        return output


class TreeTransformerLayer(eqx.Module):
    """Single transformer layer for tree encoding with optional cross-attention."""

    attn: TreeAttention
    cross_attn: CrossAttention | None
    mlp: TreeMLP
    ln1: hnn.LayerNorm
    ln2: hnn.LayerNorm
    ln_cross: hnn.LayerNorm | None
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeTransformerLayer":
        if config.use_conditioning:
            k1, k2, k3 = jrandom.split(key, 3)
        else:
            k1, k2 = jrandom.split(key, 2)
            k3 = None

        attn = TreeAttention.init(config, key=k1)
        mlp = TreeMLP.init(config, key=k2)
        ln1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        ln2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        # Cross-attention for conditioning
        if config.use_conditioning:
            cross_attn = CrossAttention.init(config, key=k3)
            ln_cross = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        else:
            cross_attn = None
            ln_cross = None

        return TreeTransformerLayer(attn, cross_attn, mlp, ln1, ln2, ln_cross, config)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: NamedArray | None = None,
        condition: NamedArray | None = None,
        condition_mask: NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Apply transformer layer with pre-norm residual connections.

        Args:
            x: Input embeddings (Nodes, Embed)
            mask: Valid node mask
            condition: Conditioning embeddings (CondLen, Embed) - optional
            condition_mask: Valid conditioning mask - optional
        """
        # Self-attention with residual
        h = self.ln1(x)
        h = self.attn(h, mask=mask, key=key)
        x = x + h

        # Cross-attention with residual (if conditioning is provided)
        if self.cross_attn is not None and condition is not None:
            h = self.ln_cross(x)
            h = self.cross_attn(h, condition, condition_mask=condition_mask, key=key)
            x = x + h

        # MLP with residual
        h = self.ln2(x)
        h = self.mlp(h, key=key)
        x = x + h

        return x


class TreeEncoder(eqx.Module):
    """Encoder for tree structure using stacked transformer layers."""

    embedding: TreeEmbedding
    layers: list[TreeTransformerLayer]
    final_norm: hnn.LayerNorm
    condition_encoder: ConditionEncoder | None
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeEncoder":
        if config.use_conditioning:
            keys = jrandom.split(key, config.num_layers + 2)
            cond_key = keys[-1]
        else:
            keys = jrandom.split(key, config.num_layers + 1)
            cond_key = None

        embedding = TreeEmbedding.init(config, key=keys[0])
        layers = [TreeTransformerLayer.init(config, key=keys[i + 1]) for i in range(config.num_layers)]
        final_norm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        # Conditioning encoder
        if config.use_conditioning:
            condition_encoder = ConditionEncoder.init(config, key=cond_key)
        else:
            condition_encoder = None

        return TreeEncoder(embedding, layers, final_norm, condition_encoder, config)

    @named_call
    def __call__(
        self,
        node_types: NamedArray,
        node_values: NamedArray,
        depth: NamedArray,
        mask: NamedArray | None = None,
        condition_tokens: NamedArray | None = None,
        condition_mask: NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """Encode tree structure with optional conditioning.

        Args:
            node_types: (Nodes,) node type IDs
            node_values: (Nodes, ValueLen) value token IDs
            depth: (Nodes,) depth in tree
            mask: (Nodes,) valid node mask
            condition_tokens: (CondLen,) conditioning token IDs - optional
            condition_mask: (CondLen,) valid conditioning mask - optional

        Returns:
            (Nodes, Embed) encoded node representations
        """
        # Encode conditioning if provided
        condition = None
        if self.condition_encoder is not None and condition_tokens is not None:
            condition = self.condition_encoder(condition_tokens, condition_mask, key=key)

        # Embed inputs
        x = self.embedding(node_types, node_values, depth, key=key)

        # Apply transformer layers
        keys = jrandom.split(key, self.config.num_layers) if key is not None else [None] * self.config.num_layers
        for layer, k in zip(self.layers, keys):
            x = layer(x, mask=mask, condition=condition, condition_mask=condition_mask, key=k)

        # Final normalization
        x = self.final_norm(x)

        return x


class EditPredictor(eqx.Module):
    """Predicts edit location and replacement for tree diffusion.

    Outputs:
    1. Edit location: Which node to modify
    2. Replacement type: What node type to use as replacement
    3. Replacement value: What value tokens for the replacement
    """

    location_head: hnn.Linear
    type_head: hnn.Linear
    value_head: hnn.Linear
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "EditPredictor":
        k1, k2, k3 = jrandom.split(key, 3)

        # Location prediction: (Nodes, Embed) -> (Nodes,) logits
        # We use a simple dot product with a learned vector for location prediction
        LocationOut = Axis("location_out", 1)
        location_head = hnn.Linear.init(In=config.Embed, Out=LocationOut, key=k1, use_bias=True)

        # Type prediction: (Embed,) -> (NodeVocab,) logits
        type_head = hnn.Linear.init(In=config.Embed, Out=config.NodeVocab, key=k2, use_bias=True)

        # Value prediction: (Embed,) -> (ValueLen, ValueVocab) logits
        ValueOut = Axis("value_out", config.max_value_len * config.value_vocab_size)
        value_head = hnn.Linear.init(In=config.Embed, Out=ValueOut, key=k3, use_bias=True)

        return EditPredictor(location_head, type_head, value_head, config)

    @named_call
    def __call__(
        self,
        encoded: NamedArray,  # (Nodes, Embed)
        mask: NamedArray | None = None,  # (Nodes,) valid mask
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Predict edit location and replacement.

        Args:
            encoded: Encoded node representations
            mask: Valid node mask

        Returns:
            Tuple of:
            - location_logits: (Nodes,) logits for edit location
            - type_logits: (NodeVocab,) logits for replacement type
            - value_logits: (ValueLen, ValueVocab) logits for replacement value
        """
        Nodes = self.config.Nodes
        LocationOut = Axis("location_out", 1)

        # Location logits: project each node to a scalar
        location_logits_2d = self.location_head(encoded)  # (Nodes, LocationOut)
        # Flatten to (Nodes,) by summing over the size-1 dimension
        location_logits = hax.sum(location_logits_2d, axis=LocationOut)  # (Nodes,)

        # Mask invalid locations
        if mask is not None:
            location_logits = hax.where(mask, location_logits, -1e9)

        # Pool encoded representations for type/value prediction
        # Use attention-weighted pooling based on location logits
        location_weights = hax.nn.softmax(location_logits, axis=Nodes)
        pooled = hax.dot(location_weights, encoded, axis=Nodes)  # (Embed,)

        # Type logits
        type_logits = self.type_head(pooled)  # (NodeVocab,)

        # Value logits
        ValueOut = Axis("value_out", self.config.max_value_len * self.config.value_vocab_size)
        value_logits_flat = self.value_head(pooled)  # (ValueOut,)
        value_logits = value_logits_flat.unflatten_axis(
            ValueOut, (self.config.ValueLen, self.config.ValueVocab)
        )  # (ValueLen, ValueVocab)

        return location_logits, type_logits, value_logits


class TreeDiffusionModel(eqx.Module):
    """Full tree diffusion model for program synthesis.

    Combines the tree encoder and edit predictor to predict
    the reverse diffusion step (denoising).

    Supports optional conditioning on function signature + docstring
    for guided generation.
    """

    encoder: TreeEncoder
    predictor: EditPredictor
    config: TreeDiffusionConfig = eqx.field(static=True)

    @staticmethod
    def init(config: TreeDiffusionConfig, *, key: PRNGKeyArray) -> "TreeDiffusionModel":
        k1, k2 = jrandom.split(key, 2)

        encoder = TreeEncoder.init(config, key=k1)
        predictor = EditPredictor.init(config, key=k2)

        return TreeDiffusionModel(encoder, predictor, config)

    @named_call
    def __call__(
        self,
        node_types: NamedArray,
        node_values: NamedArray,
        depth: NamedArray,
        mask: NamedArray | None = None,
        condition_tokens: NamedArray | None = None,
        condition_mask: NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Forward pass: encode tree and predict edit.

        Args:
            node_types: (Nodes,) node type IDs
            node_values: (Nodes, ValueLen) value token IDs
            depth: (Nodes,) depth in tree
            mask: (Nodes,) valid node mask
            condition_tokens: (CondLen,) conditioning token IDs - optional
            condition_mask: (CondLen,) valid conditioning mask - optional
            key: PRNG key for dropout

        Returns:
            Tuple of (location_logits, type_logits, value_logits)
        """
        # Encode the tree with optional conditioning
        encoded = self.encoder(
            node_types,
            node_values,
            depth,
            mask=mask,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            key=key,
        )

        # Predict edit
        location_logits, type_logits, value_logits = self.predictor(encoded, mask=mask)

        return location_logits, type_logits, value_logits


def compute_loss(
    model: TreeDiffusionModel,
    batch: dict[str, NamedArray],
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute training loss for tree diffusion.

    Args:
        model: The tree diffusion model
        batch: Dictionary containing:
            - corrupted_node_types: (Batch, Nodes)
            - corrupted_node_values: (Batch, Nodes, ValueLen)
            - corrupted_depth: (Batch, Nodes)
            - corrupted_node_mask: (Batch, Nodes)
            - edit_location: (Batch,) target location indices
            - replacement_type: (Batch,) target type IDs
            - replacement_value: (Batch, ValueLen) target value IDs
        key: PRNG key

    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    config = model.config

    # Forward pass
    location_logits, type_logits, value_logits = model(
        batch["corrupted_node_types"],
        batch["corrupted_node_values"],
        batch["corrupted_depth"],
        mask=batch["corrupted_node_mask"],
        key=key,
    )

    # Location loss (cross-entropy)
    location_targets = batch["edit_location"]
    location_loss = hax.nn.cross_entropy_loss(
        location_logits,
        config.Nodes,
        location_targets,
    )

    # Type loss (cross-entropy)
    type_targets = batch["replacement_type"]
    type_loss = hax.nn.cross_entropy_loss(
        type_logits,
        config.NodeVocab,
        type_targets,
    )

    # Value loss (cross-entropy per position)
    value_targets = batch["replacement_value"]  # (Batch, ValueLen)
    value_loss = hax.nn.cross_entropy_loss(
        value_logits,
        config.ValueVocab,
        value_targets,
    )
    value_loss = hax.mean(value_loss, axis=config.ValueLen)

    # Total loss
    total_loss = location_loss + type_loss + value_loss

    metrics = {
        "location_loss": location_loss,
        "type_loss": type_loss,
        "value_loss": value_loss,
        "total_loss": total_loss,
    }

    return total_loss, metrics
