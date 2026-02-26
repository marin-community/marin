# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
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

"""Autoregressive edit-prediction model for tree diffusion.

A causal transformer that predicts single program edits: given the current
program as context, it autoregressively generates a position token (WHERE to
edit) followed by replacement tokens (WHAT to insert).

Key differences from the old D3PM model:
1. Causal attention (standard LLM) instead of bidirectional
2. No timestep embedding (tree diffusion doesn't use a fixed schedule)
3. Predicts one edit at a time, not all tokens simultaneously

The transformer backbone (blocks, attention, MLP, norms, RoPE) is identical
to Grug and can be initialized directly from pretrained LLM weights.
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import (
    apply_rotary_embedding as grug_apply_rotary,
    attention as grug_attention,
)

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import (
    TreeDiffusionAttentionParams,
    TreeDiffusionBlockParams,
    _init_weight,
    mlp,
    rms_norm,
)

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class EditModelParams:
    """Parameters for the AR edit-prediction model.

    Contains token embeddings, output projection, transformer blocks, and
    final layer norm. No timestep conditioning -- tree diffusion uses
    iterative edits rather than a fixed noise schedule.
    """

    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[TreeDiffusionBlockParams, ...]
    final_norm: jax.Array


def init_edit_params(cfg: TreeDiffusionConfig, *, key: PRNGKeyArray) -> EditModelParams:
    """Initialize edit model parameters.

    Args:
        cfg: Model configuration.
        key: PRNG key.

    Returns:
        Initialized EditModelParams.
    """
    head_dim = cfg.inferred_head_dim
    key, embed_key, out_key = random.split(key, 3)
    layer_keys = random.split(key, cfg.num_layers)

    token_embed = _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std)
    output_proj = _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std)
    final_norm = jnp.ones((cfg.hidden_dim,), dtype=jnp.float32)

    blocks: list[TreeDiffusionBlockParams] = []
    D, N, M, H, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim

    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = random.split(layer_keys[i], 7)

        attn = TreeDiffusionAttentionParams(
            w_q=_init_weight(k_q, (D, N * H), cfg.initializer_std),
            w_k=_init_weight(k_k, (D, M * H), cfg.initializer_std),
            w_v=_init_weight(k_v, (D, M * H), cfg.initializer_std),
            w_o=_init_weight(k_o, (N * H, D), cfg.initializer_std),
        )

        blocks.append(
            TreeDiffusionBlockParams(
                attn=attn,
                rms_attn=jnp.ones((D,), dtype=jnp.float32),
                rms_mlp=jnp.ones((D,), dtype=jnp.float32),
                mlp_gate=_init_weight(k_gate, (D, I), cfg.initializer_std),
                mlp_up=_init_weight(k_up, (D, I), cfg.initializer_std),
                mlp_down=_init_weight(k_down, (I, D), cfg.initializer_std),
            )
        )

    return EditModelParams(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def _make_causal_mask(seq_len: int) -> Float[Array, "S S"]:
    """Create a causal attention mask.

    Returns a (seq_len, seq_len) boolean mask where True means "allowed to
    attend". Position i can attend to positions 0..i (inclusive).
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def forward(
    params: EditModelParams,
    token_ids: Int[Array, "B S"],
    cfg: TreeDiffusionConfig,
) -> Float[Array, "B S V"]:
    """Causal AR forward pass for edit prediction.

    Args:
        params: Model parameters.
        token_ids: Input token IDs. The sequence is
            [context..., SOS, POS, replacement..., EOS, PAD...].
        cfg: Model configuration.

    Returns:
        Logits of shape (batch, seq, vocab). For training, shift by 1
        to predict the next token at each position.
    """
    compute_dtype = jnp.dtype(cfg.compute_dtype)
    head_dim = cfg.inferred_head_dim
    _batch_size, seq_len = token_ids.shape

    hidden = params.token_embed[token_ids].astype(compute_dtype)

    # Causal mask: position i can attend to 0..i. Combined with padding.
    causal = _make_causal_mask(seq_len)
    not_pad = token_ids != cfg.pad_token_id
    # (B, S, S): causal AND both positions are not padding.
    attn_mask = causal[None, :, :] & not_pad[:, None, :] & not_pad[:, :, None]

    def _block_fn(hidden, block):
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)

        q = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q),
            "b s (n d) -> b s n d",
            d=head_dim,
        )
        k = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k),
            "b s (m d) -> b s m d",
            d=head_dim,
        )
        v = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v),
            "b s (m d) -> b s m d",
            d=head_dim,
        )

        q, k = grug_apply_rotary(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)

        # Causal attention with padding mask.
        attn_out = grug_attention(q, k, v, mask=attn_mask)
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o)

        hidden = hidden + attn_out

        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = mlp(block, mlp_in)
        hidden = hidden + mlp_out
        return hidden

    block_fn = jax.checkpoint(_block_fn) if cfg.gradient_checkpointing else _block_fn

    for block in params.blocks:
        hidden = block_fn(hidden, block)

    # Project to vocab logits in float32 for numerical stability.
    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    logits = jnp.einsum("bsh,hv->bsv", hidden.astype(jnp.float32), params.output_proj)
    return logits


def ar_loss(
    params: EditModelParams,
    token_ids: Int[Array, "B S"],
    loss_mask: Float[Array, "B S"],
    cfg: TreeDiffusionConfig,
) -> tuple[Float[Array, ""], dict]:
    """Compute AR cross-entropy loss on edit predictions.

    Standard next-token prediction loss, but only on the edit portion
    of the sequence (position token + replacement tokens + EOS),
    controlled by loss_mask.

    Args:
        params: Model parameters.
        token_ids: Full sequence [context, SOS, POS, replacement, EOS, PAD...].
        loss_mask: Float mask, 1.0 for tokens that contribute to loss
            (POS, replacement, EOS), 0.0 for context and padding.
        cfg: Model configuration.

    Returns:
        Tuple of (scalar_loss, metrics_dict).
    """
    logits = forward(params, token_ids, cfg)

    # Shift: predict token at position i+1 from logits at position i.
    shifted_logits = logits[:, :-1, :]
    shifted_targets = token_ids[:, 1:]
    shifted_mask = loss_mask[:, 1:]

    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, shifted_targets[..., None], axis=-1).squeeze(-1)

    masked_loss = -target_log_probs * shifted_mask
    num_loss_tokens = jnp.sum(shifted_mask)
    loss = jnp.sum(masked_loss) / jnp.maximum(num_loss_tokens, 1.0)

    # Metrics.
    predictions = jnp.argmax(shifted_logits, axis=-1)
    correct = (predictions == shifted_targets).astype(jnp.float32) * shifted_mask
    accuracy = jnp.sum(correct) / jnp.maximum(num_loss_tokens, 1.0)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "num_loss_tokens": num_loss_tokens,
    }

    return loss, metrics
