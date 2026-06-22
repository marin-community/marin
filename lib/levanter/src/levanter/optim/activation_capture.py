# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer input-activation Gram capture for activation-aware Muon (idea 3).

The activation-aware optimizer (:mod:`levanter.optim.activation_aware`) whitens the
update by the layer's *input* second moment ``Σ = A Aᵀ``, which is not recoverable from
the gradient and so must be read off the forward pass. This module computes, in a single
forward over the transformer blocks (reusing the model's own submodules via
``Stacked.scan_via``), the two block-level input Grams per layer:

* ``attn_in`` — ``input_layernorm(residual)``, the input to attention q/k/v;
* ``mlp_in``  — ``post_attention_layernorm(residual)``, the input to MLP gate/up.

Each Gram is summed over the (batch, position) tokens, giving a ``[Layers, Embed, Embed2]``
NamedArray that aligns with the Stacked transformer weights ``[Layers, Out, Embed]``.
"""

from typing import Optional

import haliax as hax
from haliax import NamedArray

from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaLMHeadModel


def _token_gram(act: NamedArray, embed: hax.Axis) -> NamedArray:
    """AAᵀ for one activation tensor: contract over all non-Embed (token) axes → (Embed, Embed2)."""
    embed2 = embed.alias(embed.name + "_in2")
    token_axes = tuple(ax for ax in act.axes if ax.name != embed.name)
    return hax.dot(act, act.rename({embed: embed2}), axis=token_axes)


def _capture_block(block, x: NamedArray, *, mask, pos_ids, embed: hax.Axis):
    """Mirror LlamaDecoderLayer.__call__, returning (block_output, {attn_in_gram, mlp_in_gram})."""
    attn_in = block.input_layernorm(x)
    attn_output = block.self_attn(x=attn_in, mask=mask, key=None, pos_ids=pos_ids)
    if block.post_attn_layernorm is not None:
        attn_output = block.post_attn_layernorm(attn_output)
    h = x + attn_output

    mlp_in = block.post_attention_layernorm(h)
    mlp_output = block.mlp(mlp_in, key=None)
    if block.post_mlp_layernorm is not None:
        mlp_output = block.post_mlp_layernorm(mlp_output)
    out = h + mlp_output

    grams = {"attn_in": _token_gram(attn_in, embed), "mlp_in": _token_gram(mlp_in, embed)}
    return out, grams


def compute_input_grams(
    model: LlamaLMHeadModel,
    tokens: NamedArray,
    attn_mask: Optional[AttentionMask | NamedArray] = None,
    *,
    pos_ids: Optional[NamedArray] = None,
) -> dict[str, NamedArray]:
    """Per-layer block input Grams Σ=AAᵀ for the activation-aware optimizer.

    Returns ``{"attn_in": NamedArray[Layers, Embed, Embed2], "mlp_in": NamedArray[Layers, Embed, Embed2]}``,
    each summed over the (batch, position) tokens. Runs a single forward over the blocks; no
    backward. The leading ``Layers`` axis matches the Stacked transformer weights.
    """
    embed = model.config.Embed
    x = model.embeddings.embed(tokens)
    _, grams = model.transformer.layers.scan_via(lambda b, c, **kw: _capture_block(b, c, embed=embed, **kw))(
        x, mask=attn_mask, pos_ids=pos_ids  # pyrefly: ignore[unexpected-keyword]
    )
    return grams
