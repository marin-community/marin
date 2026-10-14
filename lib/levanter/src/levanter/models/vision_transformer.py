# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer blocks shared by the SigLIP vision encoders."""

from typing import Callable, Optional, Protocol

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import numpy as np
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.scan import Stacked
from jaxtyping import PRNGKeyArray

from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.utils.activation import ActivationFunctionEnum


SPLASH_BLOCK_SIZE = 128


class VisionBlockConfig(Protocol):
    @property
    def Embed(self) -> Axis: ...

    @property
    def Mlp(self) -> Axis: ...

    @property
    def Heads(self) -> Axis: ...

    @property
    def HeadSize(self) -> Axis: ...

    @property
    def hidden_act(self) -> ActivationFunctionEnum: ...

    @property
    def layer_norm_eps(self) -> float: ...

    @property
    def inference(self) -> bool: ...

    @property
    def use_flash_attention(self) -> bool: ...

    @property
    def attn_backend(self) -> Optional[AttentionBackend]: ...

    @property
    def flash_attention_block_size(self) -> Optional[int]: ...

    @property
    def attention_dropout(self) -> float: ...


class VisionMLP(eqx.Module):
    """Two-layer feedforward block with the configured activation."""

    fc1: hnn.Linear  # projection from Embed to Mlp (intermediate)
    fc2: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn: ActivationFunctionEnum, *, key: PRNGKeyArray) -> "VisionMLP":
        k_fc1, k_fc2 = maybe_rng_split(key, 2)

        fc1 = hnn.Linear.init(In=Embed, Out=Mlp, key=k_fc1, use_bias=True, out_first=True)
        fc2 = hnn.Linear.init(In=Mlp, Out=Embed, key=k_fc2, use_bias=True, out_first=True)

        return VisionMLP(fc1, fc2, activation_fn.to_fn())

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionAttention(eqx.Module):
    """Self-attention with mask-preserving padding for Splash kernels."""

    config: VisionBlockConfig = eqx.field(static=True)
    q_proj: hnn.Linear  # Query projection from Embed to (Heads, HeadSize)
    k_proj: hnn.Linear  # Key projection from Embed to (Heads, HeadSize)
    v_proj: hnn.Linear  # Value projection from Embed to (Heads, HeadSize)
    out_proj: hnn.Linear  # Output projection from (Heads, HeadSize) to Embed
    inference: bool  # Whether to run in inference mode (disables dropout)

    @staticmethod
    def init(config: VisionBlockConfig, *, key: PRNGKeyArray) -> "VisionAttention":
        k_q, k_k, k_v, k_out = maybe_rng_split(key, 4)

        Embed = config.Embed
        Heads = config.Heads
        HeadSize = config.HeadSize

        q_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_q, use_bias=True, out_first=True)
        k_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_k, use_bias=True, out_first=True)
        v_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_v, use_bias=True, out_first=True)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=True, out_first=True)

        return VisionAttention(config, q_proj, k_proj, v_proj, out_proj, config.inference)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        k_drop = maybe_rng_split(key, 1)[0] if key is not None else None

        seq_axis_name = None
        embed_axis = self.config.Embed
        for ax in x.axes:
            if ax.name in ("num_patches", "position"):
                seq_axis_name = ax.name
                break
        if seq_axis_name is None:
            for ax in x.axes:
                if ax != embed_axis and ax.name not in ("batch", "Batch"):
                    seq_axis_name = ax.name
                    break
        if seq_axis_name is None:
            raise ValueError(f"Could not find sequence axis in input {x.axes}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute padding for Splash Attention (requires sequence length to be multiple of 128)
        seq_axis = q.resolve_axis(seq_axis_name)
        orig_seq_len = seq_axis.size
        pad_len = (SPLASH_BLOCK_SIZE - (orig_seq_len % SPLASH_BLOCK_SIZE)) % SPLASH_BLOCK_SIZE

        if pad_len > 0:
            q = hax.pad(q, {seq_axis_name: (0, pad_len)})
            k = hax.pad(k, {seq_axis_name: (0, pad_len)})
            v = hax.pad(v, {seq_axis_name: (0, pad_len)})

            padded_seq_len = orig_seq_len + pad_len
            PaddedSeqAxis = Axis(seq_axis_name, padded_seq_len)
            PaddedKeyAxis = Axis("key_position", padded_seq_len)

            # This allows Splash Attention to use NumpyMask which requires static arrays

            q_indices = np.arange(padded_seq_len)
            k_indices = np.arange(padded_seq_len)
            pad_mask_np = (q_indices[:, None] < orig_seq_len) & (k_indices[None, :] < orig_seq_len)
            pad_mask = hax.named(pad_mask_np, (PaddedSeqAxis, PaddedKeyAxis))

            if mask is not None:
                original_mask = mask.materialize(seq_axis, Axis("key_position", orig_seq_len))
                if original_mask is not None:
                    original_mask = original_mask.broadcast_axis((seq_axis, Axis("key_position", orig_seq_len)))
                    original_mask = hax.pad(
                        original_mask,
                        {seq_axis_name: (0, pad_len), "key_position": (0, pad_len)},
                        constant_values=False,
                    )
                    pad_mask = pad_mask & original_mask
            combined_mask = AttentionMask.explicit(pad_mask)
        else:
            combined_mask = mask
        k = k.rename({seq_axis_name: "key_position"})
        v = v.rename({seq_axis_name: "key_position"})

        attn_output = dot_product_attention(
            seq_axis_name,
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=combined_mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            dropout=self.config.attention_dropout,
            prng=k_drop,
            attention_dtype=x.dtype,
        )

        if pad_len > 0:
            attn_output = attn_output[seq_axis_name, :orig_seq_len]

        return self.out_proj(attn_output.astype(x.dtype))


class VisionEncoderLayer(eqx.Module):
    """Pre-normalized attention and feedforward residual blocks."""

    config: VisionBlockConfig = eqx.field(static=True)
    layer_norm1: hnn.LayerNorm  # Pre-attention layer norm
    self_attn: VisionAttention  # Self-attention module
    layer_norm2: hnn.LayerNorm  # Pre-MLP layer norm
    mlp: VisionMLP  # MLP module

    @staticmethod
    def init(config: VisionBlockConfig, *, key: PRNGKeyArray) -> "VisionEncoderLayer":
        k_attn, k_mlp = maybe_rng_split(key, 2)

        layer_norm1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)
        layer_norm2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        self_attn = VisionAttention.init(config, key=k_attn)
        mlp = VisionMLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp)

        return VisionEncoderLayer(config, layer_norm1, self_attn, layer_norm2, mlp)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        k_attn, _ = maybe_rng_split(key, 2)

        residual = x
        x_norm = self.layer_norm1(x)
        attn_output = self.self_attn(x_norm, mask=mask, key=k_attn)
        x = residual + attn_output

        residual = x
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output

        return x


def vision_activation(name: str) -> ActivationFunctionEnum:
    aliases = {"gelu_pytorch_tanh": "gelu_new", "swish": "silu"}
    return ActivationFunctionEnum(aliases.get(name, name))


def vision_hidden_states(
    layers: Stacked[VisionEncoderLayer],
    hidden_states: NamedArray,
    mask: Optional[AttentionMask],
    output_hidden_states: bool,
    *,
    key: Optional[PRNGKeyArray],
) -> tuple[NamedArray, tuple[NamedArray, ...]]:
    """Return final encoder features and requested pre-normalization layer states."""
    keys = maybe_rng_split(key, layers.Block.size) if key is not None else None
    if not output_hidden_states:
        final = layers.fold(hidden_states, mask, key=keys)
        return final, (final,)

    def apply_layer(
        block: VisionEncoderLayer,
        carry: NamedArray,
        mask: Optional[AttentionMask],
        key: Optional[PRNGKeyArray],
    ) -> tuple[NamedArray, NamedArray]:
        output = block(carry, mask, key=key)
        return output, output

    final, stacked = layers.scan_via(apply_layer)(hidden_states, mask, keys)
    return final, (hidden_states, *hax.unbind(stacked, layers.Block))
