# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental upstream FlashAttention-4 THD/varlen wrapper for Grug attention."""

import importlib
import math
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from levanter.grug.attention._core import AttentionMask


@lru_cache(maxsize=1)
def _flash_attn_varlen_func() -> Any:
    try:
        module = importlib.import_module("flash_attn.cute")
        return module.flash_attn_varlen_func
    except (ImportError, AttributeError):
        try:
            module = importlib.import_module("flash_attn.cute.interface")
            return module.flash_attn_varlen_func
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "gpu_fa4_thd_attention requires flash-attn-4 with the CuTe varlen API "
                "`flash_attn.cute.flash_attn_varlen_func`."
            ) from exc


@lru_cache(maxsize=1)
def _flash_attn_internal_funcs() -> tuple[Any, Any]:
    try:
        module = importlib.import_module("flash_attn.cute")
        return module._flash_attn_fwd, module._flash_attn_bwd
    except (ImportError, AttributeError):
        try:
            module = importlib.import_module("flash_attn.cute.interface")
            return module._flash_attn_fwd, module._flash_attn_bwd
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "gpu_fa4_thd_attention requires flash-attn-4 with the CuTe internal APIs "
                "`flash_attn.cute.interface._flash_attn_fwd/_flash_attn_bwd`."
            ) from exc


def _is_torch_tensor(x: Any) -> bool:
    module = type(x).__module__
    return module == "torch" or module.startswith("torch.")


def _validate_simple_causal_self_attention(
    q: Any,
    k: Any,
    v: Any,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> None:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{backend_name} does not support dense masks.")
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError(f"{backend_name} requires an AttentionMask with packed segment_ids.")
    if not mask.is_causal:
        raise NotImplementedError(f"{backend_name} supports only causal self-attention.")
    if mask.sliding_window is not None:
        raise NotImplementedError(f"{backend_name} does not support sliding-window attention.")

    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError(
            f"{backend_name} expects q/k/v with shape [B,S,H,D], got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{backend_name} requires matching batch sizes, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise NotImplementedError(f"{backend_name} supports only self-attention with q_len == kv_len.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(f"{backend_name} requires Dq == Dk == Dv, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{backend_name} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"{backend_name} requires matching K/V heads, got k={k.shape}, v={v.shape}")

    if mask.segment_ids is None and mask.thd_segment_metadata is None:
        raise NotImplementedError(f"{backend_name} requires packed segment_ids or precomputed THD segment metadata.")


def _torch_batched_segment_ids(segment_ids: Any, *, batch_size: int, seq_len: int) -> Any:

    if not _is_torch_tensor(segment_ids):
        raise TypeError("segment_ids must be a torch tensor.")
    if segment_ids.ndim == 1:
        if segment_ids.shape[0] != seq_len:
            raise ValueError(f"1D segment_ids must match sequence length {seq_len}, got {segment_ids.shape}")
        return segment_ids.unsqueeze(0).expand(batch_size, seq_len)
    if segment_ids.ndim == 2:
        if segment_ids.shape[0] not in (1, batch_size) or segment_ids.shape[1] != seq_len:
            raise ValueError(f"2D segment_ids must have shape [1|{batch_size}, {seq_len}], got {segment_ids.shape}")
        if segment_ids.shape[0] == 1 and batch_size != 1:
            return segment_ids.expand(batch_size, seq_len)
        return segment_ids
    raise ValueError(f"segment_ids must be 1D or 2D, got ndim={segment_ids.ndim}")


def _torch_thd_reshape_metadata(q_segment_ids: Any, kv_segment_ids: Any, *, batch_size: int, seq_len: int) -> Any:
    import torch  # noqa: PLC0415

    q_segment_ids = _torch_batched_segment_ids(q_segment_ids, batch_size=batch_size, seq_len=seq_len)
    kv_segment_ids = _torch_batched_segment_ids(kv_segment_ids, batch_size=batch_size, seq_len=seq_len)
    if torch.any(q_segment_ids != kv_segment_ids):
        raise NotImplementedError("torch_fa4_thd_attention requires matching q/kv segment_ids.")

    if torch.any(q_segment_ids < 0):
        raise NotImplementedError("torch_fa4_thd_attention simple THD reshape does not support padded/invalid tokens.")
    previous = torch.cat([q_segment_ids[:, :1], q_segment_ids[:, :-1]], dim=1)
    first = torch.zeros_like(q_segment_ids, dtype=torch.bool)
    first[:, 0] = True
    starts = first | (q_segment_ids != previous)
    if torch.any(q_segment_ids[:, 1:] < q_segment_ids[:, :-1]):
        raise NotImplementedError(
            "segment_ids must be contiguous and monotonically nondecreasing within each batch row for THD reshape."
        )
    starts_flat = starts.reshape(-1)
    start_positions = torch.nonzero(starts_flat, as_tuple=False).flatten().to(torch.int32)
    total_tokens = batch_size * seq_len
    if start_positions.numel() == 0 or int(start_positions[0].item()) != 0:
        raise ValueError("segment_ids must contain at least one valid contiguous segment.")
    cu_seqlens = torch.cat(
        [
            start_positions,
            torch.tensor([total_tokens], device=q_segment_ids.device, dtype=torch.int32),
        ],
        dim=0,
    )
    lengths = torch.diff(cu_seqlens)
    if torch.any(lengths <= 0):
        raise ValueError("segment_ids must form non-empty contiguous packed segments.")
    return cu_seqlens, int(lengths.max().item())


def _torch_thd_views(q: Any, k: Any, v: Any, segment_ids: tuple[Any, Any]) -> tuple[Any, Any, Any, Any, int]:
    batch_size, seq_len, _, head_dim = q.shape
    cu_seqlens, max_seqlen = _torch_thd_reshape_metadata(
        segment_ids[0],
        segment_ids[1],
        batch_size=batch_size,
        seq_len=seq_len,
    )
    return (
        q.reshape(batch_size * seq_len, q.shape[2], head_dim),
        k.reshape(batch_size * seq_len, k.shape[2], head_dim),
        v.reshape(batch_size * seq_len, v.shape[2], head_dim),
        cu_seqlens,
        max_seqlen,
    )


def _torch_cu_seqlens_from_segment_lengths(segment_lengths: Any, num_segments: Any, *, total_tokens: int) -> Any:
    import torch  # noqa: PLC0415

    if not _is_torch_tensor(segment_lengths):
        raise TypeError("segment_lengths must be a torch tensor.")
    if not _is_torch_tensor(num_segments):
        raise TypeError("num_segments must be a torch tensor.")
    if segment_lengths.ndim == 1:
        segment_lengths = segment_lengths.unsqueeze(0)
    if segment_lengths.ndim != 2:
        raise ValueError(f"segment_lengths must have shape [B,M] or [M], got {segment_lengths.shape}")

    num_segments = num_segments.reshape(-1).to(torch.int64)
    if num_segments.shape[0] != segment_lengths.shape[0]:
        raise ValueError(
            f"num_segments must have one entry per batch row, got {num_segments.shape} and {segment_lengths.shape}"
        )

    max_segments = segment_lengths.shape[1]
    if total_tokens % segment_lengths.shape[0] != 0:
        raise ValueError(f"total_tokens={total_tokens} is not divisible by batch rows={segment_lengths.shape[0]}.")
    tokens_per_row = total_tokens // segment_lengths.shape[0]
    segment_index = torch.arange(max_segments, device=segment_lengths.device, dtype=torch.int64)
    padding_row = num_segments == 0
    segment_lengths = torch.where(
        padding_row[:, None] & (segment_index[None, :] == 0),
        torch.full_like(segment_lengths, tokens_per_row),
        segment_lengths,
    )
    num_segments = torch.where(padding_row, torch.ones_like(num_segments), num_segments)

    keep = segment_index.unsqueeze(0) < num_segments[:, None]
    lengths = segment_lengths.to(torch.int32)[keep]
    if lengths.numel() == 0:
        raise ValueError("THD segment metadata must contain at least one segment.")
    if torch.any(lengths <= 0):
        raise ValueError("THD segment metadata contains a non-positive active segment length.")

    cu_seqlens = torch.cat(
        [
            torch.zeros((1,), device=segment_lengths.device, dtype=torch.int32),
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError(
            f"THD segment metadata covers {int(cu_seqlens[-1].item())} tokens, but q/k/v contain {total_tokens}."
        )
    return cu_seqlens, int(lengths.max().item())


def _torch_thd_views_from_segment_lengths(
    q: Any,
    k: Any,
    v: Any,
    segment_lengths: Any,
    num_segments: Any,
) -> tuple[Any, Any, Any, Any, int]:
    batch_size, seq_len, _, head_dim = q.shape
    cu_seqlens, max_seqlen = _torch_cu_seqlens_from_segment_lengths(
        segment_lengths,
        num_segments,
        total_tokens=batch_size * seq_len,
    )
    return (
        q.reshape(batch_size * seq_len, q.shape[2], head_dim),
        k.reshape(batch_size * seq_len, k.shape[2], head_dim),
        v.reshape(batch_size * seq_len, v.shape[2], head_dim),
        cu_seqlens,
        max_seqlen,
    )


def torch_fa4_thd_attention_internal(
    q: Any,
    k: Any,
    v: Any,
    segment_ids: tuple[Any, Any],
    *,
    causal: bool = True,
) -> Any:
    """Run upstream FA4 CuTe varlen forward using the internal fwd API."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v)):
        raise TypeError("torch_fa4_thd_attention_internal expects torch tensors.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_attention_internal supports only causal=True.")
    mask = AttentionMask.causal().with_segment_ids(*segment_ids)
    _validate_simple_causal_self_attention(q, k, v, mask, backend_name="torch_fa4_thd_attention_internal")

    q = q.detach()
    k = k.detach()
    v = v.detach()
    batch_size, seq_len, _, head_dim = q.shape
    q_thd, k_thd, v_thd, cu_seqlens, max_seqlen = _torch_thd_views(q, k, v, segment_ids)
    flash_attn_fwd, _ = _flash_attn_internal_funcs()
    out, _ = flash_attn_fwd(
        q_thd,
        k_thd,
        v_thd,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_lse=True,
    )
    return out.reshape(batch_size, seq_len, q.shape[2], head_dim)


def torch_fa4_thd_attention_internal_from_lengths(
    q: Any,
    k: Any,
    v: Any,
    segment_lengths: Any,
    num_segments: Any,
    *,
    causal: bool = True,
) -> Any:
    """Run upstream FA4 CuTe varlen forward from fixed-shape THD segment metadata."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v)):
        raise TypeError("torch_fa4_thd_attention_internal_from_lengths expects torch tensors.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_attention_internal_from_lengths supports only causal=True.")

    q = q.detach()
    k = k.detach()
    v = v.detach()
    batch_size, seq_len, _, head_dim = q.shape
    q_thd, k_thd, v_thd, cu_seqlens, max_seqlen = _torch_thd_views_from_segment_lengths(
        q,
        k,
        v,
        segment_lengths,
        num_segments,
    )
    flash_attn_fwd, _ = _flash_attn_internal_funcs()
    out, _ = flash_attn_fwd(
        q_thd,
        k_thd,
        v_thd,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_lse=True,
    )
    return out.reshape(batch_size, seq_len, q.shape[2], head_dim)


def torch_fa4_thd_attention_internal_fwd_bwd(
    q: Any,
    k: Any,
    v: Any,
    dout: Any,
    segment_ids: tuple[Any, Any],
    *,
    causal: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Run upstream FA4 CuTe varlen fwd+bwd using internal fwd/bwd APIs."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v) and _is_torch_tensor(dout)):
        raise TypeError("torch_fa4_thd_attention_internal_fwd_bwd expects torch tensors.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_attention_internal_fwd_bwd supports only causal=True.")
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4 or len(dout.shape) != 4:
        raise ValueError(
            "torch_fa4_thd_attention_internal_fwd_bwd expects q/k/v/dout with shape [B,S,H,D], "
            f"got q={q.shape}, k={k.shape}, v={v.shape}, dout={dout.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or q.shape[0] != dout.shape[0]:
        raise ValueError("torch_fa4_thd_attention_internal_fwd_bwd requires matching batch sizes.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1] or q.shape[1] != dout.shape[1]:
        raise NotImplementedError("torch_fa4_thd_attention_internal_fwd_bwd supports only self-attention.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1] or q.shape[-1] != dout.shape[-1]:
        raise ValueError("torch_fa4_thd_attention_internal_fwd_bwd requires matching head dimensions.")
    if q.shape[2] != dout.shape[2] or q.shape[2] % k.shape[2] != 0 or k.shape[2] != v.shape[2]:
        raise ValueError(
            "torch_fa4_thd_attention_internal_fwd_bwd requires Hq divisible by Hkv and matching dout/K/V heads."
        )

    q = q.detach()
    k = k.detach()
    v = v.detach()
    dout = dout.detach()
    batch_size, seq_len, _, head_dim = q.shape
    q_thd, k_thd, v_thd, cu_seqlens, max_seqlen = _torch_thd_views(q, k, v, segment_ids)
    dout_thd = dout.reshape(batch_size * seq_len, q.shape[2], head_dim)
    flash_attn_fwd, flash_attn_bwd = _flash_attn_internal_funcs()
    out, lse = flash_attn_fwd(
        q_thd,
        k_thd,
        v_thd,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_lse=True,
    )
    dq, dk, dv = flash_attn_bwd(
        q_thd,
        k_thd,
        v_thd,
        out,
        dout_thd,
        lse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    return (
        out.reshape(batch_size, seq_len, q.shape[2], head_dim),
        dq.reshape(q.shape),
        dk.reshape(k.shape),
        dv.reshape(v.shape),
    )


def torch_fa4_thd_attention_internal_fwd_bwd_from_lengths(
    q: Any,
    k: Any,
    v: Any,
    dout: Any,
    segment_lengths: Any,
    num_segments: Any,
    *,
    causal: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Run upstream FA4 CuTe varlen fwd+bwd from fixed-shape THD segment metadata."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v) and _is_torch_tensor(dout)):
        raise TypeError("torch_fa4_thd_attention_internal_fwd_bwd_from_lengths expects torch tensors.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_attention_internal_fwd_bwd_from_lengths supports only causal=True.")

    q = q.detach()
    k = k.detach()
    v = v.detach()
    dout = dout.detach()
    batch_size, seq_len, _, head_dim = q.shape
    q_thd, k_thd, v_thd, cu_seqlens, max_seqlen = _torch_thd_views_from_segment_lengths(
        q,
        k,
        v,
        segment_lengths,
        num_segments,
    )
    dout_thd = dout.reshape(batch_size * seq_len, q.shape[2], head_dim)
    flash_attn_fwd, flash_attn_bwd = _flash_attn_internal_funcs()
    out, lse = flash_attn_fwd(
        q_thd,
        k_thd,
        v_thd,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_lse=True,
    )
    dq, dk, dv = flash_attn_bwd(
        q_thd,
        k_thd,
        v_thd,
        out,
        dout_thd,
        lse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    return (
        out.reshape(q.shape),
        dq.reshape(q.shape),
        dk.reshape(k.shape),
        dv.reshape(v.shape),
    )


def torch_fa4_thd_cu_lens_fwd_bwd(
    q: Any,
    k: Any,
    v: Any,
    dout: Any,
    cu_seqlens: Any,
    max_seqlen: int,
    *,
    causal: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Run upstream FA4 CuTe varlen fwd+bwd from precomputed cumulative sequence lengths."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v) and _is_torch_tensor(dout)):
        raise TypeError("torch_fa4_thd_cu_lens_fwd_bwd expects torch tensors.")
    if not _is_torch_tensor(cu_seqlens):
        raise TypeError("cu_seqlens must be a torch tensor.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_cu_lens_fwd_bwd supports only causal=True.")
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4 or len(dout.shape) != 4:
        raise ValueError("torch_fa4_thd_cu_lens_fwd_bwd expects q/k/v/dout with shape [B,S,H,D].")

    q = q.detach()
    k = k.detach()
    v = v.detach()
    dout = dout.detach()
    batch_size, seq_len, _, head_dim = q.shape
    q_thd = q.reshape(batch_size * seq_len, q.shape[2], head_dim)
    k_thd = k.reshape(batch_size * seq_len, k.shape[2], head_dim)
    v_thd = v.reshape(batch_size * seq_len, v.shape[2], head_dim)
    dout_thd = dout.reshape(batch_size * seq_len, q.shape[2], head_dim)
    flash_attn_fwd, flash_attn_bwd = _flash_attn_internal_funcs()
    out, lse = flash_attn_fwd(
        q_thd,
        k_thd,
        v_thd,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_lse=True,
    )
    dq, dk, dv = flash_attn_bwd(
        q_thd,
        k_thd,
        v_thd,
        out,
        dout_thd,
        lse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    return (
        out.reshape(q.shape),
        dq.reshape(q.shape),
        dk.reshape(k.shape),
        dv.reshape(v.shape),
    )


def torch_fa4_thd_attention(q: Any, k: Any, v: Any, segment_ids: tuple[Any, Any], *, causal: bool = True) -> Any:
    """Run upstream FA4 CuTe varlen attention on torch BSHD tensors with packed segment IDs."""
    if not (_is_torch_tensor(q) and _is_torch_tensor(k) and _is_torch_tensor(v)):
        raise TypeError("torch_fa4_thd_attention expects torch tensors.")
    if not causal:
        raise NotImplementedError("torch_fa4_thd_attention supports only causal=True.")
    mask = AttentionMask.causal().with_segment_ids(*segment_ids)
    _validate_simple_causal_self_attention(q, k, v, mask, backend_name="torch_fa4_thd_attention")

    batch_size, seq_len, _, head_dim = q.shape
    cu_seqlens, max_seqlen = _torch_thd_reshape_metadata(
        segment_ids[0],
        segment_ids[1],
        batch_size=batch_size,
        seq_len=seq_len,
    )
    out = _flash_attn_varlen_func()(
        q.reshape(batch_size * seq_len, q.shape[2], head_dim),
        k.reshape(batch_size * seq_len, k.shape[2], head_dim),
        v.reshape(batch_size * seq_len, v.shape[2], head_dim),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(batch_size, seq_len, q.shape[2], head_dim)


def _torch_from_jax(x: jax.Array) -> Any:
    import torch  # noqa: PLC0415

    try:
        return torch.utils.dlpack.from_dlpack(x)
    except Exception as exc:
        raise RuntimeError("gpu_fa4_thd_attention can only bridge concrete device JAX arrays to torch.") from exc


def _jax_from_torch(x: Any) -> jax.Array:
    return jax.dlpack.from_dlpack(x.contiguous())


def _jax_fa4_thd_attention_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
) -> jax.Array:
    q_t = _torch_from_jax(q)
    k_t = _torch_from_jax(k)
    v_t = _torch_from_jax(v)
    q_segment_ids_t = _torch_from_jax(q_segment_ids)
    kv_segment_ids_t = _torch_from_jax(kv_segment_ids)
    return _jax_from_torch(torch_fa4_thd_attention_internal(q_t, k_t, v_t, (q_segment_ids_t, kv_segment_ids_t)))


def _jax_fa4_thd_attention_lengths_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
) -> jax.Array:
    q_t = _torch_from_jax(q)
    k_t = _torch_from_jax(k)
    v_t = _torch_from_jax(v)
    segment_lengths_t = _torch_from_jax(segment_lengths)
    num_segments_t = _torch_from_jax(num_segments)
    return _jax_from_torch(
        torch_fa4_thd_attention_internal_from_lengths(q_t, k_t, v_t, segment_lengths_t, num_segments_t)
    )


@jax.custom_vjp
def _jax_fa4_thd_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
) -> jax.Array:
    return _jax_fa4_thd_attention_impl(q, k, v, q_segment_ids, kv_segment_ids)


def _jax_fa4_thd_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    return _jax_fa4_thd_attention_impl(q, k, v, q_segment_ids, kv_segment_ids), (
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
    )


def _jax_fa4_thd_attention_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None]:
    q, k, v, q_segment_ids, kv_segment_ids = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None, None

    q_t = _torch_from_jax(q).detach()
    k_t = _torch_from_jax(k).detach()
    v_t = _torch_from_jax(v).detach()
    dout_t = _torch_from_jax(cotangent).detach()
    q_segment_ids_t = _torch_from_jax(q_segment_ids)
    kv_segment_ids_t = _torch_from_jax(kv_segment_ids)
    _, dq_t, dk_t, dv_t = torch_fa4_thd_attention_internal_fwd_bwd(
        q_t,
        k_t,
        v_t,
        dout_t,
        (q_segment_ids_t, kv_segment_ids_t),
    )
    return _jax_from_torch(dq_t), _jax_from_torch(dk_t), _jax_from_torch(dv_t), None, None


_jax_fa4_thd_attention.defvjp(_jax_fa4_thd_attention_fwd, _jax_fa4_thd_attention_bwd)


@jax.custom_vjp
def _jax_fa4_thd_attention_lengths(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
) -> jax.Array:
    return _jax_fa4_thd_attention_lengths_impl(q, k, v, segment_lengths, num_segments)


def _jax_fa4_thd_attention_lengths_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_lengths: jax.Array,
    num_segments: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    return _jax_fa4_thd_attention_lengths_impl(q, k, v, segment_lengths, num_segments), (
        q,
        k,
        v,
        segment_lengths,
        num_segments,
    )


def _jax_fa4_thd_attention_lengths_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None]:
    q, k, v, segment_lengths, num_segments = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None, None

    q_t = _torch_from_jax(q).detach()
    k_t = _torch_from_jax(k).detach()
    v_t = _torch_from_jax(v).detach()
    dout_t = _torch_from_jax(cotangent).detach()
    segment_lengths_t = _torch_from_jax(segment_lengths)
    num_segments_t = _torch_from_jax(num_segments)
    _, dq_t, dk_t, dv_t = torch_fa4_thd_attention_internal_fwd_bwd_from_lengths(
        q_t,
        k_t,
        v_t,
        dout_t,
        segment_lengths_t,
        num_segments_t,
    )
    return _jax_from_torch(dq_t), _jax_from_torch(dk_t), _jax_from_torch(dv_t), None, None


_jax_fa4_thd_attention_lengths.defvjp(
    _jax_fa4_thd_attention_lengths_fwd,
    _jax_fa4_thd_attention_lengths_bwd,
)


def gpu_fa4_thd_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Experimental FA4 THD/varlen backend for simple causal self-attention.

    The upstream FA4 CuTe varlen API is torch/autograd-facing, so the JAX wrapper
    is eager-only and not suitable for production JIT training.
    """
    if _is_torch_tensor(q):
        _validate_simple_causal_self_attention(q, k, v, mask, backend_name="gpu_fa4_thd_attention")
        assert isinstance(mask, AttentionMask)
        if mask.thd_segment_metadata is not None:
            metadata = mask.thd_segment_metadata
            return torch_fa4_thd_attention_internal_from_lengths(
                q,
                k,
                v,
                metadata.segment_lengths,
                metadata.num_segments,
            )
        assert mask.segment_ids is not None
        return torch_fa4_thd_attention(q, k, v, mask.segment_ids)
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_thd_attention requires the JAX GPU backend.")
    _validate_simple_causal_self_attention(q, k, v, mask, backend_name="gpu_fa4_thd_attention")
    assert isinstance(mask, AttentionMask)
    if mask.thd_segment_metadata is not None:
        metadata = mask.thd_segment_metadata
        return _jax_fa4_thd_attention_lengths(q, k, v, metadata.segment_lengths, metadata.num_segments)
    assert mask.segment_ids is not None
    q_segment_ids, kv_segment_ids = mask.segment_ids
    return _jax_fa4_thd_attention(q, k, v, q_segment_ids, kv_segment_ids)


__all__ = [
    "gpu_fa4_thd_attention",
    "torch_fa4_thd_attention",
    "torch_fa4_thd_attention_internal",
    "torch_fa4_thd_attention_internal_from_lengths",
    "torch_fa4_thd_attention_internal_fwd_bwd",
    "torch_fa4_thd_attention_internal_fwd_bwd_from_lengths",
    "torch_fa4_thd_cu_lens_fwd_bwd",
]
