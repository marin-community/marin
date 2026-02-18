# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import functools
import math
import re
import warnings
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, cast

import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int

from haliax.jax_utils import named_call
from haliax.partitioning import _get_mesh
from levanter.kernels.pallas.attention import infer_block_sizes as _infer_attention_block_sizes

Backend: TypeAlias = Literal["pallas_tpu", "pallas_gpu", "reference"]


@dataclass(frozen=True)
class RotaryConfig:
    """Lightweight rotary embedding configuration."""

    theta: float = 10000.0
    scaling_factor: float | None = None


@functools.partial(register_dataclass, data_fields=["segment_ids"], meta_fields=["is_causal", "sliding_window"])
@dataclass(frozen=True)
class AttentionMask:
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = False
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = None

    @classmethod
    def causal(cls, *, sliding_window: int | None = None) -> "AttentionMask":
        return cls(is_causal=True, sliding_window=sliding_window)

    def with_segment_ids(
        self, q_segment_ids: Int[Array, "..."], kv_segment_ids: Int[Array, "..."] | None = None
    ) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=(q_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
        )

    def materialize_mask(self, q_len: int, k_len: int) -> Bool[Array, "..."] | None:
        """Return a boolean mask (True = allowed) or None.

        Shapes:
          - If `segment_ids` is unset, returns `(q_len, k_len)` (broadcastable across batch).
          - If `segment_ids` is set with per-batch IDs, returns `(batch, q_len, k_len)`.
        """
        mask = None

        if self.is_causal:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx <= q_idx
            mask = allowed

        if self.sliding_window is not None:
            if self.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            # Standard sliding-window semantics: `sliding_window=W` means "keep the last W tokens,
            # including self". Without causality, this is "don't look too far back":
            #   k >= q - (W - 1)
            allowed = k_idx >= q_idx - (self.sliding_window - 1)
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        if self.segment_ids is not None:
            q_seg, k_seg = self.segment_ids
            if q_seg.ndim != k_seg.ndim:
                raise ValueError(f"segment_ids ndim mismatch: q={q_seg.ndim}, k={k_seg.ndim}")
            if q_seg.ndim == 1:
                allowed = q_seg[:, None] == k_seg[None, :]
            elif q_seg.ndim == 2:
                if q_seg.shape[0] != k_seg.shape[0]:
                    raise ValueError(f"segment_ids batch mismatch: q={q_seg.shape[0]}, k={k_seg.shape[0]}")
                allowed = q_seg[:, :, None] == k_seg[:, None, :]
            else:
                raise ValueError(f"segment_ids must be 1D or 2D, got ndim={q_seg.ndim}")
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        return mask


def _rotary_cache(seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[Float[Array, "S D"], Float[Array, "S D"]]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


@named_call
def apply_rotary_embedding(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    seq_len: int,
    head_dim: int,
    rope: RotaryConfig,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


def reference_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    logits_dtype: jnp.dtype | None,
) -> Float[Array, "B Q Hq D"]:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        repeat = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)

    explicit = None
    if mask is None:
        explicit = None
    elif isinstance(mask, AttentionMask):
        explicit = mask.materialize_mask(scores.shape[-2], scores.shape[-1])
    else:
        explicit = mask

    if explicit is not None:
        # Standardize dense masks to [B, Q, K] (bool = allowed, float = additive bias).
        if explicit.ndim == 2:
            explicit = explicit[None, :, :]
        if explicit.ndim != 3:
            raise ValueError(f"explicit mask must have shape [batch, q, k], got shape={explicit.shape}")
        if explicit.shape[0] not in (1, q.shape[0]):
            raise ValueError(f"explicit mask batch dim must be 1 or {q.shape[0]}, got {explicit.shape[0]}")
        if explicit.shape[1] != scores.shape[-2] or explicit.shape[2] != scores.shape[-1]:
            raise ValueError(
                "explicit mask must match attention shapes: "
                f"got mask={explicit.shape}, expected [batch,{scores.shape[-2]},{scores.shape[-1]}]"
            )

        explicit = explicit[:, None, :, :]  # -> [B, 1, Q, K], broadcast across heads.
        if explicit.dtype == jnp.bool_:
            scores = jnp.where(explicit, scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            scores = scores + explicit
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _spec_shard_factor(entry: str | tuple[str, ...] | None, mesh) -> int:
    """
    Compute the size of the mesh axes associated with a PartitionSpec entry.

    Splash attention can handle various kinds of sequence parallelism but needs this to function
    """
    if entry is None or mesh is None:
        return 1
    if isinstance(entry, str):
        return mesh.shape[entry]
    if isinstance(entry, tuple):
        factor = 1
        for name in entry:
            factor *= mesh.shape[name]
        return factor
    raise TypeError(f"Unsupported PartitionSpec entry: {entry!r}")


def _tpu_splash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B Q Hq D"]:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
    from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds

    # Splash attention expects BHSD.
    q_ = jnp.transpose(q, (0, 2, 1, 3))
    k_ = jnp.transpose(k, (0, 2, 1, 3))
    v_ = jnp.transpose(v, (0, 2, 1, 3))

    B, Hq, Sq, D = q_.shape
    _, _, Sk, _ = k_.shape

    if Sk % 128 != 0:
        raise NotImplementedError("Splash attention requires key/value sequence length to be a multiple of 128.")

    q_ = q_ * (1.0 / math.sqrt(D))

    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        raise RuntimeError("TPU splash attention requires a JAX mesh context.")

    def _named_sharding_of(x: jax.Array, *, label: str) -> NamedSharding:
        """Extract NamedSharding from a JAX value or tracer.

        In JAX, `.sharding` is not available on tracers during staging; however the sharding
        is still available on the underlying abstract value in many cases.
        """
        sharding = None
        try:
            sharding = x.sharding  # type: ignore[attr-defined]
        except Exception:
            sharding = None
        if sharding is None:
            aval = getattr(x, "aval", None)
            sharding = getattr(aval, "sharding", None) if aval is not None else None
        if not isinstance(sharding, NamedSharding):
            raise TypeError(
                f"TPU splash attention expects NamedSharding on {label} under an explicit mesh; got {sharding!r}."
            )
        return sharding

    q_sharding = _named_sharding_of(q_, label="q")
    k_sharding = _named_sharding_of(k_, label="k")
    v_sharding = _named_sharding_of(v_, label="v")

    q_pspec = q_sharding.spec
    k_pspec = k_sharding.spec
    v_pspec = v_sharding.spec

    # KV sequence must not be sharded for splash attention.
    if k_pspec[2] is not None:
        raise NotImplementedError(
            "Splash attention does not support sharding the KV sequence dimension. "
            f"Got KV sequence spec: {k_pspec[2]}"
        )

    head_shards = _spec_shard_factor(q_pspec[1], mesh)
    q_seq_shards = _spec_shard_factor(q_pspec[2], mesh)
    kv_seq_shards = _spec_shard_factor(k_pspec[2], mesh)

    # MaxText uses a block size of 512. Pick per-shard blocks that evenly divide each shard length,
    # preferring multiples of 128 when possible.
    block_size = 512

    shard_Sq = max(1, Sq // max(1, q_seq_shards))
    shard_Sk = max(1, Sk // max(1, kv_seq_shards))

    def _compatible_block(shard_len: int, max_block: int) -> int:
        """Pick largest block <= max_block that divides shard_len; prefer multiples of 128."""
        if shard_len <= 0:
            return max_block
        cap = min(max_block, shard_len)
        for step in (128, 1):
            candidate = cap - (cap % step)
            while candidate >= step:
                if shard_len % candidate == 0:
                    return candidate
                candidate -= step
        return 1

    block_q = _compatible_block(shard_Sq, block_size)
    block_kv = _compatible_block(shard_Sk, block_size)

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv_compute=block_kv,
        block_kv=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_q,
        block_q_dq=block_q,
        block_kv_dq=block_kv,
    )

    if mask is None:
        base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
        segment_ids = None
        segment_ids_axes = None
        segment_batch_axis = None
    elif isinstance(mask, AttentionMask):
        if mask.is_causal:
            base_mask = splash_attention_mask.CausalMask((Sq, Sk), offset=0, shard_count=q_seq_shards)
        else:
            base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))

        if mask.sliding_window is not None:
            if mask.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
            local_mask = splash_attention_mask.LocalMask(
                shape=(Sq, Sk),
                # Grug's `sliding_window` matches the "lookback" semantics used in the
                # reference mask materialization: allow attending to keys with
                #   k >= q - (sliding_window - 1)
                # (and optionally combine with causal).
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids
            q_seg_sharding = _named_sharding_of(q_segment_ids, label="segment_ids.q")
            kv_seg_sharding = _named_sharding_of(kv_segment_ids, label="segment_ids.kv")
            segment_ids = SplashSegmentIds(q_segment_ids, kv_segment_ids)
            segment_ids_axes = SplashSegmentIds(
                q_seg_sharding.spec,
                kv_seg_sharding.spec,
            )
            segment_batch_axis = SplashSegmentIds(
                0 if q_segment_ids.ndim == 2 else None,
                0 if kv_segment_ids.ndim == 2 else None,
            )
        else:
            segment_ids = None
            segment_ids_axes = None
            segment_batch_axis = None
    else:
        raise NotImplementedError("Dense masks are not supported for splash attention.")

    kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(Hq)])

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=kernel_mask,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )

    kernel_sharding = NamedSharding(mesh, P(q_pspec[1], q_pspec[2]))
    kernel_specs = splash_kernel.manual_sharding_spec(kernel_sharding)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_pspec, k_pspec, v_pspec, segment_ids_axes, kernel_specs),
        out_specs=q_pspec,
        check_rep=False,
    )
    def wrap(q_bhsd, k_bhsd, v_bhsd, seg_ids, kernel):
        return jax.vmap(kernel, in_axes=(0, 0, 0, segment_batch_axis))(q_bhsd, k_bhsd, v_bhsd, seg_ids)

    out = wrap(q_, k_, v_, segment_ids, splash_kernel)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def _gpu_block_size(seq_len: int, max_block: int = 128, *, block_override: int | None = None) -> int:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    if block_override is not None:
        if block_override <= 0:
            raise ValueError(f"block_override must be positive, got {block_override}")
        max_block = block_override

    cap = min(seq_len, max_block)
    for preferred in (16, 1):
        block = cap - (cap % preferred)
        while block >= preferred:
            if seq_len % block == 0:
                return block
            block -= preferred
    return 1


@dataclass(frozen=True, slots=True)
class BlockSizes:
    """GPU attention tiling configuration."""

    block_q: int | None = None
    block_k: int | None = None
    block_q_dkv: int | None = None
    block_kv_dkv: int | None = None
    block_q_dq: int | None = None
    block_kv_dq: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None


def _resolved_gpu_block_sizes(
    block_sizes: BlockSizes,
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> BlockSizes:
    tuned = _infer_attention_block_sizes(
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype=dtype,
    )
    return BlockSizes(
        block_q=block_sizes.block_q if block_sizes.block_q is not None else tuned.block_q,
        block_k=block_sizes.block_k if block_sizes.block_k is not None else tuned.block_k,
        block_q_dkv=block_sizes.block_q_dkv if block_sizes.block_q_dkv is not None else tuned.block_q_dkv,
        block_kv_dkv=block_sizes.block_kv_dkv if block_sizes.block_kv_dkv is not None else tuned.block_kv_dkv,
        block_q_dq=block_sizes.block_q_dq if block_sizes.block_q_dq is not None else tuned.block_q_dq,
        block_kv_dq=block_sizes.block_kv_dq if block_sizes.block_kv_dq is not None else tuned.block_kv_dq,
        num_warps=block_sizes.num_warps if block_sizes.num_warps is not None else tuned.num_warps,
        num_stages=block_sizes.num_stages if block_sizes.num_stages is not None else tuned.num_stages,
    )


def _backend_block_sizes(
    block_sizes: BlockSizes | dict[str, BlockSizes] | None,
    backend: Backend,
) -> BlockSizes:
    if block_sizes is None:
        resolved = BlockSizes()
    elif isinstance(block_sizes, BlockSizes):
        resolved = block_sizes
    elif not isinstance(block_sizes, dict):
        raise TypeError(
            f"block_sizes must be BlockSizes, dict[str, BlockSizes], or None; got {type(block_sizes).__name__}"
        )
    else:
        if not all(isinstance(k, str) and isinstance(v, BlockSizes) for k, v in block_sizes.items()):
            raise TypeError("dict block_sizes entries must be Backend -> BlockSizes")
        block_lookup = {_normalize_block_size_key(k): v for k, v in block_sizes.items()}
        resolved = BlockSizes()
        for key in _backend_block_size_keys(backend):
            matched = block_lookup.get(key)
            if matched is not None:
                resolved = matched
                break

    return resolved


def _normalize_block_size_key(raw: str | Backend) -> str:
    normalized = str(raw).replace("_", " ").strip().lower()
    return " ".join(normalized.split())


def _backend_block_size_keys(backend: Backend) -> list[str]:
    backend_key = _normalize_block_size_key(backend)
    backend_key_no_prefix = backend_key.replace("pallas ", "")
    candidates: list[str] = []

    def add_candidate(candidate: str | None) -> None:
        if candidate is None:
            return
        if candidate not in candidates:
            candidates.append(candidate)

    try:
        device = jax.devices()[0]
    except Exception:
        add_candidate(backend_key)
        add_candidate(backend_key_no_prefix)
        add_candidate("auto")
        return candidates

    kind = getattr(device, "device_kind", None)
    if kind is None:
        add_candidate(backend_key)
        add_candidate(backend_key_no_prefix)
        add_candidate("auto")
        return candidates

    norm_kind = _normalize_block_size_key(str(kind))
    add_candidate(norm_kind)

    parts = norm_kind.split()
    if parts:
        add_candidate(parts[0])
        add_candidate(parts[-1])
        if len(parts) >= 2:
            full_kind_key = f"{parts[0]} {parts[1]}"
            add_candidate(full_kind_key)
            m = re.fullmatch(r"v(\d+)[a-z0-9]*", parts[1])
            if m:
                family_key = f"{parts[0]} {m.group(1)}"
                add_candidate(family_key)
                v_family_key = f"{parts[0]} v{m.group(1)}"
                add_candidate(v_family_key)

    add_candidate(backend_key)
    add_candidate(backend_key_no_prefix)
    add_candidate("auto")

    return candidates


def _broadcast_segment_ids_for_gpu(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    name: str,
) -> jax.Array:
    if segment_ids.ndim == 1:
        if batch_size != 1:
            raise NotImplementedError(
                f"{name} is 1D but batch size is {batch_size}; "
                "1D segment ids are only supported for unbatched inputs."
            )
        if segment_ids.shape[0] != seq_len:
            raise ValueError(f"{name} must have length {seq_len}, got {segment_ids.shape[0]}")
        return segment_ids[None, :]

    if segment_ids.ndim == 2:
        if segment_ids.shape[1] != seq_len:
            raise ValueError(f"{name} must have shape [batch, {seq_len}], got {segment_ids.shape}")
        if segment_ids.shape[0] == 1:
            if batch_size == 1:
                return segment_ids
            return jnp.tile(segment_ids, (batch_size, 1))
        if segment_ids.shape[0] != batch_size:
            raise NotImplementedError(
                f"{name} expects batch dimension {batch_size} or a shared singleton batch, got shape "
                f"{segment_ids.shape}."
            )
        return segment_ids

    raise NotImplementedError(
        f"{name} must be [S] for unbatched inputs or 2D with shape [1, {seq_len}] / [batch, {seq_len}], "
        f"got ndim={segment_ids.ndim}."
    )


def _gpu_splash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | jax.Array | None,
    block_sizes: BlockSizes | None = None,
) -> Float[Array, "B Q Hq D"]:
    from jax.experimental.pallas.ops.gpu.attention import BlockSizes as _MhaBlockSizes
    from levanter.kernels.pallas.attention import mha as gpu_mha

    B, Q, Hq, D = q.shape
    _, K, Hkv, D_kv = k.shape
    if D != D_kv:
        raise ValueError(f"q and k must have equal head dimension, got q={D}, k={D_kv}")
    if k.shape[1] != v.shape[1] or Hkv != v.shape[2] or D_kv != v.shape[3]:
        raise ValueError("k and v must have matching [B, K, Hkv, D] shapes")
    if Hq != Hkv:
        if Hq % Hkv != 0:
            raise ValueError(f"num_heads ({Hq}) must be divisible by num_kv_heads ({Hkv})")
        repeat = Hq // Hkv
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    causal = False
    segment_ids = None

    if isinstance(mask, AttentionMask):
        if mask.sliding_window is not None and mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
        causal = bool(mask.is_causal)

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids
            q_segment_ids = _broadcast_segment_ids_for_gpu(
                q_segment_ids, batch_size=B, seq_len=Q, name="query segment ids"
            )
            kv_segment_ids = _broadcast_segment_ids_for_gpu(
                kv_segment_ids, batch_size=B, seq_len=K, name="kv segment ids"
            )
            segment_ids = (q_segment_ids, kv_segment_ids)
    elif mask is not None:
        raise NotImplementedError("GPU splash attention supports only AttentionMask or None masks.")

    if block_sizes is None:
        block_sizes = _resolved_gpu_block_sizes(
            BlockSizes(),
            batch=B,
            seq_len=Q,
            num_heads=Hq,
            head_dim=D,
            dtype=q.dtype,
        )
    else:
        block_sizes = _resolved_gpu_block_sizes(
            block_sizes,
            batch=B,
            seq_len=Q,
            num_heads=Hq,
            head_dim=D,
            dtype=q.dtype,
        )

    q_block = _gpu_block_size(Q, block_override=block_sizes.block_q)
    k_block = _gpu_block_size(K, block_override=block_sizes.block_k)
    if block_sizes.block_q_dkv is not None:
        block_q_dkv = _gpu_block_size(Q, block_override=block_sizes.block_q_dkv)
    else:
        block_q_dkv = None
    if block_sizes.block_kv_dkv is not None:
        block_kv_dkv = _gpu_block_size(K, block_override=block_sizes.block_kv_dkv)
    else:
        block_kv_dkv = None
    if block_sizes.block_q_dq is not None:
        block_q_dq = _gpu_block_size(Q, block_override=block_sizes.block_q_dq)
    else:
        block_q_dq = None
    if block_sizes.block_kv_dq is not None:
        block_kv_dq = _gpu_block_size(K, block_override=block_sizes.block_kv_dq)
    else:
        block_kv_dq = None

    mha_block_sizes = _MhaBlockSizes(
        block_q=q_block,
        block_k=k_block,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
    )

    def _run_fast() -> Float[Array, "B Q Hq D"]:
        return gpu_mha(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sm_scale=1.0 / math.sqrt(D),
            causal=causal,
            sliding_window=mask.sliding_window if isinstance(mask, AttentionMask) else None,
            block_sizes=mha_block_sizes,
            num_warps=block_sizes.num_warps if block_sizes is not None else None,
            num_stages=block_sizes.num_stages if block_sizes is not None else None,
        ).astype(v.dtype)

    return _run_fast()


def _ensure_attention_batch_dim(
    q: Float[Array, "B Q Hq D"] | Float[Array, "Q Hq D"],
    k: Float[Array, "B K Hkv D"] | Float[Array, "K Hkv D"],
    v: Float[Array, "B K Hkv D"] | Float[Array, "K Hkv D"],
) -> tuple[
    Float[Array, "B Q Hq D"], Float[Array, "B K Hkv D"], Float[Array, "B K Hkv D"], bool
]:
    if q.ndim == 4 and k.ndim == 4 and v.ndim == 4:
        return q, k, v, False

    if q.ndim == 3 and k.ndim == 3 and v.ndim == 3:
        return q[None, ...], k[None, ...], v[None, ...], True

    raise ValueError(f"q/k/v must be all rank-4 (batched) or all rank-3 (unbatched), got ndim={[q.ndim, k.ndim, v.ndim]}")


Implementation: TypeAlias = Literal["pallas_tpu", "pallas_gpu", "reference"]
ArrayImpl = Callable[
    [Float[Array, "B Q Hq D"], Float[Array, "B K Hkv D"], Float[Array, "B K Hkv D"], AttentionMask | jnp.ndarray | None],
    Float[Array, "B Q Hq D"],
]
BackendBlockSizes: TypeAlias = BlockSizes | dict[str, BlockSizes]


def _reference_attention_impl(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)


def _default_implementation() -> tuple[Implementation, ...]:
    backend = jax.default_backend()
    if backend == "tpu":
        return ("pallas_tpu", "reference")
    if backend == "gpu":
        return ("pallas_gpu", "reference")
    return ("reference",)


IMPLEMENTATIONS: dict[str, ArrayImpl] = {
    "reference": _reference_attention_impl,
    "pallas_gpu": _gpu_splash_attention,
    "pallas_tpu": _tpu_splash_attention,
}


def attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
    block_sizes: BackendBlockSizes | None = None,
) -> Float[Array, "B Q Hq D"]:
    q, k, v, squeeze_batch = _ensure_attention_batch_dim(q, k, v)

    if implementation is None:
        impls: Sequence[Implementation | ArrayImpl] = _default_implementation()
        explicit = False
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation | ArrayImpl], implementation)
        explicit = len(impls) == 1
    else:
        impls = (cast(Implementation, implementation),)
        explicit = True

    errors: list[Exception] = []
    for impl in impls:
        if callable(impl):
            try:
                result = impl(q, k, v, mask)
                return result[0] if squeeze_batch else result
            except NotImplementedError as e:
                if explicit:
                    raise
                warnings.warn(
                    "Pallas splash attention unavailable, falling back to reference: "
                    f"{e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue
        else:
            fn = IMPLEMENTATIONS.get(cast(str, impl))
            if fn is None:
                raise ValueError(f"Unsupported implementation: {impl}")
            try:
                if impl == "pallas_gpu":
                    impl_block_sizes = _backend_block_sizes(
                        block_sizes,
                        "pallas_gpu",
                    )
                    result = fn(
                        q,
                        k,
                        v,
                        mask,
                        block_sizes=impl_block_sizes,
                    )
                else:
                    result = fn(q, k, v, mask)
                return result[0] if squeeze_batch else result
            except NotImplementedError as e:
                if explicit:
                    raise
                warnings.warn(
                    "Pallas splash attention unavailable, falling back to reference: "
                    f"{e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue
    raise ExceptionGroup("all implementations failed", errors)


__all__ = [
    "AttentionMask",
    "BlockSizes",
    "RotaryConfig",
    "apply_rotary_embedding",
    "Implementation",
    "attention",
    "reference_attention",
]
