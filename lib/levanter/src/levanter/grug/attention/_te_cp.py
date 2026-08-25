# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Transformer Engine context-parallel attention for Grug.

Transformer Engine ships the distributed sequence protocol (ring / all-gather) and the matching
backward pass that Grug's FA4 kernels lack. This module is the production seam: it maps Grug's
``AttentionMask`` onto TE's THD padding-causal fused attention, applies TE's striped causal load
balancing at the attention boundary, and restores natural token order on the way out.

Hidden states stay in **contiguous** sequence shards everywhere else in the model. Grug's causal
SConv, fused RoPE, labels, loss, and packed-segment semantics all read natural token order, so the
striped permutation lives entirely inside this module: ``stripe_for_cp`` and ``unstripe_from_cp``
are exact inverses, and the same helper stripes the tensors and the sequence metadata so the two
cannot diverge. ``stripe_for_cp`` reproduces TE v2.17's ``reorder_causal_striped`` semantics,
confirmed against that source including its divisibility requirement.

Sharding contract: ``q``/``k``/``v`` arrive in natural token order under any sequence sharding, and
the output comes back in natural token order under ``q``'s own PartitionSpec, so the caller sees a
drop-in replacement for any other backend. The context-sharded striped layout exists only between
those two points. While the rest of the model keeps the sequence replicated, that costs one
all-gather of the attention output per call.

Runtime status: the pure-JAX parts of this module are unit tested on CPU, but no call into
Transformer Engine here has ever executed. TE 2.17.1 fails cuDNN backward workspace sizing with
``CUDNN_STATUS_BAD_PARAM`` on Marin's GB200 image, reproduced on NVIDIA's own unmodified CP4
example, so the fused-attention path stays unvalidated until that toolchain is fixed
(marin-community/marin#8141).

This backend shards only the attention boundary. A full context-parallel training step still needs
the rest of the model to hold hidden states in contiguous sequence shards -- a left halo for the
causal SConv sites, and loss and MoE routing reducing over the token axes -- otherwise the
attention output's context sharding is simply gathered away again.
"""

import functools
import importlib
import math
import os
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
from jax import core, numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.grug.attention._te_cp_config import ContextParallelStrategy, TeContextParallelConfig
from levanter.grug.sharding import _current_mesh, _mesh_axis_size, _mesh_has_axis

_BACKEND_NAME = "gpu_te_cp attention"
_TE_SETUP_HINT = (
    "Install it with the pinned GB200 recipe in experiments/grug/te_setup.py "
    "(TRANSFORMER_ENGINE_SETUP_SCRIPT plus transformer_engine_build_env())."
)
# TE 2.17 reads this per call and already defaults to "0" (the non-scan ring implementation, the
# only one that supports sliding windows), so this module only has to reject an inherited "1".
_RING_ATTENTION_SCAN_ENV = "NVTE_FUSED_RING_ATTENTION_USE_SCAN"
# TE's sequence descriptor reserves segment id 0 for padding; Grug marks padding with -1.
_TE_PADDING_SEGMENT_ID = 0
# Grug attention tensors and the [B, S] sequence metadata both carry the sequence on axis 1.
_SEQ_DIM = 1


@dataclass(frozen=True)
class TransformerEngineApi:
    """Every Transformer Engine symbol this backend touches.

    One explicit contract keeps TE API drift a single obvious failure instead of an import
    error deep inside a training step.
    """

    te: Any
    AttnBiasType: Any
    AttnMaskType: Any
    AttnSoftmaxType: Any
    CPStrategy: Any
    QKVLayout: Any
    SequenceDescriptor: Any
    fused_attn: Any
    is_fused_attn_kernel_available: Any
    MeshResource: Any


def load_transformer_engine() -> TransformerEngineApi:
    """Import Transformer Engine's JAX attention API, or raise with the install recipe."""
    try:
        te = importlib.import_module("transformer_engine.jax")
        attention = importlib.import_module("transformer_engine.jax.attention")
        sharding = importlib.import_module("transformer_engine.jax.sharding")
    except ImportError as exc:
        raise ImportError(f"{_BACKEND_NAME} requires Transformer Engine with JAX support. {_TE_SETUP_HINT}") from exc

    return TransformerEngineApi(
        te=te,
        AttnBiasType=attention.AttnBiasType,
        AttnMaskType=attention.AttnMaskType,
        AttnSoftmaxType=attention.AttnSoftmaxType,
        CPStrategy=attention.CPStrategy,
        QKVLayout=attention.QKVLayout,
        SequenceDescriptor=attention.SequenceDescriptor,
        fused_attn=attention.fused_attn,
        is_fused_attn_kernel_available=attention.is_fused_attn_kernel_available,
        MeshResource=sharding.MeshResource,
    )


def check_striped_sequence_length(seq_len: int, *, cp_size: int, stripe_size: int) -> None:
    """TE's striped reorder needs ``cp_size * stripe_size`` whole stripes per sequence.

    Only the DualChunkSwap strategy needs the stricter ``2 * cp_size`` split, so this is not the
    ``2 * cp_size * stripe_size`` factor the exact-shape CP benchmark used.
    """
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if stripe_size <= 0:
        raise ValueError(f"stripe_size must be positive, got {stripe_size}")
    factor = cp_size * stripe_size
    if seq_len % factor != 0:
        raise ValueError(
            f"{_BACKEND_NAME} requires seq_len divisible by cp_size * stripe_size = {factor}, got seq_len={seq_len}."
        )


def _stripe_group_count(seq_len: int, *, cp_size: int, stripe_size: int) -> int:
    check_striped_sequence_length(seq_len, cp_size=cp_size, stripe_size=stripe_size)
    return seq_len // (cp_size * stripe_size)


def stripe_for_cp(x: Array, *, cp_size: int, stripe_size: int, seq_dim: int) -> Array:
    """Reorder ``x`` along ``seq_dim`` into Transformer Engine's striped context-parallel layout.

    The sequence is cut into stripes of ``stripe_size`` tokens and stripe ``i`` is assigned to
    context rank ``i % cp_size``; rank ``r``'s stripes then land contiguously in shard ``r`` of
    the result, which is exactly what sharding the reordered array over the context axis gives
    each device. Every rank ends up with a mix of early and late tokens, so causal attention
    costs the same everywhere.

    ``seq_dim`` must be replicated: the reshape that splits it apart cannot be expressed on a
    sharded axis. Callers replicate the sequence first and shard the striped result.
    """
    groups = _stripe_group_count(x.shape[seq_dim], cp_size=cp_size, stripe_size=stripe_size)
    head, tail = x.shape[:seq_dim], x.shape[seq_dim + 1 :]
    split = jnp.reshape(x, (*head, groups, cp_size, stripe_size, *tail))
    return jnp.reshape(jnp.swapaxes(split, seq_dim, seq_dim + 1), x.shape)


def unstripe_from_cp(x: Array, *, cp_size: int, stripe_size: int, seq_dim: int) -> Array:
    """Exact inverse of :func:`stripe_for_cp`, restoring natural token order.

    ``seq_dim`` must be replicated, for the same reason.
    """
    groups = _stripe_group_count(x.shape[seq_dim], cp_size=cp_size, stripe_size=stripe_size)
    head, tail = x.shape[:seq_dim], x.shape[seq_dim + 1 :]
    split = jnp.reshape(x, (*head, cp_size, groups, stripe_size, *tail))
    return jnp.reshape(jnp.swapaxes(split, seq_dim, seq_dim + 1), x.shape)


def _segment_starts(segment_ids: Int[Array, "B S"]) -> Bool[Array, "B S"]:
    batch = segment_ids.shape[0]
    changed = segment_ids[:, 1:] != segment_ids[:, :-1]
    return jnp.concatenate([jnp.ones((batch, 1), dtype=jnp.bool_), changed], axis=1)


def segment_positions_from_segment_ids(segment_ids: Int[Array, "B S"]) -> Int[Array, "B S"]:
    """In-segment token index for contiguously packed ``segment_ids``.

    TE needs per-token positions alongside segment ids, and they must be the *original*
    in-segment indices so causality survives the striped permutation.
    """
    seq_len = segment_ids.shape[1]
    index = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    run_start = jax.lax.cummax(jnp.where(_segment_starts(segment_ids), index, 0), axis=1)
    return (index - run_start).astype(jnp.int32)


def te_segment_ids(segment_ids: Int[Array, "B S"]) -> Int[Array, "B S"]:
    """Translate Grug segment ids (negative = padding) to TE's (0 = padding) convention."""
    ids = segment_ids.astype(jnp.int32)
    return jnp.where(ids < 0, _TE_PADDING_SEGMENT_ID, ids + 1)


def documents_per_sequence(te_ids: Int[Array, "B S"]) -> Int[Array, "B"]:
    """Count packed documents per sequence, ignoring padding runs."""
    return jnp.sum(_segment_starts(te_ids) & (te_ids != _TE_PADDING_SEGMENT_ID), axis=1, dtype=jnp.int32)


def _replace_spec_entry(spec: PartitionSpec, dim: int, value: str | None) -> PartitionSpec:
    entries = list(spec)
    if dim >= len(entries):
        raise ValueError(f"PartitionSpec {spec} has no entry for dim {dim}")
    entries[dim] = value
    return PartitionSpec(*entries)


def context_parallel_spec(spec: PartitionSpec, *, seq_dim: int, context_axis: str) -> PartitionSpec:
    """Return ``spec`` with the (replicated) sequence dimension sharded over ``context_axis``.

    Every other dimension keeps whatever the caller chose. The sequence entry must be replicated on
    the way in: the striped tensor is built by reshaping a sequence-replicated array, and re-sharding
    an already-sharded sequence here would silently reinterpret which tokens a device holds.
    """
    entries = list(spec)
    if seq_dim >= len(entries):
        raise ValueError(f"PartitionSpec {spec} has no entry for sequence dim {seq_dim}")
    if entries[seq_dim] is not None:
        raise ValueError(
            f"{_BACKEND_NAME} shards the sequence itself, so it must be handed a replicated sequence "
            f"dim; {spec} already shards it over {entries[seq_dim]!r}."
        )
    return _replace_spec_entry(spec, seq_dim, context_axis)


def _partition_spec_of(x: Array, *, label: str) -> PartitionSpec:
    sharding = jax.typeof(x).sharding if isinstance(x, core.Tracer) else x.sharding
    if not isinstance(sharding, NamedSharding):
        raise TypeError(
            f"{_BACKEND_NAME} needs a NamedSharding on {label} to place it on the context axis; "
            f"got {sharding!r}. Run it under an explicit Grug mesh."
        )
    return sharding.spec


def _reshard(x: Array, *, mesh: Any, spec: PartitionSpec) -> Array:
    return reshard(x, NamedSharding(mesh, spec))


def _context_parallel_size(mesh: Any, config: TeContextParallelConfig) -> int:
    if not _mesh_has_axis(mesh, config.context_axis):
        raise ValueError(
            f"{_BACKEND_NAME} requires a mesh with a {config.context_axis!r} axis; build the mesh with "
            "compact_grug_mesh(context_axis_size=...)."
        )
    cp_size = _mesh_axis_size(mesh, config.context_axis)
    if cp_size < 2:
        raise ValueError(f"{_BACKEND_NAME} requires {config.context_axis!r} axis size >= 2, got {cp_size}")
    return cp_size


def check_batch_sharding(spec: PartitionSpec, *, data_axis: str) -> None:
    """TE's ``MeshResource.dp_resource`` names one mesh axis, so the batch dim may use only one.

    Grug's own activation specs shard the batch over the compound ``("replica_dcn", "data")`` (plus
    ``"expert"`` on the EP variants), which TE cannot be told about; such a batch has to be
    resharded onto a single axis before this backend can run.
    """
    entry = spec[0]
    if entry is None:
        return
    axes = entry if isinstance(entry, tuple) else (entry,)
    if len(axes) > 1 or axes[0] != data_axis:
        raise ValueError(
            f"{_BACKEND_NAME} can only describe a batch sharded over the single mesh axis {data_axis!r} "
            f"to Transformer Engine (MeshResource.dp_resource), got {entry!r}."
        )


def _check_ring_attention_scan_env() -> None:
    if os.environ.get(_RING_ATTENTION_SCAN_ENV) == "1":
        raise ValueError(
            f"{_RING_ATTENTION_SCAN_ENV}=1 selects Transformer Engine's scan ring implementation, which "
            f"does not support sliding windows; unset it or set it to 0 for {_BACKEND_NAME}."
        )


def _validate_inputs(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> AttentionMask:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{_BACKEND_NAME} does not support dense masks.")
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError(f"{_BACKEND_NAME} requires an AttentionMask.")
    # TE's THD context-parallel kernels implement only the padding-causal mask, with no bias,
    # no dropout, and vanilla softmax.
    if not mask.is_causal:
        raise NotImplementedError(f"{_BACKEND_NAME} supports only causal self-attention.")
    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
    # TE's window_size is a static kernel argument, so the window has to be readable from the mask's
    # static field. A mask carrying FA4 metadata hides its real window in ``fa4_bounds`` -- that is
    # how the hero layer scan varies the window per layer -- which would silently run full causal.
    if mask.fa4_bounds is not None or mask.thd_segment_metadata is not None:
        raise NotImplementedError(
            f"{_BACKEND_NAME} cannot read a mask carrying FA4 metadata: its window and segment bounds "
            "live in fa4_bounds/thd_segment_metadata while TE needs a static sliding_window. Pass an "
            "AttentionMask with segment_ids and sliding_window only."
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{_BACKEND_NAME} expects q/k/v with shape [B,S,H,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{_BACKEND_NAME} requires matching batch sizes, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise NotImplementedError(f"{_BACKEND_NAME} supports only self-attention with q_len == kv_len.")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"{_BACKEND_NAME} requires matching K/V heads, got k={k.shape}, v={v.shape}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{_BACKEND_NAME} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(f"{_BACKEND_NAME} requires Dq == Dk == Dv, got {q.shape}, {k.shape}, {v.shape}")
    return mask


def te_window_size(mask: AttentionMask) -> tuple[int, int]:
    """Grug's sliding window as TE's ``(left, right)`` inclusive bounds.

    Grug's ``sliding_window=W`` keeps the last ``W`` tokens *including* the query
    (``materialize_mask``: ``k >= q - (W - 1)``), so the leftmost attended key is ``W - 1``
    positions back. ``(-1, -1)`` is TE's "unbounded".
    """
    if mask.sliding_window is None:
        return (-1, -1)
    return (mask.sliding_window - 1, 0)


def _sequence_metadata(
    mask: AttentionMask, *, batch: int, seq_len: int, max_segments_per_seq: int
) -> tuple[Int[Array, "B S"], Int[Array, "B S"]]:
    if mask.segment_ids is None:
        ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    else:
        q_ids, kv_ids = mask.segment_ids
        if q_ids.shape != kv_ids.shape:
            raise ValueError(f"{_BACKEND_NAME} requires matching q/kv segment_ids, got {q_ids.shape}, {kv_ids.shape}")
        # TE describes one packed sequence per batch row, so a q packing that differs from the kv
        # packing has no representation here.
        q_ids = eqx.error_if(
            q_ids,
            jnp.any(q_ids != kv_ids),
            f"{_BACKEND_NAME} requires matching q/kv segment_ids.",
        )
        ids = jnp.broadcast_to(jnp.reshape(q_ids, (-1, seq_len)), (batch, seq_len))

    ids = te_segment_ids(ids)
    # ``max_segments_per_seq`` is baked into the fused kernel, so an under-sized bound silently
    # drops documents; FA4's THD metadata guards its own bound the same way.
    ids = eqx.error_if(
        ids,
        jnp.any(documents_per_sequence(ids) > max_segments_per_seq),
        f"{_BACKEND_NAME}: packed segment_ids contain more documents than max_segments_per_seq.",
    )
    return ids, segment_positions_from_segment_ids(ids)


def _te_strategy(te_api: TransformerEngineApi, strategy: ContextParallelStrategy) -> Any:
    if strategy == ContextParallelStrategy.RING:
        return te_api.CPStrategy.RING
    if strategy == ContextParallelStrategy.ALL_GATHER:
        return te_api.CPStrategy.ALL_GATHER
    raise ValueError(f"Unsupported context-parallel strategy: {strategy}")


def _check_kernel_available(
    te_api: TransformerEngineApi,
    *,
    dtype: Any,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    window_size: tuple[int, int],
) -> None:
    available = te_api.is_fused_attn_kernel_available(
        True,
        dtype,
        dtype,
        te_api.QKVLayout.THD_THD_THD,
        te_api.AttnBiasType.NO_BIAS,
        te_api.AttnMaskType.PADDING_CAUSAL_MASK,
        te_api.AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,
        num_q_heads,
        num_kv_heads,
        seq_len,
        seq_len,
        head_dim,
        head_dim,
        window_size,
    )
    if not available:
        raise NotImplementedError(
            f"{_BACKEND_NAME}: Transformer Engine reports no fused-attention backend for "
            f"dtype={dtype}, Hq={num_q_heads}, Hkv={num_kv_heads}, S={seq_len}, D={head_dim}, "
            f"window={window_size}."
        )


def gpu_te_cp_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    config: TeContextParallelConfig,
) -> Float[Array, "B Q Hq D"]:
    """Context-parallel causal attention through Transformer Engine's THD fused kernels.

    ``q``/``k``/``v`` arrive in natural token order and the output leaves in natural token order
    under ``q``'s PartitionSpec; the striped, context-sharded layout exists only between the two.
    """
    attention_mask = _validate_inputs(q, k, v, mask)
    te_api = load_transformer_engine()

    mesh = _current_mesh()
    cp_size = _context_parallel_size(mesh, config)
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    window_size = te_window_size(attention_mask)
    check_striped_sequence_length(seq_len, cp_size=cp_size, stripe_size=config.stripe_size)
    if config.strategy == ContextParallelStrategy.RING:
        _check_ring_attention_scan_env()
    q_spec = _partition_spec_of(q, label="q")
    check_batch_sharding(q_spec, data_axis=config.data_axis)
    _check_kernel_available(
        te_api,
        dtype=q.dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        window_size=window_size,
    )

    segment_ids, segment_positions = _sequence_metadata(
        attention_mask,
        batch=batch,
        seq_len=seq_len,
        max_segments_per_seq=config.max_segments_per_seq,
    )
    # One helper for tensors and metadata: the permutation cannot drift between them.
    stripe = functools.partial(stripe_for_cp, cp_size=cp_size, stripe_size=config.stripe_size, seq_dim=_SEQ_DIM)

    def to_striped_context_shard(x: Array, *, label: str) -> Array:
        """Replicate the sequence, stripe it, then hand context rank ``r`` its stripes as shard ``r``."""
        natural_spec = _replace_spec_entry(_partition_spec_of(x, label=label), _SEQ_DIM, None)
        striped_spec = context_parallel_spec(natural_spec, seq_dim=_SEQ_DIM, context_axis=config.context_axis)
        natural = _reshard(x, mesh=mesh, spec=natural_spec)
        return _reshard(stripe(natural), mesh=mesh, spec=striped_spec)

    striped_qkv = tuple(
        to_striped_context_shard(value, label=label) for value, label in ((q, "q"), (k, "k"), (v, "v"))
    )
    descriptor = te_api.SequenceDescriptor.from_segment_ids_and_pos(
        to_striped_context_shard(segment_ids, label="segment_ids"),
        to_striped_context_shard(segment_positions, label="segment_positions"),
    )

    mesh_resource = te_api.MeshResource(dp_resource=config.data_axis, cp_resource=config.context_axis)
    with te_api.te.autocast(mesh_resource=mesh_resource):
        out = te_api.fused_attn(
            striped_qkv,
            None,  # bias
            descriptor,
            None,  # dropout rng
            attn_bias_type=te_api.AttnBiasType.NO_BIAS,
            attn_mask_type=te_api.AttnMaskType.PADDING_CAUSAL_MASK,
            qkv_layout=te_api.QKVLayout.THD_THD_THD,
            softmax_type=te_api.AttnSoftmaxType.VANILLA_SOFTMAX,
            scaling_factor=1.0 / math.sqrt(head_dim),
            dropout_probability=0.0,
            is_training=True,  # this is a training backend; TE's inference path drops the softmax residuals
            max_segments_per_seq=config.max_segments_per_seq,
            window_size=window_size,
            context_parallel_strategy=_te_strategy(te_api, config.strategy),
            context_parallel_causal_load_balanced=True,
            context_parallel_axis=config.context_axis,
            stripe_size=config.stripe_size,
        )

    # Gather the striped output before un-permuting it: the inverse reshape cannot split a
    # context-sharded sequence. This is the collective the contiguous-hidden-state design costs.
    out_spec = _partition_spec_of(out, label="fused_attn output")
    gathered = _reshard(out, mesh=mesh, spec=_replace_spec_entry(out_spec, _SEQ_DIM, None))
    natural = unstripe_from_cp(gathered, cp_size=cp_size, stripe_size=config.stripe_size, seq_dim=_SEQ_DIM)
    return _reshard(natural.astype(v.dtype), mesh=mesh, spec=q_spec)
