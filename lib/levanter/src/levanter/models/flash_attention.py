# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# cf https://github.com/lucidrains/flash-attention-jax
# cf https://tridao.me/publications/flash2/flash2.pdf
# cf https://arxiv.org/pdf/2205.14135.pdf
from typing import Optional, Tuple

import equinox
import jax
import jax.numpy as jnp
from equinox import filter_eval_shape
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import AxisSelection, AxisSpec, ds
from haliax.jax_utils import named_call
from haliax.types import PrecisionLike

from levanter.layers.attention import AttentionMask, materialize_mask


BLOCK_SIZE = 1024


@named_call
def flash_attention(
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: Optional[AttentionMask | hax.NamedArray] = None,
    bias: Optional[hax.NamedArray] = None,
    *,
    dropout: float = 0.0,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    scaling_factor: float | None = None,
    logits_soft_cap: float | None = None,
):
    """
    Crappy pure-jax Flash Attention impl, vaguely following the v2 paper.

    Args:
        Key: axis of key dim.
    """
    if not inference and dropout > 0 and key is None:
        raise ValueError("key must be provided for training")

    if dropout < 0 or dropout > 1:
        raise ValueError(f"invalid dropout {dropout}")

    # TODO: is it maybe better to do this just-in-time blockwise?
    if dtype is not None:
        q = q.astype(dtype)
        k = k.astype(dtype)

    if block_size is None:
        block_size = BLOCK_SIZE

    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)

    if QPos.size < block_size or KPos.size < block_size:
        from levanter.layers.attention import simple_attention_with_dropout

        return simple_attention_with_dropout(
            QPos,
            KPos,
            Key,
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            dropout=dropout,
            inference=inference,
            prng=key,
            logits_soft_cap=logits_soft_cap,
        )

    if scaling_factor is None:
        if Key.size == 0:
            raise ValueError("Key axis must be non-empty")
        scaling_factor = 1 / jnp.sqrt(Key.size)
    q = q * scaling_factor

    return _flash_attention(
        (q, k, v),
        QPos,
        KPos,
        Key,
        mask,
        bias,
        dropout,
        inference=inference,
        key=key,
        block_size=block_size,
        precision=precision,
        logits_soft_cap=logits_soft_cap,
    )


@equinox.filter_custom_vjp
def _flash_attention(
    qkv: Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray],
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    mask: Optional[AttentionMask | hax.NamedArray] = None,
    bias: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: int,
    precision: PrecisionLike,
    logits_soft_cap: Optional[float],
) -> hax.NamedArray:
    return _flash_attention_forward(
        None,
        qkv,
        QPos,
        KPos,
        Key,
        mask,
        bias,
        dropout,
        inference=inference,
        key=key,
        block_size=block_size,
        precision=precision,
        logits_soft_cap=logits_soft_cap,
    )[0]


@named_call
def _flash_attention_forward(
    ignore,
    qkv,
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.AxisSelector,
    mask: Optional[AttentionMask | hax.NamedArray],
    bias: Optional[hax.NamedArray],
    dropout: float,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray],
    block_size: int,
    precision: PrecisionLike,
    logits_soft_cap: Optional[float],
):
    del ignore
    q, k, v = qkv
    if QPos.size % block_size != 0:
        raise ValueError(f"q axis size {q.axis_size(QPos)} is not a multiple of {block_size}")
    if KPos.size % block_size != 0:
        raise ValueError(f"k axis size {k.axis_size(KPos)} is not a multiple of {block_size}")

    # number of blocks for Q and K
    Tr = QPos.size // block_size
    Tc = KPos.size // block_size

    q_batch_axes: Tuple[hax.Axis, ...] = hax.eliminate_axes(q.axes, (QPos, Key))

    # output variables: O is the attention output, ell is the per-position log normalizer
    o_shape = _infer_attention_output_block_shape(QPos, KPos, Key, q, k, v)
    o = hax.zeros(o_shape, q.dtype)
    o = hax.auto_sharded(o)
    ell = hax.zeros((*q_batch_axes, QPos))
    ell = hax.auto_sharded(ell)

    is_causal = isinstance(mask, AttentionMask) and mask.is_causal
    # Non-zero offsets (dynamic or otherwise) are not supported in this implementation
    if isinstance(mask, AttentionMask) and mask.is_causal and mask.causal_offset is not None:
        raise NotImplementedError("Non-zero causal offsets are not implemented for flash attention")

    # i refers to the block index in Q, j refers to the block index in K.

    @named_call
    def do_o_block(state):
        i, o, ell = state

        # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
        q_i = q[QPos, ds.block(i, block_size)]

        # Step 2: init O_i = 0, sumexp_i = 0, max_i = -inf
        o_i = o[QPos, ds.block(i, block_size)]
        QPosBlock = QPos.resize(block_size)
        sumexp_i = hax.zeros(q_batch_axes + (QPosBlock,), q.dtype)
        max_i = hax.full(q_batch_axes + (QPosBlock,), -jnp.inf, q.dtype)

        @named_call
        def do_qk_block(state):
            """computes softmax(Q_i K_j^T) V_j"""
            # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
            #         K and V into 𝑇𝑐 = \ceil(𝑁/Bc) blocks of size Bc x d each.
            i, j, o_i, q_i, sumexp_i, old_max_i = state
            k_j = k[KPos, ds.block(j, block_size)]
            v_j = v[KPos, ds.block(j, block_size)]

            # Step 8: compute Sij = QiKj^T
            attn_ij = hax.dot(q_i, k_j, precision=precision, axis=Key)

            if bias is not None:
                if bias.has_axis(QPos.name):
                    bias_i = bias[QPos, ds.block(i, block_size)]
                else:
                    bias_i = bias

                if bias.has_axis(KPos.name):
                    bias_ij = bias_i[KPos, ds.block(j, block_size)]
                else:
                    bias_ij = bias_i

                attn_ij = attn_ij + bias_ij

            if logits_soft_cap is not None:
                attn_ij = hax.tanh(attn_ij / logits_soft_cap) * logits_soft_cap

            if mask is not None:
                mask_ij = _materialize_mask_slice(mask, i, j, QPos, KPos, block_size)
                attn_ij = hax.where(mask_ij, attn_ij, -1e10)

            if dropout > 0 and not inference:
                attn_ij = hax.nn.dropout(attn_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            # Step 9: Compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j)), P_i^j = exp(S_i^j - m_i^j),
            # ...    l_i^j = exp(m_i^{j-1} - m_i^j) + rowsum(P_i^j)
            max_i = hax.maximum(old_max_i, hax.max(attn_ij, axis=KPos.name))
            P_ij = hax.exp(attn_ij - max_i)

            exp_diff = hax.exp(old_max_i - max_i)
            sumexp_i = exp_diff * sumexp_i + hax.sum(P_ij, axis=KPos.name)

            # Step 10: Compute O_i = diag(exp(m_i^{j-1} - m_i^j) O_i + P_i^j V_j
            o_i = exp_diff * o_i + hax.dot(P_ij, v_j, axis=KPos.name)

            return (i, j + 1, o_i, q_i, sumexp_i, max_i)

        # For causal (offset==0), limit key blocks up to the diagonal
        j_end = jnp.minimum(i + 1, Tc) if is_causal else Tc

        _, _, o_i, _, sumexp_i, max_i = jax.lax.while_loop(
            lambda state: state[1] < j_end, do_qk_block, (i, 0, o_i, q_i, sumexp_i, max_i)
        )

        # Step 12: compute O_i = diag(\ell_i^{Tc})^{-1} O_i^{Tc}
        o_i = o_i / sumexp_i
        # Step 13: compute L_i = m_i^{Tc} + log(\ell_i^{Tc})
        ell_i = max_i + hax.log(sumexp_i)

        o = o.updated_slice({QPos: i * block_size}, o_i)
        ell = ell.updated_slice({QPos: i * block_size}, ell_i.astype(ell.dtype))

        return i + 1, o, ell

    _, o, ell = jax.lax.while_loop(lambda state: state[0] < Tr, do_o_block, (0, o, ell))

    return o, (o, ell)


@named_call
def _flash_attention_backward(
    residuals,
    grad_in: hax.NamedArray,
    ignore,
    qkv,
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.AxisSelector,
    mask: Optional[AttentionMask | hax.NamedArray] = None,
    bias: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
    block_size: int,
    precision: PrecisionLike,
    logits_soft_cap: Optional[float] = None,
):
    del ignore
    O, L = residuals
    q, k, v = qkv
    dO = grad_in

    Tr = QPos.size // block_size
    Tc = KPos.size // block_size

    if isinstance(mask, hax.NamedArray):
        mask = mask.broadcast_axis((QPos, KPos))  # make sure mask is broadcastable

    # Compute D = rowsum(dO * O), write D to HBM and divide it into Tr blocks of size Br each.
    # in the FA2 paper D is said to be \in R^{d}, but that doesn't make sense.
    # Triton impl has it as R^{QPos}, which makes more sense.
    D = hax.sum(dO * O, axis=Key)

    dQ = (q * 0.0).astype(q.dtype)
    dK = (k * 0.0).astype(k.dtype)
    dV = (v * 0.0).astype(v.dtype)

    is_causal = isinstance(mask, AttentionMask) and mask.is_causal

    @named_call
    def do_kv_block(state):
        j, dQ, dK, dV = state
        k_j = k[KPos, ds.block(j, block_size)]
        v_j = v[KPos, ds.block(j, block_size)]

        dK_j = dK[KPos, ds.block(j, block_size)]
        dV_j = dV[KPos, ds.block(j, block_size)]

        @named_call
        def do_inner_block(state):
            i, j, dQ, dK_j, dV_j = state
            q_i = q[QPos, ds.block(i, block_size)]

            dQ_i = dQ[QPos, ds.block(i, block_size)]
            dO_i = dO[QPos, ds.block(i, block_size)]
            L_i = L[QPos, ds.block(i, block_size)]
            D_i = D[QPos, ds.block(i, block_size)]

            attn_ij = hax.dot(q_i, k_j, precision=precision, axis=Key)

            if dropout > 0 and not inference:
                attn_ij = hax.nn.dropout(attn_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            if bias is not None:
                bias_ij = bias[QPos, ds.block(i, block_size), KPos, ds.block(j, block_size)]
                attn_ij = attn_ij + bias_ij

            if logits_soft_cap is not None:
                attn_ij = hax.tanh(attn_ij / logits_soft_cap) * logits_soft_cap

            if mask is not None:
                mask_ij = _materialize_mask_slice(mask, i, j, QPos, KPos, block_size)
                attn_ij = hax.where(mask_ij, attn_ij, -1e10)

            p_ij = hax.exp(attn_ij - L_i)

            if dropout > 0 and not inference:
                p_ij = hax.nn.dropout(p_ij, dropout, inference=False, key=jax.random.fold_in(key, i * Tc + j))

            dP_ij = hax.dot(dO_i, v_j, axis=Key)
            dAttn_ij = p_ij * (dP_ij - D_i)
            dAttn_ij = dAttn_ij.astype(dQ_i.dtype)

            dV_ji = hax.dot(p_ij, dO_i, axis=QPos.name).astype(dV_j.dtype)
            dK_ji = hax.dot(dAttn_ij, q_i, axis=QPos.name).astype(dK_j.dtype)

            # GQA-specific: eliminate unnecessary axes (e.g. 'q_heads_per_group')
            unnecessary_axes = hax.eliminate_axes(dV_ji.axes, _strip_sizes(v.axes))
            dV_ji = hax.sum(dV_ji, unnecessary_axes)
            dK_ji = hax.sum(dK_ji, unnecessary_axes)

            dV_j = dV_j + dV_ji
            dK_j = dK_j + dK_ji

            dQ_i = dQ_i + hax.dot(dAttn_ij, k_j, axis=KPos.name).astype(dQ.dtype)
            # dQ[i*block_size:(i+1)*block_size] = dQi
            dQ = dQ.updated_slice({QPos: i * block_size}, dQ_i)

            return i + 1, j, dQ, dK_j, dV_j

        # dQ, dK_j, dV_j = hax.fold(do_inner_block, Tr)((dQ, dK_j, dV_j), jnp.arange(Tr.size))
        i_start = j if is_causal else 0
        i, j, dQ, dK_j, dV_j = jax.lax.while_loop(
            lambda state: state[0] < Tr, do_inner_block, (i_start, j, dQ, dK_j, dV_j)
        )

        dK = dK.updated_slice({KPos: j * block_size}, dK_j)
        dV = dV.updated_slice({KPos: j * block_size}, dV_j)

        return j + 1, dQ, dK, dV

    # dQ, (dK, dV) = hax.scan(do_kv_block, Tc)(dQ, jnp.arange(Tc.size))
    j, dQ, dK, dV = jax.lax.while_loop(lambda state: state[0] < Tc, do_kv_block, (0, dQ, dK, dV))
    return dQ.rearrange(q.axes), dK.rearrange(k.axes), dV.rearrange(v.axes)


_flash_attention.def_fwd(_flash_attention_forward)
_flash_attention.def_bwd(_flash_attention_backward)


def _infer_attention_output_block_shape(QPos, KPos, Key, q_i, k, v):
    out_shape = filter_eval_shape(hnn.attention.dot_product_attention, KPos, Key, q_i, k, v)
    return out_shape.axes


def _materialize_mask_slice(mask, i, j, QPos, KPos, block_size):
    return materialize_mask(mask, QPos, KPos, q_slice=hax.ds.block(i, block_size), k_slice=hax.ds.block(j, block_size))


def _strip_sizes(axes: AxisSpec) -> AxisSelection:
    """Strip sizes from axes, returning only the names."""
    if isinstance(axes, hax.Axis):
        return axes.name
    return tuple(axis.name for axis in axes)
