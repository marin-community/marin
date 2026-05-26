# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# KL SOAP H: SHARDING-AWARE block-wise SOAP-style Hessian-eigenbasis preconditioner
# with Adam moments in the eigenbasis. Adapts the upstream KLSOAPH from
# KellerJordan/modded-nanogpt PR #290 to MoE-scale weight matrices.
#
# Design points (vs the earlier "replicate everything" variant):
#
#  - Pad each (rows, cols) weight to multiples of ``B * axis_size`` along
#    whichever mesh axis shards it (``data`` and/or ``model``). Padding adds
#    a few percent of bytes but ensures every per-chip slice is a contiguous
#    integer number of B-blocks.
#  - Reshape (rows, cols) -> (R, B, C, B) -> (R, C, B, B) using
#    ``jax.lax.reshape`` with an explicit ``sharding`` hint that places the
#    original (rows-axis, cols-axis) onto (R, C) and leaves the trailing two
#    B-axes replicated. With block-aligned padding the reshape is a pure
#    per-chip local reshape — no cross-shard splitting, no all-gather.
#  - All SOAP state tensors live in this same block-form sharding
#    ``(*leading, R, C, B, B)`` with the (R, C) axes inheriting the
#    parameter's sharding and (B, B) replicated. eigh/qr/einsum then run
#    per-block locally on each chip.
#  - The full-shape direction is reassembled (sharding-preserving inverse
#    of the block reshape) before return so the downstream hyperball
#    post-step in ``scale_with_grug_klsoaph`` normalizes the original
#    (unblocked) parameter and update.
#
# Per-block algorithm follows upstream KLSOAPH bit-for-bit:
#  1. First call: initialize Q from descending-eigh of (g g.T / cols,
#     g.T g / rows) computed on the first gradient; esi = init_factor**-0.5.
#     First call returns zero direction.
#  2. Subsequent calls: project gradient via current Q, Adam in projected
#     basis, build direction.
#  3. Whitened Gram update: gg_l += (g @ q_r * esi_r)^T @ (g @ q_r * esi_r) / cols;
#     symmetric counterpart for gg_r. Update esi from projected-gradient diagonals.
#  4. Refresh every ``precond_freq`` steps: warm-started QR iteration
#     (``qr(GG @ Q)``) and reproject exp_avg through old->new basis.

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map

_DEFAULT_BLOCK_SIZE: int = 128


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------


def _axis_size(spec_entry, mesh) -> int:
    """Return the mesh-axis size for a PartitionSpec entry, or 1 if None / unsharded."""
    if spec_entry is None or mesh is None:
        return 1
    # Spec entries may be strings or tuples of strings (for axis groups).
    if isinstance(spec_entry, tuple):
        size = 1
        for a in spec_entry:
            size *= int(mesh.shape.get(a, 1)) if a is not None else 1
        return size
    return int(mesh.shape.get(spec_entry, 1))


def _effective_spec(spec, mesh) -> tuple:
    """Return a spec with mesh-axes of size 1 replaced by None.

    Sharding a tensor axis along a mesh axis of size 1 is a no-op
    physically, but JAX's einsum/dot_general can still flag the spec
    as 'sharded' which leads to false-positive ShardingTypeError when
    that axis later becomes a contracting dim. Strip those down-front.
    """
    if mesh is None:
        return spec
    return tuple(s if _axis_size(s, mesh) > 1 else None for s in spec)


def _param_sharding(param) -> tuple[NamedSharding | None, tuple]:
    """Return (NamedSharding-or-None, spec-tuple) for a parameter.

    Works on both concrete arrays (``param.sharding``) and tracers
    (``param.aval.sharding`` — `.sharding` itself raises on tracers).
    Mesh axes of size 1 are normalized to None so they don't get treated
    as sharded contracting dims downstream.
    """
    s = None
    if hasattr(param, "aval") and hasattr(param.aval, "sharding"):
        s = param.aval.sharding
    if s is None:
        # Concrete arrays may have .sharding directly.
        try:
            s = param.sharding
        except (AttributeError, Exception):
            s = None
    if isinstance(s, NamedSharding) and s.mesh is not None and not s.mesh.empty:
        spec = tuple(s.spec)
        # Right-pad spec with Nones to match ndim (PartitionSpec may be shorter).
        spec = spec + (None,) * (param.ndim - len(spec))
        spec = _effective_spec(spec, s.mesh)
        return s, spec
    return None, (None,) * param.ndim


def _padded_inner_shape(rows: int, cols: int, B: int, rows_axis_size: int, cols_axis_size: int) -> tuple[int, int]:
    """Round (rows, cols) up to multiples of ``B * axis_size`` for each axis."""

    def _round(n, m):
        return ((n + m - 1) // m) * m

    rows_p = _round(rows, B * rows_axis_size)
    cols_p = _round(cols, B * cols_axis_size)
    return rows_p, cols_p


def _block_state_sharding(spec: tuple, mesh) -> NamedSharding | None:
    """Build a sharding for a (*leading, R, C, B, B) state tensor.

    The (R, C) axes inherit (rows_axis, cols_axis) from the original
    parameter; the trailing two B-axes are replicated (None).
    """
    if mesh is None:
        return None
    block_spec = P(*spec[:-2], spec[-2], spec[-1], None, None)
    return NamedSharding(mesh, block_spec)


def _block_esi_sharding(spec: tuple, mesh, side: str) -> NamedSharding | None:
    """Sharding for the (*leading, R, C, B) esi state."""
    if mesh is None:
        return None
    block_spec = P(*spec[:-2], spec[-2], spec[-1], None)
    return NamedSharding(mesh, block_spec)


# ---------------------------------------------------------------------------
# Block reshape (sharding-preserving)
# ---------------------------------------------------------------------------


def _pad_to(x: jnp.ndarray, rows_padded: int, cols_padded: int) -> jnp.ndarray:
    rows = x.shape[-2]
    cols = x.shape[-1]
    if rows == rows_padded and cols == cols_padded:
        return x
    pad_widths = [(0, 0)] * (x.ndim - 2) + [(0, rows_padded - rows), (0, cols_padded - cols)]
    return jnp.pad(x, pad_widths)


def _unpad_to(x: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
    if x.shape[-2] == rows and x.shape[-1] == cols:
        return x
    return x[..., :rows, :cols]


def _to_blocks(x: jnp.ndarray, B: int, spec: tuple, mesh) -> jnp.ndarray:
    """Reshape ``(*leading, rows, cols)`` → ``(*leading, R, C, B, B)`` preserving sharding.

    Caller must ensure ``rows`` and ``cols`` are multiples of
    ``B * axis_size`` for whichever mesh axis shards each, so that the
    reshape is a pure per-chip local reshape with no cross-shard splitting.
    """
    rows = x.shape[-2]
    cols = x.shape[-1]
    R = rows // B
    C = cols // B
    if mesh is None:
        x = x.reshape(*x.shape[:-2], R, B, C, B)
        return jnp.swapaxes(x, -3, -2)

    # Two-step reshape (one sharded axis at a time) with explicit out sharding.
    # Step 1: split cols.  P(*leading, rows_axis, cols_axis) -> P(*leading, rows_axis, cols_axis, None).
    step1_shape = (*x.shape[:-1], C, B)
    step1_spec = P(*spec[:-1], spec[-1], None)
    x = jax.lax.reshape(x, step1_shape, out_sharding=NamedSharding(mesh, step1_spec))
    # Step 2: split rows.  P(*leading, rows_axis, cols_axis, None) -> P(*leading, rows_axis, None, cols_axis, None).
    step2_shape = (*x.shape[:-3], R, B, C, B)
    step2_spec = P(*spec[:-2], spec[-2], None, spec[-1], None)
    x = jax.lax.reshape(x, step2_shape, out_sharding=NamedSharding(mesh, step2_spec))
    # Step 3: swap (-3, -2) to get (R, C, B, B).
    return jnp.swapaxes(x, -3, -2)


def _from_blocks(blocks: jnp.ndarray, original_shape: tuple, B: int, spec: tuple, mesh) -> jnp.ndarray:
    """Inverse of ``_to_blocks`` — back to ``(*leading, rows_padded, cols_padded)``.

    Sharding is restored to ``(*leading, rows_axis, cols_axis)``. Caller is
    responsible for unpadding back to the original (rows, cols).
    """
    # blocks: (*leading, R, C, B, B). swap to (*leading, R, B, C, B).
    x = jnp.swapaxes(blocks, -3, -2)
    R = x.shape[-4]
    C = x.shape[-2]
    rows_padded = R * B
    cols_padded = C * B
    if mesh is None:
        x = x.reshape(*x.shape[:-4], rows_padded, cols_padded)
        return x
    # Inverse two-step reshape with explicit sharding.
    # First merge (R, B) -> rows_padded.
    step1_shape = (*x.shape[:-4], rows_padded, C, B)
    step1_spec = P(*spec[:-2], spec[-2], spec[-1], None)
    x = jax.lax.reshape(x, step1_shape, out_sharding=NamedSharding(mesh, step1_spec))
    # Then merge (C, B) -> cols_padded.
    step2_shape = (*x.shape[:-2], cols_padded)
    step2_spec = P(*spec[:-2], spec[-2], spec[-1])
    x = jax.lax.reshape(x, step2_shape, out_sharding=NamedSharding(mesh, step2_spec))
    return x


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ScaleByKLSoapHState(NamedTuple):
    """Per-leaf block-wise SOAP state.

    Per matrix leaf with original shape ``(*leading, rows, cols)`` we keep:

      exp_avg, exp_avg_sq:    (*leading, R, C, B, B)
      gg_l, q_l:              (*leading, R, C, B, B)   -- block-local row-Gram / its eigenbasis
      gg_r, q_r:              (*leading, R, C, B, B)   -- block-local col-Gram / its eigenbasis
      esi_l, esi_r:           (*leading, R, C, B)

    where R = rows_padded / B and C = cols_padded / B.

    ``shapes`` and ``specs`` are static per-leaf metadata (original shape,
    padded shape, parameter sharding spec). They are stored as a side-pytree
    so that update_fn can re-pad new gradients to the same padded shape.
    """

    count: chex.Array
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    gg_l: optax.Updates
    gg_r: optax.Updates
    q_l: optax.Updates
    q_r: optax.Updates
    esi_l: optax.Updates
    esi_r: optax.Updates


class _LeafMeta(NamedTuple):
    """Static per-leaf metadata. Hashable, not a JAX array."""

    original_shape: tuple
    padded_shape: tuple
    spec: tuple
    has_mesh: bool


class _SoapStepResult:
    """Plain-Python wrapper for the nine outputs of one SOAP step.

    Not a registered pytree, so jax.tree.map treats it as a leaf.
    """

    __slots__ = ("direction", "esi_l", "esi_r", "exp_avg", "exp_avg_sq", "gg_l", "gg_r", "q_l", "q_r")

    def __init__(self, direction, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r):
        self.direction = direction
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.gg_l = gg_l
        self.gg_r = gg_r
        self.q_l = q_l
        self.q_r = q_r
        self.esi_l = esi_l
        self.esi_r = esi_r


def _is_matrix_param(p) -> bool:
    return hasattr(p, "ndim") and p.ndim >= 2


# ---------------------------------------------------------------------------
# Per-leaf state initialization (with sharded zero tensors)
# ---------------------------------------------------------------------------


def _zeros_with_sharding(shape: tuple, sharding: NamedSharding | None) -> jnp.ndarray:
    z = jnp.zeros(shape, dtype=jnp.float32)
    if sharding is not None:
        z = jax.sharding.reshard(z, sharding)
    return z


def _full_with_sharding(shape: tuple, fill: float, sharding: NamedSharding | None) -> jnp.ndarray:
    z = jnp.full(shape, fill, dtype=jnp.float32)
    if sharding is not None:
        z = jax.sharding.reshard(z, sharding)
    return z


def _eye_blocks_with_sharding(state_shape: tuple, B: int, sharding: NamedSharding | None) -> jnp.ndarray:
    """Broadcast a (B, B) identity over (*leading, R, C, B, B)."""
    eye = jnp.eye(B, dtype=jnp.float32)
    z = jnp.broadcast_to(eye, state_shape).astype(jnp.float32)
    if sharding is not None:
        # Explicit reshard (rather than with_sharding_constraint which asserts).
        z = jax.sharding.reshard(z, sharding)
    return z


def _init_state_for_leaf(param, B: int, init_factor: float) -> tuple[_LeafMeta, dict] | tuple[None, None]:
    """Allocate per-leaf state with sharding aligned to block structure."""
    if not _is_matrix_param(param):
        return None, None

    rows = param.shape[-2]
    cols = param.shape[-1]
    sharding, spec = _param_sharding(param)
    mesh = sharding.mesh if sharding is not None else None
    rows_axis_size = _axis_size(spec[-2], mesh)
    cols_axis_size = _axis_size(spec[-1], mesh)
    rows_p, cols_p = _padded_inner_shape(rows, cols, B, rows_axis_size, cols_axis_size)
    R = rows_p // B
    C = cols_p // B
    leading = param.shape[:-2]
    state_shape = (*leading, R, C, B, B)
    esi_shape = (*leading, R, C, B)

    block_sharding = _block_state_sharding(spec, mesh)
    esi_sharding = _block_esi_sharding(spec, mesh, side="l")

    state = dict(
        exp_avg=_zeros_with_sharding(state_shape, block_sharding),
        exp_avg_sq=_zeros_with_sharding(state_shape, block_sharding),
        gg_l=_zeros_with_sharding(state_shape, block_sharding),
        gg_r=_zeros_with_sharding(state_shape, block_sharding),
        q_l=_eye_blocks_with_sharding(state_shape, B, block_sharding),
        q_r=_eye_blocks_with_sharding(state_shape, B, block_sharding),
        esi_l=_full_with_sharding(esi_shape, init_factor**-0.5, esi_sharding),
        esi_r=_full_with_sharding(esi_shape, init_factor**-0.5, esi_sharding),
    )
    meta = _LeafMeta(
        original_shape=tuple(param.shape),
        padded_shape=(*leading, rows_p, cols_p),
        spec=spec,
        has_mesh=mesh is not None,
    )
    return meta, state


# ---------------------------------------------------------------------------
# Per-leaf SOAP step (each block independent, sharded across (E, R, C))
# ---------------------------------------------------------------------------


def _symmetrize(matrix):
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _klsoaph_step_local(
    g32_in: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    gg_l: jnp.ndarray,
    gg_r: jnp.ndarray,
    q_l: jnp.ndarray,
    q_r: jnp.ndarray,
    esi_l: jnp.ndarray,
    esi_r: jnp.ndarray,
    step: jnp.ndarray,
    beta1: float,
    beta2: float,
    shampoo_beta: float,
    eps: float,
    precond_freq: int,
    init_factor: float,
    block_size: int,
):
    """Single-chip body of the per-block SOAP step.

    Assumes all inputs are already per-chip slices (no cross-chip
    communication needed). Plain ``jnp.einsum`` / ``jnp.linalg.eigh`` /
    ``jnp.linalg.qr`` — no ``out_sharding=`` hints, no per-linalg
    ``shard_map`` wrappers. Called inside one outer ``shard_map`` so XLA
    can fuse einsums + eigh + qr in one local kernel pipeline.
    """
    B = block_size
    inner_rows = B
    inner_cols = B
    g32 = g32_in.astype(jnp.float32)

    def _flip(m):
        sym = 0.5 * (m + jnp.swapaxes(m, -1, -2))
        _, q = jnp.linalg.eigh(sym)
        return q[..., :, ::-1]

    def _update_esi(esi, diag):
        old_eigen = jnp.reciprocal(jnp.square(esi))
        old_eigen = jnp.nan_to_num(old_eigen, nan=0.0, posinf=0.0, neginf=0.0)
        eigen = shampoo_beta * old_eigen + (1.0 - shampoo_beta) * diag
        eigen = jnp.maximum(eigen, 1e-30)
        inv_sqrt = jax.lax.rsqrt(eigen)
        inv_sqrt = jnp.minimum(inv_sqrt, 4000.0)
        inv_sqrt = jnp.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        return inv_sqrt

    # We dispatch init vs normal via ``lax.cond`` rather than an arithmetic
    # blend. Inside a shard_map local body the predicate has no sharding
    # conflict, and lax.cond gives XLA the freedom to dead-code the
    # uncalled branch — saving the (very expensive on TPU) batched eigh on
    # every step after the first.

    def _init_branch(operands):
        g32, exp_avg, exp_avg_sq, _gg_l, _gg_r, _q_l, _q_r, esi_l, esi_r = operands
        init_gg_l = _symmetrize(jnp.einsum("...ik,...jk->...ij", g32, g32) / inner_cols)
        init_gg_r = _symmetrize(jnp.einsum("...ki,...kj->...ij", g32, g32) / inner_rows)
        init_q_l = _flip(init_gg_l)
        init_q_r = _flip(init_gg_r)
        init_esi_l = jnp.full_like(esi_l, init_factor**-0.5)
        init_esi_r = jnp.full_like(esi_r, init_factor**-0.5)
        return (
            jnp.zeros_like(g32),
            jnp.zeros_like(exp_avg),
            jnp.zeros_like(exp_avg_sq),
            init_gg_l,
            init_gg_r,
            init_q_l,
            init_q_r,
            init_esi_l,
            init_esi_r,
        )

    def _normal_branch(operands):
        g32, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r = operands

        # project, Adam-in-projected-basis, direction
        g_qr = jnp.einsum("...ij,...jk->...ik", g32, q_r)
        g_proj = jnp.einsum("...ki,...kj->...ij", q_l, g_qr)
        nb_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g_proj
        nb_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * g_proj * g_proj
        precond_proj = nb_exp_avg / (jnp.sqrt(nb_exp_avg_sq) + eps)
        p_qrt = jnp.einsum("...ij,...kj->...ik", precond_proj, q_r)
        direction = jnp.einsum("...ij,...jk->...ik", q_l, p_qrt)

        # whitened Gram update
        g_qr_white = g_qr * esi_r[..., None, :]
        left_target = jnp.einsum("...ik,...jk->...ij", g_qr_white, g_qr_white) / inner_cols
        # qlT_g = q_l.T @ g; since q_r is orthogonal (output of QR), we can derive
        # this from g_proj = q_l.T @ g @ q_r via g_proj @ q_r.T (saves 1 einsum/step).
        qlT_g = jnp.einsum("...ij,...kj->...ik", g_proj, q_r)
        qlT_g_white = qlT_g * esi_l[..., :, None]
        right_target = jnp.einsum("...ki,...kj->...ij", qlT_g_white, qlT_g_white) / inner_rows
        nb_gg_l = _symmetrize(shampoo_beta * gg_l + (1.0 - shampoo_beta) * left_target)
        nb_gg_r = _symmetrize(shampoo_beta * gg_r + (1.0 - shampoo_beta) * right_target)

        # ESI update
        proj_col_white = g_proj * esi_r[..., None, :]
        left_diag = jnp.mean(proj_col_white * proj_col_white, axis=-1)
        proj_row_white = g_proj * esi_l[..., :, None]
        right_diag = jnp.mean(proj_row_white * proj_row_white, axis=-2)
        nb_esi_l = _update_esi(esi_l, left_diag)
        nb_esi_r = _update_esi(esi_r, right_diag)

        # warm-started QR refresh (also via lax.cond on the freq predicate)
        def _do_refresh(_):
            ea_qrT = jnp.einsum("...ij,...kj->...ik", nb_exp_avg, q_r)
            ea_original = jnp.einsum("...ij,...jk->...ik", q_l, ea_qrT)
            gg_l_q = jnp.einsum("...ij,...jk->...ik", nb_gg_l, q_l)
            gg_r_q = jnp.einsum("...ij,...jk->...ik", nb_gg_r, q_r)
            ql_refresh, _ = jnp.linalg.qr(gg_l_q)
            qr_refresh, _ = jnp.linalg.qr(gg_r_q)
            ea_qr_new = jnp.einsum("...ij,...jk->...ik", ea_original, qr_refresh)
            ea_refresh = jnp.einsum("...ki,...kj->...ij", ql_refresh, ea_qr_new)
            return ql_refresh, qr_refresh, ea_refresh

        def _no_refresh(_):
            return q_l, q_r, nb_exp_avg

        should_refresh = jnp.equal(jnp.remainder(step, precond_freq), 0)
        new_q_l, new_q_r, new_exp_avg_out = jax.lax.cond(
            should_refresh, _do_refresh, _no_refresh, operand=None
        )

        return (
            direction,
            new_exp_avg_out,
            nb_exp_avg_sq,
            nb_gg_l,
            nb_gg_r,
            new_q_l,
            new_q_r,
            nb_esi_l,
            nb_esi_r,
        )

    operands = (g32, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r)
    is_first = jnp.equal(step, 1)
    return jax.lax.cond(is_first, _init_branch, _normal_branch, operands)


def _klsoaph_step_blocked(
    grad_blocks: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    gg_l: jnp.ndarray,
    gg_r: jnp.ndarray,
    q_l: jnp.ndarray,
    q_r: jnp.ndarray,
    esi_l: jnp.ndarray,
    esi_r: jnp.ndarray,
    step: jnp.ndarray,
    beta1: float,
    beta2: float,
    shampoo_beta: float,
    eps: float,
    precond_freq: int,
    init_factor: float,
    block_size: int,
    mesh,
    block_spec,
    esi_spec,
):
    """One per-leaf SOAP step wrapped in a single outer ``shard_map``.

    State tensors are all block-form with the (R, C) axes carrying the
    parameter's sharding and the trailing (B, B) (or (B,) for esi) axes
    replicated. We dispatch the entire per-block step inside one
    ``shard_map`` so XLA sees a single local pipeline (einsum + eigh +
    qr fused per chip), instead of paying per-linalg shard_map dispatch
    overhead. After step-1 (compile) the per-step cost should match a
    plain per-chip JIT.
    """
    if mesh is None or mesh.empty:
        return _klsoaph_step_local(
            grad_blocks,
            exp_avg,
            exp_avg_sq,
            gg_l,
            gg_r,
            q_l,
            q_r,
            esi_l,
            esi_r,
            step,
            beta1=beta1,
            beta2=beta2,
            shampoo_beta=shampoo_beta,
            eps=eps,
            precond_freq=precond_freq,
            init_factor=init_factor,
            block_size=block_size,
        )

    def _local(g, ea, eas, ggl, ggr, ql, qr, esil, esir, st):
        return _klsoaph_step_local(
            g,
            ea,
            eas,
            ggl,
            ggr,
            ql,
            qr,
            esil,
            esir,
            st,
            beta1=beta1,
            beta2=beta2,
            shampoo_beta=shampoo_beta,
            eps=eps,
            precond_freq=precond_freq,
            init_factor=init_factor,
            block_size=block_size,
        )

    in_specs = (
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        esi_spec,
        esi_spec,
        P(),
    )
    out_specs = (
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        block_spec,
        esi_spec,
        esi_spec,
    )

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(grad_blocks, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r, step)


# ---------------------------------------------------------------------------
# Public transform
# ---------------------------------------------------------------------------


def scale_by_klsoaph(
    beta1: float = 0.95,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 5,
    init_factor: float = 0.1,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> optax.GradientTransformation:
    """Sharding-aware block-wise SOAP preconditioner.

    State allocation:
      - For each matrix leaf (ndim >= 2): pads (rows, cols) up to multiples
        of ``B * mesh_axis_size`` along whichever mesh axis shards each.
        Allocates state in block form ``(*leading, R, C, B, B)`` with the
        (R, C) axes carrying the parameter's sharding and (B, B) axes
        replicated.
      - For non-matrix leaves: no state.

    Per step:
      - Pad gradient to the per-leaf padded shape.
      - Reshape to block form with explicit ``jax.lax.reshape`` sharding
        hint (two-step split, one sharded axis at a time).
      - Run per-block SOAP; einsum/eigh/qr contract only on the replicated
        (B, B) trailing axes — fully local per chip.
      - Reshape direction back to padded shape, unpad to original.

    Returns full-shape direction (per-leaf) so the downstream
    ``_scale_invariant_hyperball_updates`` post-step normalizes the
    original (unblocked) parameter and update.
    """
    B = block_size

    def init_fn(params):
        # Allocate per-leaf state. Padded shape and sharding are derived from
        # each param's shape + sharding spec; update_fn re-derives the same
        # metadata from the params it receives (cheap, no closure needed).
        flat_params = jax.tree_util.tree_leaves(params)
        treedef = jax.tree_util.tree_structure(params)
        per_leaf_state = [_init_state_for_leaf(p, B, init_factor)[1] for p in flat_params]

        def gather(field):
            return jax.tree_util.tree_unflatten(treedef, [None if s is None else s[field] for s in per_leaf_state])

        return ScaleByKLSoapHState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=gather("exp_avg"),
            exp_avg_sq=gather("exp_avg_sq"),
            gg_l=gather("gg_l"),
            gg_r=gather("gg_r"),
            q_l=gather("q_l"),
            q_r=gather("q_r"),
            esi_l=gather("esi_l"),
            esi_r=gather("esi_r"),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_by_klsoaph requires params= so it can re-derive per-leaf sharding meta")
        next_count = optax.safe_increment(state.count)

        flat_params = jax.tree_util.tree_leaves(params)
        flat_updates = jax.tree_util.tree_leaves(updates)
        flat_state = {
            "exp_avg": jax.tree_util.tree_leaves(state.exp_avg),
            "exp_avg_sq": jax.tree_util.tree_leaves(state.exp_avg_sq),
            "gg_l": jax.tree_util.tree_leaves(state.gg_l),
            "gg_r": jax.tree_util.tree_leaves(state.gg_r),
            "q_l": jax.tree_util.tree_leaves(state.q_l),
            "q_r": jax.tree_util.tree_leaves(state.q_r),
            "esi_l": jax.tree_util.tree_leaves(state.esi_l),
            "esi_r": jax.tree_util.tree_leaves(state.esi_r),
        }
        treedef = jax.tree_util.tree_structure(updates)

        out_buckets = {k: [] for k in ("dir", *flat_state.keys())}
        for i, (grad, param) in enumerate(zip(flat_updates, flat_params, strict=True)):
            cur_state = {k: v[i] for k, v in flat_state.items()}
            if grad is None or not _is_matrix_param(param):
                out_buckets["dir"].append(grad)
                for k, v in cur_state.items():
                    out_buckets[k].append(v)
                continue

            rows = param.shape[-2]
            cols = param.shape[-1]
            sharding, spec = _param_sharding(param)
            mesh = sharding.mesh if sharding is not None else None
            mesh = mesh if (mesh is not None and not mesh.empty) else None
            rows_axis_size = _axis_size(spec[-2], mesh)
            cols_axis_size = _axis_size(spec[-1], mesh)
            rows_p, cols_p = _padded_inner_shape(rows, cols, B, rows_axis_size, cols_axis_size)
            padded_shape = (*param.shape[:-2], rows_p, cols_p)

            padded = _pad_to(grad, rows_p, cols_p)
            grad_blocks = _to_blocks(padded, B, spec, mesh)
            block_sharding = _block_state_sharding(spec, mesh)
            esi_sharding = _block_esi_sharding(spec, mesh, side="l")
            block_spec_ = block_sharding.spec if block_sharding is not None else None
            esi_spec_ = esi_sharding.spec if esi_sharding is not None else None

            out = _klsoaph_step_blocked(
                grad_blocks,
                cur_state["exp_avg"],
                cur_state["exp_avg_sq"],
                cur_state["gg_l"],
                cur_state["gg_r"],
                cur_state["q_l"],
                cur_state["q_r"],
                cur_state["esi_l"],
                cur_state["esi_r"],
                next_count,
                beta1=beta1,
                beta2=beta2,
                shampoo_beta=shampoo_beta,
                eps=eps,
                precond_freq=precond_freq,
                init_factor=init_factor,
                block_size=B,
                mesh=mesh,
                block_spec=block_spec_,
                esi_spec=esi_spec_,
            )
            direction_padded = _from_blocks(out[0], padded_shape, B, spec, mesh)
            direction = _unpad_to(direction_padded, rows, cols).astype(grad.dtype)

            out_buckets["dir"].append(direction)
            out_buckets["exp_avg"].append(out[1])
            out_buckets["exp_avg_sq"].append(out[2])
            out_buckets["gg_l"].append(out[3])
            out_buckets["gg_r"].append(out[4])
            out_buckets["q_l"].append(out[5])
            out_buckets["q_r"].append(out[6])
            out_buckets["esi_l"].append(out[7])
            out_buckets["esi_r"].append(out[8])

        new_state = ScaleByKLSoapHState(
            count=next_count,
            exp_avg=jax.tree_util.tree_unflatten(treedef, out_buckets["exp_avg"]),
            exp_avg_sq=jax.tree_util.tree_unflatten(treedef, out_buckets["exp_avg_sq"]),
            gg_l=jax.tree_util.tree_unflatten(treedef, out_buckets["gg_l"]),
            gg_r=jax.tree_util.tree_unflatten(treedef, out_buckets["gg_r"]),
            q_l=jax.tree_util.tree_unflatten(treedef, out_buckets["q_l"]),
            q_r=jax.tree_util.tree_unflatten(treedef, out_buckets["q_r"]),
            esi_l=jax.tree_util.tree_unflatten(treedef, out_buckets["esi_l"]),
            esi_r=jax.tree_util.tree_unflatten(treedef, out_buckets["esi_r"]),
        )
        directions_tree = jax.tree_util.tree_unflatten(treedef, out_buckets["dir"])
        return directions_tree, new_state

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByKLSoapHState", "scale_by_klsoaph"]
