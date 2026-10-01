# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH variants that work in the eigenbasis of each matrix's Kronecker gradient factors.

Per matrix ``W`` (``[..., m, n]``; leading axes are independent matrices, such as the layer or expert stack) the
transform keeps Shampoo's factors ``L = EMA(G Gᵀ)`` and ``R = EMA(Gᵀ G)``, their eigenvectors ``Q_L, Q_R``
(recomputed every ``refresh_every`` steps, as in SOAP), and EMAs of the rotated gradient ``G' = Q_Lᵀ G Q_R`` and of
its square. When the basis changes, the first moment is rotated into the new basis and the second is kept (SOAP's
approximation). The direction ``D`` depends on ``mode``; the optimizer then takes MuonH's Frobenius hyperball step
with it (``GrugMoeMuonHConfig.eig_families``).

- ``muon``: ``NS(M)`` with ``M = Q_L m' Q_Rᵀ`` the plain-EMA momentum. The control for the plumbing.
- ``snr``: ``Q_L (m' / sqrt(v')) Q_Rᵀ`` with the same β in both moments, so every rotated coordinate's weight is its
  signal-to-noise ratio, at most 1 (ANVIL III's per-direction SNR, Hyperstition 2026). For a constant gradient
  this is exactly Muon's polar factor.
- ``soap``: the same with a slower second moment (``beta2``): Adam in Shampoo's eigenbasis (SOAP, arXiv 2409.11321).
- ``muon_weighted``: Muon's ``NS(M)``, each rotated coordinate scaled by ``|m'| / sqrt(v')``: Muon's geometry with
  inconsistent coordinates muted.
- ``soap_muon``: ``NS(Q_L (m' / sqrt(v')) Q_Rᵀ)`` with the slower SOAP second moment (``beta2``): SOAP's direction
  orthogonalized by Newton-Schulz (SOAP-Muon, the modded-nanogpt optimization track's largest single gain).
- ``prewhiten``: ``NS(L^{-p} M)``, Shampoo's left (input-side) factor applied before Newton-Schulz. MuonEq's
  per-input-row normalization is the diagonal, history-free version of this.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec, reshard

from experiments.grug.fast_track.grugmuon_stacked import _target_named_sharding
from experiments.grug.fast_track.stiefel import _msign

EIG_MODES = ("muon", "snr", "soap", "muon_weighted", "prewhiten", "soap_muon")
_EPS = 1e-30
_RELATIVE_EPS = 1e-6


class EigMuonState(NamedTuple):
    count: jax.Array
    left: optax.Updates
    right: optax.Updates
    q_left: optax.Updates
    q_right: optax.Updates
    first: optax.Updates
    second: optax.Updates
    left_eigvals: optax.Updates


class _LeafResult(NamedTuple):
    direction: jax.Array
    left: jax.Array
    right: jax.Array
    q_left: jax.Array
    q_right: jax.Array
    first: jax.Array
    second: jax.Array
    left_eigvals: jax.Array


def _is_matrix(x) -> bool:
    return hasattr(x, "ndim") and x.ndim >= 2


def _whole(x: jax.Array, like) -> jax.Array:
    """``x`` with each matrix whole on one device (stacked leading axes keep ``like``'s sharding)."""
    target = _target_named_sharding(like)
    if target is None:
        return x
    spec = tuple(target.spec) + (None,) * (like.ndim - len(target.spec))
    lead = spec[: like.ndim - 2]
    return reshard(x, PartitionSpec(*lead, *(None,) * (x.ndim - len(lead))))


def _swap(x: jax.Array) -> jax.Array:
    return jnp.swapaxes(x, -1, -2)


def _eigh_basis(factor: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Eigenvalues and eigenvectors, largest first, of a symmetric PSD factor (damped so ``eigh`` stays stable)."""
    d = factor.shape[-1]
    scale = jnp.trace(factor, axis1=-2, axis2=-1)[..., None, None] / d
    vals, vecs = jnp.linalg.eigh(factor + 1e-6 * scale * jnp.eye(d, dtype=factor.dtype) + _EPS)
    return vals[..., ::-1], vecs[..., ::-1]


def scale_by_eig_direction(
    mode: str,
    beta: float = 0.95,
    beta2: float = 0.99,
    factor_beta: float = 0.95,
    refresh_every: int = 10,
    whiten_power: float = 0.25,
    ns_steps: int = 5,
) -> optax.GradientTransformation:
    """The eigenbasis direction for each matrix leaf (see the module docstring); other leaves pass through."""
    if mode not in EIG_MODES:
        raise ValueError(f"mode must be one of {EIG_MODES}, got {mode!r}")
    second_beta = beta2 if mode in ("soap", "soap_muon") else beta

    def init_fn(params):
        def zeros_like_matrix(p, rows: int, cols: int):
            if not _is_matrix(p):
                return None
            return _whole(jnp.zeros((*p.shape[:-2], rows, cols), jnp.float32), p)

        def eye(p, d: int):
            if not _is_matrix(p):
                return None
            return _whole(jnp.broadcast_to(jnp.eye(d, dtype=jnp.float32), (*p.shape[:-2], d, d)), p)

        def eigvals(p):
            if not _is_matrix(p):
                return None
            return _whole(jnp.ones((*p.shape[:-2], p.shape[-2]), jnp.float32), p)

        return EigMuonState(
            count=jnp.zeros([], jnp.int32),
            left=jax.tree.map(
                lambda p: zeros_like_matrix(p, p.shape[-2], p.shape[-2]) if _is_matrix(p) else None, params
            ),
            right=jax.tree.map(
                lambda p: zeros_like_matrix(p, p.shape[-1], p.shape[-1]) if _is_matrix(p) else None, params
            ),
            q_left=jax.tree.map(lambda p: eye(p, p.shape[-2]) if _is_matrix(p) else None, params),
            q_right=jax.tree.map(lambda p: eye(p, p.shape[-1]) if _is_matrix(p) else None, params),
            first=jax.tree.map(lambda p: zeros_like_matrix(p, *p.shape[-2:]) if _is_matrix(p) else None, params),
            second=jax.tree.map(lambda p: zeros_like_matrix(p, *p.shape[-2:]) if _is_matrix(p) else None, params),
            left_eigvals=jax.tree.map(eigvals, params),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_by_eig_direction requires params (for each matrix's sharding)")
        count = state.count + 1
        refresh = (count == 1) | (count % refresh_every == 0)
        t = count.astype(jnp.float32)
        first_corr = 1.0 - beta**t
        second_corr = 1.0 - second_beta**t

        def leaf(g, p, left, right, q_left, q_right, first, second, left_vals):
            if g is None or not _is_matrix(g):
                return _LeafResult(g, left, right, q_left, q_right, first, second, left_vals)
            g32 = _whole(g.astype(jnp.float32), p)
            left = factor_beta * left + (1.0 - factor_beta) * (g32 @ _swap(g32))
            right = factor_beta * right + (1.0 - factor_beta) * (_swap(g32) @ g32)

            def new_basis(args):
                left, right, q_left, q_right, first, _ = args
                vals_l, new_l = _eigh_basis(left)
                _, new_r = _eigh_basis(right)
                # Carry the first moment into the new basis (an exact change of coordinates).
                first = _swap(new_l) @ q_left @ first @ _swap(q_right) @ new_r
                return new_l, new_r, first, vals_l

            def same_basis(args):
                _, _, q_left, q_right, first, vals_l = args
                return q_left, q_right, first, vals_l

            q_left, q_right, first, left_vals = jax.lax.cond(
                refresh, new_basis, same_basis, (left, right, q_left, q_right, first, left_vals)
            )
            rotated = _swap(q_left) @ g32 @ q_right
            first = beta * first + (1.0 - beta) * rotated
            second = second_beta * second + (1.0 - second_beta) * jnp.square(rotated)
            m_hat = first / first_corr
            v_hat = second / second_corr
            # A scale-free Adam epsilon: coordinates far below the matrix's typical size (round-off in the rotated
            # gradient, for one) must not reach an SNR of 1 just because their sign is steady.
            floor = _RELATIVE_EPS * jnp.mean(v_hat, axis=(-2, -1), keepdims=True)
            snr = m_hat / jnp.sqrt(v_hat + floor + _EPS)
            if mode in ("snr", "soap"):
                direction = q_left @ snr @ _swap(q_right)
            elif mode == "soap_muon":
                direction = _msign(q_left @ snr @ _swap(q_right), ns_steps)
            else:
                if mode == "prewhiten":
                    scale = left_vals / jnp.mean(left_vals, axis=-1, keepdims=True)
                    whitened = jnp.maximum(scale, 1e-4) ** (-whiten_power)
                    momentum = q_left @ (whitened[..., :, None] * m_hat) @ _swap(q_right)
                else:
                    momentum = q_left @ m_hat @ _swap(q_right)
                direction = _msign(momentum, ns_steps)
                if mode == "muon_weighted":
                    weights = jnp.abs(snr)
                    direction = q_left @ (weights * (_swap(q_left) @ direction @ q_right)) @ _swap(q_right)
            target = _target_named_sharding(p)
            if target is not None:
                direction = reshard(direction, target.spec)
            return _LeafResult(direction.astype(g.dtype), left, right, q_left, q_right, first, second, left_vals)

        flat = jax.tree.map(
            leaf,
            updates,
            params,
            state.left,
            state.right,
            state.q_left,
            state.q_right,
            state.first,
            state.second,
            state.left_eigvals,
            is_leaf=lambda x: x is None,
        )

        def part(i):
            return jax.tree.map(lambda r: r[i], flat, is_leaf=lambda x: isinstance(x, _LeafResult))

        new_state = EigMuonState(count, part(1), part(2), part(3), part(4), part(5), part(6), part(7))
        return part(0), new_state

    return optax.GradientTransformation(init_fn, update_fn)
