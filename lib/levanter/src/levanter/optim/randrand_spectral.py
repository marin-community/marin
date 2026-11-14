# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Two-bank sketch-SVD RandRAND, filtering before and after Adam.

The two probe banks each track H @ Omega for raw-gradient covariance H.
SVD of that sketch (not eigendecomposition of H) supplies the filter directions.
A warm standby replaces the active bank periodically; only the retired bank is
redrawn and cleared. Adam moments persist. No full Gram matrices are allocated.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from levanter.optim import randrand_core as rr


@dataclass(frozen=True)
class SpectralSettings:
    probe_every: int = 5
    regularization: float = 1e-6

    def __post_init__(self):
        if isinstance(self.probe_every, bool) or not isinstance(self.probe_every, int) or self.probe_every < 1:
            raise ValueError("probe_every must be a positive integer")
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")


class SketchBank(NamedTuple):
    left_probe: jax.Array
    right_probe: jax.Array
    left_ema: jax.Array
    right_ema: jax.Array


class SpectralMatrixState(NamedTuple):
    active: SketchBank
    standby: SketchBank
    left_basis: jax.Array
    right_basis: jax.Array
    left_values: jax.Array
    right_values: jax.Array
    refresh_count: jax.Array
    enabled: jax.Array


def draw_bank(key, rows, columns, width):
    """Independent orthonormal column probes, with matching zero sketches."""
    left_key, right_key = jax.random.split(key)
    left, _ = jnp.linalg.qr(jax.random.normal(left_key, (rows, width), dtype=jnp.float32), mode="reduced")
    right, _ = jnp.linalg.qr(jax.random.normal(right_key, (columns, width), dtype=jnp.float32), mode="reduced")
    return SketchBank(left, right, jnp.zeros_like(left), jnp.zeros_like(right))


def sketch_basis(sketch, rank):
    basis, values, _ = jnp.linalg.svd(sketch, full_matrices=False)
    return basis[:, :rank], values[:rank] * math.sqrt(sketch.shape[0] / sketch.shape[1])


def spectral_filter(x, state, regularization):
    """Apply left/right fourth-root attenuation with the last value as cutoff."""
    # Fixed shapes allow JIT and checkpoints even for an all-zero sketch. Its
    # zero spectrum gives scales of one, so that side's filter is the identity.
    for left, basis, values in (
        (True, state.left_basis, state.left_values),
        (False, state.right_basis, state.right_values),
    ):
        if basis.shape[1] <= 1:
            continue
        basis = basis[:, :-1]
        mu = jnp.maximum(regularization * values[0], jnp.finfo(jnp.float32).tiny)
        scale = jnp.sqrt(jnp.sqrt((values[-1] + mu) / (values[:-1] + mu)))
        # Preserve the notebook's residual-plus-scaled-projection order.
        if left:
            projected = rr.matmul(basis.T, x)
            x = (x - rr.matmul(basis, projected)) + rr.matmul(basis, scale[:, None] * projected)
        else:
            projected = rr.matmul(x, basis)
            x = (x - rr.matmul(projected, basis.T)) + rr.matmul(projected * scale[None, :], basis.T)
    return x


def _refresh(state, key, spectral):
    refresh_count = state.refresh_count + 1
    rotate = refresh_count % spectral.probe_every == 0
    bank = jax.lax.cond(rotate, lambda _: state.standby, lambda _: state.active, None)
    left_basis, left_values = sketch_basis(bank.left_ema, state.left_basis.shape[1])
    right_basis, right_values = sketch_basis(bank.right_ema, state.right_basis.shape[1])
    standby = jax.lax.cond(
        rotate,
        lambda _: draw_bank(key, bank.left_probe.shape[0], bank.right_probe.shape[0], bank.left_probe.shape[1]),
        lambda _: state.standby,
        None,
    )
    return SpectralMatrixState(
        bank,
        standby,
        left_basis,
        right_basis,
        left_values,
        right_values,
        refresh_count,
        (left_values[0] > 0) | (right_values[0] > 0),
    )


def _accumulate(bank, gradient, beta):
    return bank._replace(
        left_ema=beta * bank.left_ema + (1 - beta) * rr.matmul(gradient, rr.matmul(gradient.T, bank.left_probe)),
        right_ema=beta * bank.right_ema + (1 - beta) * rr.matmul(gradient.T, rr.matmul(gradient, bank.right_probe)),
    )


def randrand(param_labels, config, spectral=SpectralSettings(), *, total_steps=rr.TOTAL_STEPS, schedule=None):
    """Build spectral RandRAND with the shared route, clipping and LR controls.

    Paired-projection controls (pair_k, correction_scale, low_rank_correction,
    max_gram_dim) do not apply. rank_sketch is the actual width, with no padding,
    capped at the smaller matrix dimension. EMA sketches have no dimension
    normalization or bias correction, matching the reference notebook.
    """
    labels, tree = jax.tree_util.tree_flatten(param_labels)
    if not labels or any(label not in ("matrix", "embed", "vector", "head") for label in labels):
        raise ValueError("Labels must be matrix/embed/vector/head")
    if schedule is None:
        if not 0 <= config.warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")

        def schedule(count):
            return rr.lr_multiplier(
                count, total_steps=total_steps, warmup_steps=config.warmup_steps, min_lr_ratio=config.min_lr_ratio
            )

    def flatten(values):
        leaves, structure = jax.tree_util.tree_flatten(values)
        if structure != tree:
            raise ValueError("Parameters, gradients and labels must have identical structures")
        return tuple(leaves)

    def init_fn(params):
        ps = flatten(params)
        key = jax.random.PRNGKey(config.basis_seed)
        matrices = []
        for p, label in zip(ps, labels, strict=True):
            if p.dtype != jnp.float32:
                raise ValueError("Spectral RandRAND requires FP32 master parameters")
            if label != "matrix":
                matrices.append(None)
                continue
            if p.ndim != 2:
                raise ValueError("Body weights must be complete 2D matrices")
            if min(p.shape) < config.min_dim_for_deflation:
                matrices.append(None)
                continue
            rows, columns = p.shape
            width = min(config.rank_sketch, rows, columns)
            rank = min(config.rank, width)
            key, active_key, standby_key = jax.random.split(key, 3)
            matrices.append(
                SpectralMatrixState(
                    draw_bank(active_key, rows, columns, width),
                    draw_bank(standby_key, rows, columns, width),
                    jnp.zeros((rows, rank), jnp.float32),
                    jnp.zeros((columns, rank), jnp.float32),
                    jnp.zeros(rank, jnp.float32),
                    jnp.zeros(rank, jnp.float32),
                    jnp.int32(0),
                    jnp.bool_(False),
                )
            )
        zeros = tuple(jnp.zeros_like(p) for p in ps)
        return rr.RandRANDState(jnp.int32(0), zeros, zeros, tuple(matrices), key)

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("Spectral RandRAND requires current parameters")
        ps = flatten(params)
        gs = rr._clip(tuple(g.astype(jnp.float32) for g in flatten(grads)), labels, config.max_grad_norm)
        elapsed = state.count - config.randrand_warmup
        due = (state.count > 0) & (elapsed >= 0) & (elapsed % config.basis_every == 0)

        def refresh(operand):
            matrices, key = operand
            key, refresh_key = jax.random.split(key)
            return (
                tuple(
                    None if ms is None else _refresh(ms, jax.random.fold_in(refresh_key, i), spectral)
                    for i, ms in enumerate(matrices)
                ),
                key,
            )

        matrices, key = jax.lax.cond(due, refresh, lambda x: x, (state.matrices, state.key))
        count = optax.safe_int32_increment(state.count)
        ratio = schedule(state.count)
        out, first, second, next_matrices = [], [], [], []
        for p, raw_g, mu, nu, ms, label in zip(ps, gs, state.first, state.second, matrices, labels, strict=True):
            g = raw_g
            if ms is not None:
                g = jax.lax.cond(ms.enabled, lambda x: spectral_filter(x, ms, spectral.regularization), lambda x: x, g)
            beta1, beta2 = getattr(config, f"{label}_betas")
            mu = beta1 * mu + (1 - beta1) * g
            nu = beta2 * nu + (1 - beta2) * jnp.square(g)
            eps = config.matrix_eps if label == "matrix" else config.adam_eps
            direction = (mu / (1 - jnp.float32(beta1) ** count)) / (
                jnp.sqrt(nu / (1 - jnp.float32(beta2) ** count)) + eps
            )
            lr = ratio * getattr(config, f"{label}_lr")
            wd = getattr(config, f"{label}_weight_decay")
            if ms is not None:
                direction = jax.lax.cond(
                    ms.enabled, lambda x: spectral_filter(x, ms, spectral.regularization), lambda x: x, direction
                )
                lr *= jnp.where(ms.enabled, config.boost_factor, 1.0)
                ms = ms._replace(
                    active=_accumulate(ms.active, raw_g, config.basis_ema_beta),
                    standby=_accumulate(ms.standby, raw_g, config.basis_ema_beta),
                )
            # Decay uses the boosted LR too. Return the rounded candidate difference
            # so the existing blend observes the same weight-space candidate.
            candidate = p * (1 - lr * wd) - lr * direction
            out.append(candidate - p)
            first.append(mu)
            second.append(nu)
            next_matrices.append(ms)
        return tree.unflatten(out), rr.RandRANDState(count, tuple(first), tuple(second), tuple(next_matrices), key)

    return optax.GradientTransformation(init_fn, update_fn)
