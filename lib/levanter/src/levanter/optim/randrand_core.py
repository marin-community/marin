# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""RandRAND's paired full-Adam-then-split algorithm in JAX/Optax.

No PyTorch dependency. Parameters and all optimizer buffers are FP32. Supply a
matching pytree of labels: 'matrix', 'embed', 'vector', 'head'. Matrix leaves must
be complete (output, input) matrices, not attention heads. Named/stacked tensors
are handled by the separate Levanter adapter.

    tx = randrand(labels, RandRANDConfig(), total_steps=4959)
    state = tx.init(params)
    updates, state = jax.jit(tx.update)(grads, state, params)
    params = optax.apply_updates(params, updates)

Unlike the PyTorch class, update includes the three clipping buckets and LR
schedule. Feed averaged, unscaled gradients; do not clip or schedule again.
All gradient leaves must be present. Freeze parameters before constructing the
optimizer rather than passing changing None gradients. State includes an explicit
PRNG key, so checkpoint/resume does not depend on the model's RNG stream.

The randomized range finder follows PyTorch 2.9.1 svd_lowrank, niter=2, with the
same sketch width. JAX and PyTorch generate different sketches for equal seeds.
Matmul requests HIGHEST precision to retain FP32 optimizer arithmetic on TPUs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

MATRIX_LR = 0.0045
MATRIX_BETAS = (0.95, 0.99)
MATRIX_WEIGHT_DECAY = 0.0303
MATRIX_EPS = 1e-8
EMBED_LR = 0.2
EMBED_BETAS = (0.9, 0.98)
EMBED_WEIGHT_DECAY = 0.0
HEAD_LR = 0.006
HEAD_BETAS = (0.9, 0.98)
HEAD_WEIGHT_DECAY = 0.0
VECTOR_LR = 0.03
VECTOR_BETAS = (0.9, 0.98)
VECTOR_WEIGHT_DECAY = 0.0
ADAM_EPS = 1e-15
DEFLATION_RANK = 32
DEFLATION_SKETCH_DIM = 64
DEFLATION_WARMUP = 100
BASIS_EVERY = 10
BASIS_EMA_BETA = 0.95
MAX_GRAM_DIM = 8192
MIN_DIM_FOR_DEFLATION = 32
BOOST_FACTOR = 5.0
LOW_RANK_CORRECTION = True
CORRECTION_SCALE = 0.5
PAIRED_K = 25
WARM_UP = 1000
MIN_LR_RATIO = 0.0
GRAD_CLIP = 1.0
TOTAL_STEPS = 4959


@dataclass(frozen=True)
class RandRANDConfig:
    """Best-run defaults; independent AdamW settings for auxiliary routes."""

    matrix_lr: float = MATRIX_LR
    matrix_betas: tuple[float, float] = MATRIX_BETAS
    matrix_weight_decay: float = MATRIX_WEIGHT_DECAY
    matrix_eps: float = MATRIX_EPS
    embed_lr: float = EMBED_LR
    embed_betas: tuple[float, float] = EMBED_BETAS
    embed_weight_decay: float = EMBED_WEIGHT_DECAY
    head_lr: float = HEAD_LR
    head_betas: tuple[float, float] = HEAD_BETAS
    head_weight_decay: float = HEAD_WEIGHT_DECAY
    vector_lr: float = VECTOR_LR
    vector_betas: tuple[float, float] = VECTOR_BETAS
    vector_weight_decay: float = VECTOR_WEIGHT_DECAY
    adam_eps: float = ADAM_EPS
    rank: int = DEFLATION_RANK
    rank_sketch: int = DEFLATION_SKETCH_DIM
    randrand_warmup: int = DEFLATION_WARMUP
    basis_every: int = BASIS_EVERY
    basis_ema_beta: float = BASIS_EMA_BETA
    max_gram_dim: int = MAX_GRAM_DIM
    min_dim_for_deflation: int = MIN_DIM_FOR_DEFLATION
    boost_factor: float = BOOST_FACTOR
    low_rank_correction: bool = LOW_RANK_CORRECTION
    correction_scale: float = CORRECTION_SCALE
    pair_k: int = PAIRED_K
    warmup_steps: int = WARM_UP
    min_lr_ratio: float = MIN_LR_RATIO
    max_grad_norm: float | None = GRAD_CLIP
    basis_seed: int = 42

    def __post_init__(self):
        for route in ("matrix", "embed", "vector", "head"):
            for name in ("lr", "weight_decay"):
                value = getattr(self, f"{route}_{name}")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{route}_{name} must be finite and nonnegative")
            betas = getattr(self, f"{route}_betas")
            if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
                raise ValueError(f"{route}_betas must contain two values in [0,1)")
        for name in ("matrix_eps", "adam_eps", "boost_factor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "rank",
            "rank_sketch",
            "randrand_warmup",
            "basis_every",
            "max_gram_dim",
            "min_dim_for_deflation",
            "pair_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rank_sketch < self.rank:
            raise ValueError("rank_sketch must be >= rank")
        if not 0 <= self.basis_ema_beta < 1:
            raise ValueError("basis_ema_beta must be in [0,1)")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if not math.isfinite(self.correction_scale) or self.correction_scale < 0:
            raise ValueError("correction_scale must be finite and nonnegative")
        if not isinstance(self.low_rank_correction, bool):
            raise ValueError("low_rank_correction must be boolean")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive, or None to disable clipping")
        if not isinstance(self.basis_seed, int) or not 0 <= self.basis_seed < 2**32:
            raise ValueError("basis_seed must be an integer in [0,2**32)")


class MatrixState(NamedTuple):
    left_gram: jax.Array
    right_gram: jax.Array
    left_basis: jax.Array
    right_basis: jax.Array
    pair_mask: jax.Array


class RandRANDState(NamedTuple):
    count: jax.Array
    first: tuple
    second: tuple
    matrices: tuple
    key: jax.Array


def matmul(a, b):
    """FP32 contraction, including on hardware with reduced default precision."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def lowrank_basis_from_sketch(gram, sketch, rank):
    """PyTorch 2.9.1 square-matrix svd_lowrank, given the Gaussian sketch.

    Accepting the sketch explicitly permits cross-framework numerical testing.
    Sign choices in SVD are immaterial to the reconstructed paired projection.
    """
    q, _ = jnp.linalg.qr(matmul(gram, sketch), mode="reduced")
    for _ in range(2):
        q, _ = jnp.linalg.qr(matmul(gram.T, q), mode="reduced")
        q, _ = jnp.linalg.qr(matmul(gram, q), mode="reduced")
    small_u, values, _ = jnp.linalg.svd(matmul(q.T, gram), full_matrices=False)
    return matmul(q, small_u)[:, :rank], values[:rank]


def _refresh_matrix(state, key, config):
    rank = state.pair_mask.shape[0]
    left_key, right_key = jax.random.split(key)
    bases, values = [], []
    for gram, side_key in ((state.left_gram, left_key), (state.right_gram, right_key)):
        width = min(max(rank, config.rank_sketch) + 8, gram.shape[0])
        sketch = jax.random.normal(side_key, (gram.shape[0], width), dtype=jnp.float32)
        basis, spectrum = lowrank_basis_from_sketch(gram, sketch, rank)
        bases.append(basis)
        values.append(spectrum)
    scores = (values[0][:, None] * values[1][None, :]).reshape(-1)
    _, indices = jax.lax.top_k(scores, config.pair_k)
    mask = jnp.zeros_like(scores).at[indices].set(1).reshape(rank, rank)
    return MatrixState(state.left_gram, state.right_gram, bases[0], bases[1], mask)


def lr_multiplier(count, *, total_steps=TOTAL_STEPS, warmup_steps=WARM_UP, min_lr_ratio=MIN_LR_RATIO):
    """Return the LR multiplier indexed by completed optimizer steps.

    With positive warmup, start at zero and reach one at warmup_steps.
    With no warmup, start at one and follow cosine decay from the first step.
    """
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")
    count = jnp.asarray(count, jnp.float32)
    fraction = jnp.clip((count - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0)
    cosine = min_lr_ratio + 0.5 * (1 + jnp.cos(jnp.pi * fraction)) * (1 - min_lr_ratio)
    return jnp.where(count < warmup_steps, count / max(1, warmup_steps), cosine)


def _clip(grads, labels, max_norm):
    if max_norm is None:
        return grads
    buckets = tuple("aux" if label in ("embed", "vector") else label for label in labels)
    scales = {}
    for bucket in ("matrix", "aux", "head"):
        # Match torch clip_grad_norm_'s epsilon, including below-threshold behavior.
        squares = [jnp.sum(g * g) for g, b in zip(grads, buckets, strict=True) if b == bucket]
        norm = jnp.sqrt(sum(squares, jnp.float32(0)))
        scales[bucket] = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return tuple(g * scales[b] for g, b in zip(grads, buckets, strict=True))


def randrand(
    param_labels,
    config=None,
    *,
    total_steps=TOTAL_STEPS,
    schedule: Callable | None = None,
):
    """Create an Optax transform with explicit parameter routing and full FP32 state.

    Args:
        param_labels: Pytree matching params, with one route string per array.
        config: RandRANDConfig; defaults reproduce the best-run hyperparameters.
        total_steps: Training horizon for the included cosine schedule.
        schedule: Optional multiplier(count) replacing the included schedule.
    """
    config = RandRANDConfig() if config is None else config
    labels, label_tree = jax.tree_util.tree_flatten(param_labels)
    if not labels or any(label not in ("matrix", "embed", "vector", "head") for label in labels):
        raise ValueError("Labels must be matrix/embed/vector/head, with at least one array")
    if schedule is None:
        if not 0 <= config.warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")

        def schedule(count):
            return lr_multiplier(
                count,
                total_steps=total_steps,
                warmup_steps=config.warmup_steps,
                min_lr_ratio=config.min_lr_ratio,
            )

    def flatten(tree):
        leaves, structure = jax.tree_util.tree_flatten(tree)
        if structure != label_tree:
            raise ValueError("Params, gradients and route labels must have the same pytree structure")
        return tuple(leaves)

    def init_fn(params):
        ps = flatten(params)
        matrices = []
        for p, label in zip(ps, labels, strict=True):
            if p.dtype != jnp.float32:
                raise ValueError("RandRAND JAX requires FP32 master parameters; use BF16 for model compute only")
            if label != "matrix":
                matrices.append(None)
                continue
            if p.ndim != 2 or min(p.shape) < config.min_dim_for_deflation or max(p.shape) > config.max_gram_dim:
                raise ValueError(
                    f"Body matrix {p.shape} must be 2D with dimensions in "
                    f"[{config.min_dim_for_deflation}, {config.max_gram_dim}]"
                )
            m, n = p.shape
            rank = min(config.rank, m, n)
            if not 1 <= config.pair_k < rank**2:
                raise ValueError("Require 1 <= pair_k < min(rank,rows,columns)**2")
            matrices.append(
                MatrixState(
                    jnp.zeros((m, m), jnp.float32),
                    jnp.zeros((n, n), jnp.float32),
                    jnp.zeros((m, rank), jnp.float32),
                    jnp.zeros((n, rank), jnp.float32),
                    jnp.zeros((rank, rank), jnp.float32),
                )
            )
        zeros = tuple(jnp.zeros_like(p, dtype=jnp.float32) for p in ps)
        return RandRANDState(
            jnp.int32(0),
            zeros,
            zeros,
            tuple(matrices),
            jax.random.PRNGKey(config.basis_seed),
        )

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("RandRAND requires current parameters for decoupled weight decay")
        ps = flatten(params)
        gs = _clip(
            tuple(g.astype(jnp.float32) for g in flatten(grads)),
            labels,
            config.max_grad_norm,
        )
        elapsed = state.count - config.randrand_warmup
        due = (elapsed >= 0) & (elapsed % config.basis_every == 0)

        def refresh(operand):
            matrices, key = operand
            next_key, refresh_key = jax.random.split(key)
            return (
                tuple(
                    None if ms is None else _refresh_matrix(ms, jax.random.fold_in(refresh_key, i), config)
                    for i, ms in enumerate(matrices)
                ),
                next_key,
            )

        matrices, key = jax.lax.cond(due, refresh, lambda x: x, (state.matrices, state.key))
        active = state.count >= config.randrand_warmup
        count = optax.safe_int32_increment(state.count)
        ratio = schedule(state.count)
        out, first, second, next_matrices = [], [], [], []
        for p, g, mu, nu, ms, label in zip(ps, gs, state.first, state.second, matrices, labels, strict=True):
            beta1, beta2 = getattr(config, f"{label}_betas")
            mu = beta1 * mu + (1 - beta1) * g
            nu = beta2 * nu + (1 - beta2) * jnp.square(g)
            # Match AdamW's epsilon placement after sqrt of the corrected moment.
            eps = config.matrix_eps if label == "matrix" else config.adam_eps
            direction = (mu / (1 - jnp.float32(beta1) ** count)) / (
                jnp.sqrt(nu / (1 - jnp.float32(beta2) ** count)) + eps
            )
            lr = ratio * getattr(config, f"{label}_lr")
            wd = getattr(config, f"{label}_weight_decay")
            if ms is not None:
                coefficients = matmul(matmul(ms.left_basis.T, direction), ms.right_basis) * ms.pair_mask
                selected = matmul(matmul(ms.left_basis, coefficients), ms.right_basis.T)
                remaining = direction - selected
                boost = jnp.where(active, config.boost_factor, 1.0)
                correction = config.correction_scale if config.low_rank_correction else 0.0
                update = -lr * boost * remaining - lr * correction * selected - lr * boost * wd * p
                beta = config.basis_ema_beta
                ms = ms._replace(
                    left_gram=beta * ms.left_gram + ((1 - beta) / p.shape[1]) * matmul(g, g.T),
                    right_gram=beta * ms.right_gram + ((1 - beta) / p.shape[0]) * matmul(g.T, g),
                )
            else:
                update = -lr * direction - lr * wd * p
            out.append(update)
            first.append(mu)
            second.append(nu)
            next_matrices.append(ms)
        return label_tree.unflatten(out), RandRANDState(count, tuple(first), tuple(second), tuple(next_matrices), key)

    return optax.GradientTransformation(init_fn, update_fn)
