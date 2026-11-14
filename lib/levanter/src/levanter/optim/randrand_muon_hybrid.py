# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Adaptive RandRAND/MuonH weight blending from the w_v06_slr0.015 experiment.

Both candidates start from the same weights. RandRAND remains active after the
blend rises; Muon momentum is accumulated even while its blend weight is zero.
"""

import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import haliax as hax
import jax
import jax.numpy as jnp
import optax

from levanter.optim import randrand_core as rr
from levanter.optim import randrand_spectral as spectral_rr
from levanter.optim.config import OptimizerConfig
from levanter.optim.randrand import NamedRandRANDState, RandRANDConfig, _pack, _unpack, parameter_layout


HYPERBALL_NORM_FLOOR = 1e-10


@dataclass(frozen=True)
class MuonBlendConfig:
    threshold: float = 0.0
    width: float = 0.05
    ema: float = 0.98
    min_step: int = 1000
    ramp: int = 400
    learning_rate: float = 0.015
    momentum: float = 0.95
    ns_steps: int = 5

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not math.isfinite(self.width) or self.width <= 0:
            raise ValueError("Require finite threshold and positive finite width")
        if not 0 <= self.ema < 1 or not 0 <= self.momentum < 1:
            raise ValueError("EMA and momentum must be in [0,1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("MuonH learning rate must be finite and nonnegative")
        for name, minimum in (("min_step", 0), ("ramp", 1), ("ns_steps", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class BlendState(NamedTuple):
    cosine: jax.Array
    cosine_ema: jax.Array
    initialized: jax.Array
    weight: jax.Array
    matrix_norm: jax.Array
    initial_matrix_norm: jax.Array


class HybridState(NamedTuple):
    randrand: rr.RandRANDState
    momentum: tuple
    previous_update: tuple
    blend: BlendState


def muonh_direction(matrix, steps=5):
    """Notebook NS5: FP32 quintic iteration, no shape scaling."""
    x = matrix.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x) + 1e-5)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = rr.matmul(x, x.T)
        x = 3.4445 * x + rr.matmul(-4.7750 * gram + 2.0315 * rr.matmul(gram, gram), x)
    return x.T if transposed else x


def randrand_muon_hybrid(
    labels, config, switch=MuonBlendConfig(), *, total_steps=rr.TOTAL_STEPS, schedule=None, spectral=None
):
    """Build the plain-array transform; clipping and scheduling happen once."""
    flat_labels, tree = jax.tree_util.tree_flatten(labels)
    matrix_indices = tuple(i for i, label in enumerate(flat_labels) if label == "matrix")
    if not matrix_indices:
        raise ValueError("The switch requires at least one body matrix")
    if schedule is None:
        if not 0 <= config.warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")

        def schedule(count):
            return rr.lr_multiplier(
                count, total_steps=total_steps, warmup_steps=config.warmup_steps, min_lr_ratio=config.min_lr_ratio
            )

    unclipped = replace(config, max_grad_norm=None)
    base = (
        rr.randrand(labels, unclipped, total_steps=total_steps, schedule=schedule)
        if spectral is None
        else spectral_rr.randrand(labels, unclipped, spectral, total_steps=total_steps, schedule=schedule)
    )

    def flatten(values):
        leaves, structure = jax.tree_util.tree_flatten(values)
        if structure != tree:
            raise ValueError("Parameters, gradients and labels must have identical structures")
        return tuple(leaves)

    def init_fn(params):
        ps = flatten(params)
        raw = base.init(params)
        buffers = tuple(
            jnp.zeros_like(p) if label == "matrix" else None for p, label in zip(ps, flat_labels, strict=True)
        )
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(ps[i])) for i in matrix_indices))
        blend = BlendState(jnp.float32(0), jnp.float32(0), jnp.bool_(False), jnp.float32(0), norm, norm)
        return HybridState(raw, buffers, buffers, blend)

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("The blend requires current parameters")
        ps = flatten(params)
        gs = rr._clip(tuple(g.astype(jnp.float32) for g in flatten(grads)), flat_labels, config.max_grad_norm)
        # These are reductions over complete logical matrices, including sharded axes.
        dot = sum(jnp.sum(gs[i] * state.previous_update[i]) for i in matrix_indices)
        sq_g = sum(jnp.sum(jnp.square(gs[i])) for i in matrix_indices)
        sq_u = sum(jnp.sum(jnp.square(state.previous_update[i])) for i in matrix_indices)
        valid = (sq_g > 0) & (sq_u > 0)
        cosine = dot / jnp.sqrt(jnp.where(valid, sq_g * sq_u, 1.0))
        valid = valid & jnp.isfinite(cosine)
        ema = jnp.where(
            valid,
            jnp.where(
                state.blend.initialized, switch.ema * state.blend.cosine_ema + (1 - switch.ema) * cosine, cosine
            ),
            state.blend.cosine_ema,
        )
        initialized = state.blend.initialized | valid
        target = jnp.clip((switch.threshold - ema) / switch.width, 0.0, 1.0)
        weight = jnp.where(
            (state.randrand.count >= switch.min_step) & initialized,
            jnp.minimum(jnp.maximum(state.blend.weight, target), state.blend.weight + 1.0 / switch.ramp),
            state.blend.weight,
        )
        updates, raw = base.update(tree.unflatten(gs), state.randrand, params)
        updates = list(flatten(updates))
        momentum, previous = list(state.momentum), list(state.previous_update)
        ratio = schedule(state.randrand.count)
        norm_sq = jnp.float32(0)
        for i in matrix_indices:
            p, g = ps[i], gs[i]
            momentum[i] = switch.momentum * state.momentum[i] + g
            rr_weight = p + updates[i]

            def mix(_):
                direction = muonh_direction(g + switch.momentum * momentum[i], switch.ns_steps)
                radius = jnp.linalg.norm(p)
                candidate = (
                    p
                    - (switch.learning_rate * ratio)
                    * (radius / jnp.maximum(jnp.linalg.norm(direction), HYPERBALL_NORM_FLOOR))
                    * direction
                )
                muon_weight = candidate * (radius / jnp.maximum(jnp.linalg.norm(candidate), HYPERBALL_NORM_FLOOR))
                return (1 - weight) * rr_weight + weight * muon_weight

            new_weight = jax.lax.cond(weight > 0, mix, lambda _: rr_weight, operand=None)
            # Record the actual rounded FP32 parameter difference, as the notebook does.
            updates[i] = jnp.where(weight > 0, new_weight - p, updates[i])
            previous[i] = p - (p + updates[i])
            norm_sq += jnp.sum(jnp.square(p + updates[i]))
        blend = BlendState(
            jnp.where(valid, cosine, 0.0), ema, initialized, weight, jnp.sqrt(norm_sq), state.blend.initial_matrix_norm
        )
        return tree.unflatten(updates), HybridState(raw, tuple(momentum), tuple(previous), blend)

    return optax.GradientTransformation(init_fn, update_fn)


class NamedHybridState(NamedTuple):
    randrand: NamedRandRANDState
    momentum: tuple
    previous_update: tuple
    blend: BlendState

    @property
    def hyperparams(self):
        # Trainer's standard optimizer logger records these scalar values.
        return {
            "switch/blend": self.blend.weight,
            "switch/cosine": self.blend.cosine,
            "switch/cosine_ema": self.blend.cosine_ema,
            "switch/initialized": self.blend.initialized,
            "switch/matrix_norm": self.blend.matrix_norm,
            "switch/matrix_norm_over_init": self.blend.matrix_norm
            / jnp.maximum(self.blend.initial_matrix_norm, 1e-30),
        }


def _name_matrix_buffers(buffers, params, layouts):
    is_named = lambda x: isinstance(x, hax.NamedArray)
    leaves = jax.tree_util.tree_leaves(params, is_leaf=is_named)
    matrix_params = tuple(p for p, layout in zip(leaves, layouts, strict=True) if layout.kind == "matrix")
    matrix_layouts = tuple(layout for layout in layouts if layout.kind == "matrix")
    tree = jax.tree_util.tree_structure(matrix_params, is_leaf=is_named)
    named = iter(_unpack(tuple(b for b in buffers if b is not None), matrix_params, matrix_layouts, tree))
    return tuple(next(named) if layout.kind == "matrix" else None for layout in layouts)


def _pack_matrix_buffers(buffers, layouts):
    matrix_layouts = tuple(layout for layout in layouts if layout.kind == "matrix")
    packed = iter(_pack(tuple(b for b in buffers if b is not None), matrix_layouts))
    return tuple(
        next(packed) if layout.kind == "matrix" else None for layout in layouts for _ in range(layout.batch_size)
    )


def named_hybrid(config, switch, *, total_steps, schedule, spectral=None):
    """Preserve parameter axes on Adam, Muon momentum and previous-update buffers."""

    def transform(layouts):
        labels = tuple(layout.kind for layout in layouts for _ in range(layout.batch_size))
        return randrand_muon_hybrid(
            labels, config, switch, total_steps=total_steps, schedule=schedule, spectral=spectral
        )

    def name(raw, params, layouts, tree):
        rr_state = NamedRandRANDState(
            raw.randrand.count,
            _unpack(raw.randrand.first, params, layouts, tree),
            _unpack(raw.randrand.second, params, layouts, tree),
            raw.randrand.matrices,
            raw.randrand.key,
        )
        return NamedHybridState(
            rr_state,
            _name_matrix_buffers(raw.momentum, params, layouts),
            _name_matrix_buffers(raw.previous_update, params, layouts),
            raw.blend,
        )

    def init_fn(params):
        layouts, tree = parameter_layout(params)
        return name(transform(layouts).init(_pack(params, layouts)), params, layouts, tree)

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("RandRAND/MuonH requires params")
        layouts, tree = parameter_layout(params)
        raw_rr = rr.RandRANDState(
            state.randrand.count,
            _pack(state.randrand.first, layouts),
            _pack(state.randrand.second, layouts),
            state.randrand.matrices,
            state.randrand.key,
        )
        raw = HybridState(
            raw_rr,
            _pack_matrix_buffers(state.momentum, layouts),
            _pack_matrix_buffers(state.previous_update, layouts),
            state.blend,
        )
        updates, raw = transform(layouts).update(_pack(grads, layouts), raw, _pack(params, layouts))
        return _unpack(updates, params, layouts, tree), name(raw, params, layouts, tree)

    return optax.with_extra_args_support(optax.GradientTransformation(init_fn, update_fn))


@OptimizerConfig.register_subclass("randrand_muon_hybrid")
@dataclass(frozen=True)
class RandRANDMuonHybridConfig(RandRANDConfig):
    """w_v06_slr0.015 defaults for body-matrix blending and auxiliary AdamW routes."""

    learning_rate: float = 0.006
    matrix_betas: tuple[float, float] = (0.95, 0.95)
    embed_lr: float = 0.1
    vector_lr: float = 0.06
    low_rank_correction: bool = False
    correction_scale: float = 0.0
    switch: MuonBlendConfig = field(default_factory=MuonBlendConfig)

    def build(self, num_train_steps):
        return named_hybrid(
            self.algorithm_config(),
            self.switch,
            spectral=self.spectral_settings(),
            total_steps=num_train_steps,
            schedule=self.lr_scheduler(num_train_steps, override_lr=1.0),
        )
