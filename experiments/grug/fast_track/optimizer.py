# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import re
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig, _convert_frac_or_steps
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.fast_track.adamh import scale_by_adamh
from experiments.grug.fast_track.eig_muon import EIG_MODES, scale_by_eig_direction
from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon, _target_named_sharding
from experiments.grug.fast_track.okls import OKLS_MATMUL_DTYPES, scale_with_grug_okls
from experiments.grug.fast_track.stiefel import scale_with_stiefel_muon


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state
        updates = _match_named_sharding_to_params(updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _pin_sharding(x, ref):
    """Reshard ``x`` to ``ref``'s named sharding so a following norm reduces correctly.

    ``new_param`` is a computed intermediate; leaving it with an SPMD-inferred sharding lets the sharded
    ``norm(new_param)`` over-count and collapse the tensor (issue #8073). This reshard is a same-layout
    no-op at runtime.
    """
    sharding = _target_named_sharding(ref)
    return jax.sharding.reshard(x, sharding) if sharding is not None else x


def _scale_invariant_hyperball_updates(
    params, direction_updates, learning_rate, per_expert: bool = False, lr_mults=None
):
    """MuonH hyperball step: move along the orthogonalized direction, then project back to the
    parameter's Frobenius sphere (scale-invariant update). Stacked leaves take one sphere per layer, and
    with ``per_expert`` the 4-D expert stacks ``[L, E, in, out]`` take one sphere per (layer, expert).
    ``lr_mults`` (a tree like ``params``, leaves broadcastable per sphere, or None) scales each step's
    learning rate; a zero multiplier leaves that sphere where it is."""
    direction_updates = _match_named_sharding_to_params(direction_updates, params)
    if lr_mults is None:
        lr_mults = jax.tree.map(lambda _: None, params)

    def scale_invariant_update(param, update, lr_mult):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        lr = learning_rate if lr_mult is None else learning_rate * lr_mult
        if param.ndim == 2:
            # jnp.linalg.norm over a sharded matrix mis-lowers under SPMD and over-counts (issue #8073);
            # sum-of-squares in float32 plus a same-layout reshard of the intermediate reduces correctly.
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param.astype(jnp.float32))))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update.astype(jnp.float32))))
            new_param = param - lr * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param = _pin_sharding(new_param, param)
            new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param.astype(jnp.float32))))
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = (2, 3) if per_expert and param.ndim == 4 else tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - lr * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param = _pin_sharding(new_param, param)  # correct the sharded norm reduction (issue #8073)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(scale_invariant_update, params, direction_updates, lr_mults, is_leaf=lambda x: x is None)


MUONH_RETRACTIONS = ("frobenius", "spectral")
# Power iterations per step for sigma_1 (warm-started from the previous step's vector), and at init (cold).
_SPECTRAL_POWER_ITERS = 3
_SPECTRAL_INIT_POWER_ITERS = 30


class SpectralSphereState(NamedTuple):
    """Per-matrix MuonSphere state: warm-start right singular vector ``v`` ``[..., 1, fan_out]``, target
    spectral radius ``radius`` and the last pre-retraction sigma_1 estimate ``sigma`` (``[..., 1, 1]``)."""

    v: optax.Updates
    radius: optax.Updates
    sigma: optax.Updates


def _top_singular_pair(w32: jax.Array, v: jax.Array, iters: int) -> tuple[jax.Array, jax.Array]:
    """``(sigma_1, v_1)`` of every matrix ``[..., in, out]`` by ``iters`` power iterations from ``v``.

    Broadcast-multiply + sum rather than einsum, as in ``_shrink_top_direction``: a reduction over a
    sharded axis lowers to a plain all-reduce under explicit sharding."""
    sigma = jnp.ones((*v.shape[:-1], 1), jnp.float32)
    for _ in range(iters):
        u = jnp.sum(w32 * v, axis=-1, keepdims=True)
        u = u / (jnp.sqrt(jnp.sum(jnp.square(u), axis=-2, keepdims=True)) + 1e-12)
        v = jnp.sum(w32 * u, axis=-2, keepdims=True)
        sigma = jnp.sqrt(jnp.sum(jnp.square(v), axis=-1, keepdims=True))
        v = v / (sigma + 1e-12)
    return sigma, v


def _spectral_sphere_init(params, radius_c: float | None) -> SpectralSphereState:
    """Cold-start power iteration on the initial weights. ``radius_c`` sets R = c sqrt(fan_out / fan_in)
    (arXiv 2601.08393, spectral muP); None keeps each matrix's initial sigma_1 (MuonH's keep-the-init-norm
    rule, in the spectral norm)."""

    def leaf(p):
        if not hasattr(p, "ndim") or p.ndim < 2:
            return None, None, None
        w32 = p.astype(jnp.float32)
        v0 = jnp.ones((*p.shape[:-2], 1, p.shape[-1]), jnp.float32) / jnp.sqrt(p.shape[-1])
        sigma, v = _top_singular_pair(w32, v0, _SPECTRAL_INIT_POWER_ITERS)
        fan_in, fan_out = p.shape[-2:]
        radius = sigma if radius_c is None else jnp.full_like(sigma, radius_c * (fan_out / fan_in) ** 0.5)
        return v, radius, sigma

    out = jax.tree.map(leaf, params)
    pick = lambda i: jax.tree.map(lambda _, o: o[i], params, out)  # noqa: E731
    return SpectralSphereState(pick(0), pick(1), pick(2))


def _spectral_sphere_updates(params, direction_updates, learning_rate, state: SpectralSphereState, lr_mults=None):
    """MuonSphere step (arXiv 2601.08393, SSO with lambda = 0): retract ``W <- W R / sigma_1(W)``, then
    ``W <- W - lr R Phi``, with ``Phi`` the direction at msign scale (``||Phi||_F = sqrt(min(in, out))``,
    unit spectral norm for an exactly orthogonal direction). The spectral norm of the step is ``lr R``, so
    ``lr`` is the relative step in the spectral norm (MuonH: in the Frobenius norm). As in the paper the
    retraction is applied before the step, sharing one warm-started power iteration per matrix per step;
    the stored weights therefore sit ``O(lr)`` off the sphere. One sphere per matrix: per layer of a
    ``[L, in, out]`` stack and per (layer, expert) of an ``[L, E, in, out]`` stack. ``lr_mults`` scales each
    matrix's step as in ``_scale_invariant_hyperball_updates`` (a zero multiplier still retracts)."""
    direction_updates = _match_named_sharding_to_params(direction_updates, params)
    if lr_mults is None:
        lr_mults = jax.tree.map(lambda _: None, params)

    def leaf(param, update, v, radius, lr_mult):
        if param is None or v is None:
            return update, None, None
        lr = learning_rate if lr_mult is None else learning_rate * lr_mult
        w32 = param.astype(jnp.float32)
        sigma, v = _top_singular_pair(w32, v, _SPECTRAL_POWER_ITERS)
        fan_in, fan_out = param.shape[-2:]
        u32 = update.astype(jnp.float32)
        u_norm = jnp.sqrt(jnp.sum(jnp.square(u32), axis=(-2, -1), keepdims=True))
        phi = u32 * (min(fan_in, fan_out) ** 0.5 / jnp.maximum(u_norm, 1e-10))
        new_param = w32 * (radius / jnp.maximum(sigma, 1e-10)) - lr * radius * phi
        new_param = _pin_sharding(new_param, param)
        return (new_param - w32).astype(update.dtype), v, sigma

    none_leaf = lambda x: x is None  # noqa: E731
    out = jax.tree.map(leaf, params, direction_updates, state.v, state.radius, lr_mults, is_leaf=none_leaf)
    pick = lambda i: jax.tree.map(lambda _, o: o[i], params, out, is_leaf=none_leaf)  # noqa: E731
    return pick(0), SpectralSphereState(pick(1), state.radius, pick(2))


# KDA-layer leaves (``kda_blocks.stacked.attn.<leaf>``) and their update rules. The q/k/v/o and
# output-gate matrices and the random-init ``kda_dd_rope`` angle projections (w_rot_down/w_rot_up) take the
# MuonH catch-all, the ShortConv kernels and output-norm scale are ``.weight`` leaves (Adam), and the zero-init
# angle amplitude ``rot_scale`` is on the generic Adam list.
_KDA_ATTN_LEAF = re.compile(r"kda_blocks(?:_tail)?\.stacked\.attn\.(\w+)")
# Low-rank forget gate, per-head A_log, per-channel dt_bias and the zero-init push / erase-gate
# projections (MuonH cannot move a zero matrix): Adam (no weight decay).
_KDA_ADAM_LEAVES = frozenset({"w_a_down", "w_a_up", "a_log", "dt_bias", "push_decay", "w_push", "w_erase"})
# Write-strength projection: MuonH at ``kda_beta_lr_mult`` x the MuonH LR.
_KDA_BETA_LEAF = "w_beta"
# Low-rank write-strength MLP (``kda_beta_rank``): LR group chosen by ``kda_beta_mlp_group``.
_KDA_BETA_MLP_LEAVES = frozenset({"w_beta_down", "w_beta_up"})


# Matrix families that ``okls_targets`` can move from MuonH to the OKLS direction.
# Matrix types for ``muon_truncate_family`` (the attention families split by layer kind).
_TRUNCATE_FAMILIES: dict[str, re.Pattern] = {
    "kda": re.compile(r"kda_blocks(_tail)?\.stacked\.attn\.w_\w+$"),
    "mla": re.compile(r"stacked_blocks(_tail)?\.stacked\.attn\.w_\w+$"),
    "latent": re.compile(r"\.mlp\.w_latent_(down|up)$"),
    "shared": re.compile(r"\.shared\.\d+\.w_(gate|up|down)$"),
    "routed": re.compile(r"\.mlp\.expert_mlp\.w_(gate|up|down)$"),
}

_OKLS_FAMILIES: dict[str, re.Pattern] = {
    "attn": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_(q|k|v|o|g|dkv|uk|uv|q2|uk2)$"),
    "routed": re.compile(r"\.mlp\.expert_mlp\.w_(gate|up|down)$"),
    "shared": re.compile(r"\.shared\.\d+\.w_(gate|up|down)$"),
    "latent": re.compile(r"\.mlp\.w_latent_(down|up)$"),
    "gated_norm": re.compile(r"gated_norm\.w_(down|up)$"),
    # Per-projection subsets of "attn" (KDA and MLA together).
    "attn_q": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_q2?$"),
    "attn_k": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_(k|uk|uk2)$"),
    "attn_v": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_(v|uv)$"),
    "attn_o": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_o$"),
    "attn_other": re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_(g|dkv)$"),
    # The softmax (MLA) layers' q and k: KDA L2-normalizes q and k, so their scale is inert there.
    "mla_qk": re.compile(r"stacked_blocks(_tail)?\.stacked\.attn\.w_(q|uk|q2|uk2)$"),
}
# Softmax-attention (``stacked_blocks``) query and key projections: GQA ``w_q``/``w_k``, MLA ``w_q``/``w_uk`` (and
# the DIFF pair). MLA's ``w_dkv`` is left out: it feeds the values as well.
_SOFTMAX_QK = re.compile(r"stacked_blocks\.stacked\.attn\.w_(q|k|uk|q2|uk2)$")
# Query/key projections of every attention layer: KDA ``w_q`` / ``w_k`` and MLA ``w_q`` / ``w_uk``.
_QK_PROJECTIONS = re.compile(r"(stacked_blocks|kda_blocks)(_tail)?\.stacked\.attn\.w_(q|k|uk)$")


_MEMORY_VALUES = re.compile(r"memory\.\d+\.values")
_MEMORY_ADAM = re.compile(r"memory\.\d+\.(keys|w_out)")
_OUTPUT_BIGRAM = re.compile(r"(?:^|\.)output_bigram_[uw]$")


def _kda_leaf(path_lower: str) -> str | None:
    match = _KDA_ATTN_LEAF.fullmatch(path_lower)
    return None if match is None else match.group(1)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for exactly the ``attn_gate`` and MoE ``router`` weight leaves.

    Matches the leaf attribute name at the end of the path, so it selects ``...attn.attn_gate`` and
    ``...mlp.router`` but not the separate ``...mlp.router_bias`` leaf.
    """
    return path_lower.endswith((".attn_gate", ".router", ".router_down", ".router_up"))


def _is_router_weight(path_lower: str) -> bool:
    """True for the MoE router weight leaves (full-rank ``router`` or low-rank ``router_down`` / ``router_up``)."""
    return path_lower.endswith((".mlp.router", ".mlp.router_down", ".mlp.router_up"))


def _is_zero_centered_gain(path_lower: str) -> bool:
    """True for the ``gamma`` leaves of zero-centered RMSNorms (``zero_centered_gains``)."""
    return path_lower.endswith(".gamma")


def _adam_decay_coefficients(params, gate_router_weight_decay: float, gain_weight_decay: float):
    """Per-leaf decoupled weight decay of the ``adam`` group: ``gate_router_weight_decay`` on the
    ``attn_gate`` and ``router`` weight leaves, ``gain_weight_decay`` on the zero-centered norm gains
    (decaying ``gamma`` pulls the gain ``1 + gamma`` toward 1), 0 everywhere else."""
    paths = leaf_key_paths(params)

    def coefficient(_, path):
        path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
        if _is_gate_or_router_weight(path_lower):
            return gate_router_weight_decay
        if _is_zero_centered_gain(path_lower):
            return gain_weight_decay
        return 0.0

    return jax.tree.map(coefficient, params, paths)


def _ademamix_beta3_schedule(beta1: float, beta3: float, total_steps: int):
    """AdEMAMix's beta3 warmup: linear in half-life from ``beta1``'s to ``beta3``'s over ``total_steps``."""
    log_b1, log_b3 = jnp.log(beta1), jnp.log(beta3)

    def schedule(step):
        frac = jnp.clip(step / total_steps, 0.0, 1.0)
        return jnp.exp(log_b1 * log_b3 / ((1 - frac) * log_b3 + frac * log_b1))

    return schedule


class GrokfastState(NamedTuple):
    ema: optax.Updates


def scale_by_grokfast_ema(alpha: float, lamb: float) -> optax.GradientTransformation:
    """Grokfast-EMA: add ``lamb`` x an EMA (decay ``alpha``) of the gradients to each gradient."""

    def init_fn(params):
        return GrokfastState(jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        ema = jax.tree.map(lambda e, g: alpha * e + (1 - alpha) * g.astype(e.dtype), state.ema, updates)
        updates = jax.tree.map(lambda g, e: (g + lamb * e).astype(g.dtype), updates, ema)
        return updates, GrokfastState(ema)

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_adam_decoupled_decay(
    adam: optax.GradientTransformation,
    weight_decay: float,
    gain_weight_decay: float,
    total_steps: int,
    cautious_decay: bool = False,
) -> optax.GradientTransformation:
    """``scale_by_adam`` plus decoupled weight decay on ``attn_gate`` and the ``router`` weight (and
    ``gain_weight_decay`` on the zero-centered norm gains), annealed linearly to 0 over ``total_steps``.
    The coefficient reads the Adam ``count`` and the state stays ``ScaleByAdamState``, so a checkpoint
    written without decay resumes at the right step with its moments intact.

    ``cautious_decay`` (arXiv 2510.12402, Algorithm 1): decay only the coordinates where the Adam update
    ``u`` and the parameter agree in sign, ``u + lambda I(u x >= 0) x``, i.e. where the update already
    shrinks the magnitude."""

    def init_fn(params):
        return adam.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_scale_by_adam_decoupled_decay requires params for decoupled decay")
        step = optax.tree_utils.tree_get(state, "count")
        updates, next_state = adam.update(updates, state, params)
        anneal = jnp.clip(1.0 - step / total_steps, 0.0, None)
        coefficients = _adam_decay_coefficients(params, weight_decay, gain_weight_decay)

        def decay(u, p, c):
            if not c:
                return u
            if cautious_decay:
                return u + anneal * c * p * (u * p >= 0).astype(p.dtype)
            return u + anneal * c * p

        updates = jax.tree.map(decay, updates, params, coefficients)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_sinkhorn_momentum(
    *, momentum: float, iters: int, nesterov: bool, eps: float = 1e-8
) -> optax.GradientTransformation:
    """Momentum followed by Sinkhorn balancing (DeepSeek-V4.1-Flash; SinkGD, Scetbon et al. 2025).

    Keeps one momentum buffer per matrix. The direction alternately rescales every row, then every
    column, of the (Nesterov) momentum to unit RMS, ``iters`` times, so every token row and every
    output channel of an embedding / head matrix moves at a comparable rate. Only a momentum buffer is
    stored, like Muon.
    """

    def _rms(x, axis):
        return jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True))

    def _balance(m):
        x = m.astype(jnp.float32)
        for _ in range(iters):
            x = x / (_rms(x, -1) + eps)
            x = x / (_rms(x, -2) + eps)
        return x

    def init_fn(params):
        return jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params)

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree.map(lambda m, g: momentum * m + g.astype(jnp.float32), state, updates)
        direction_source = (
            jax.tree.map(lambda m, g: g.astype(jnp.float32) + momentum * m, mu, updates) if nesterov else mu
        )
        directions = jax.tree.map(lambda m, g: _balance(m).astype(g.dtype), direction_source, updates)
        return directions, mu

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_grug_muon_free(
    *,
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    learning_rate,
    coefficient_type: CoefficientType,
    head_dim: int | None,
    weight_decay: float,
) -> optax.GradientTransformation:
    """Muon without the hyperball: the MuonH step size, but the norm is free.

    The update is ``-lr * |W_0| * d / |d| - lr * weight_decay * W``, where ``d`` is the Newton-Schulz
    direction and ``|W_0|`` the matrix's initial Frobenius norm (per layer for stacked leaves), so the
    step matches MuonH's first step. With orthogonal steps, the norm settles near
    ``|W|^2 / |W_0|^2 = lr / (2 * weight_decay)``.
    """
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        head_dim=head_dim,
    )

    def _norm(x):
        axes = None if x.ndim == 2 else tuple(range(1, x.ndim))
        return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axes, keepdims=x.ndim != 2))

    def init_fn(params):
        norms = jax.tree.map(lambda x: _norm(x) if hasattr(x, "ndim") and x.ndim >= 2 else None, params)
        return muon_transform.init(params), norms

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muon_free requires params for weight decay")
        muon_state, init_norms = state
        directions, muon_state = muon_transform.update(updates, muon_state, params)
        directions = _match_named_sharding_to_params(directions, params)

        def step(param, direction, init_norm):
            if direction is None or init_norm is None:
                return direction
            scaled = direction.astype(jnp.float32) * init_norm / jnp.maximum(_norm(direction), 1e-10)
            return (-learning_rate * (scaled + weight_decay * param.astype(jnp.float32))).astype(param.dtype)

        new_updates = jax.tree.map(step, params, directions, init_norms, is_leaf=lambda x: x is None)
        return new_updates, (muon_state, init_norms)

    return optax.GradientTransformation(init_fn, update_fn)


def _eig_hyperball(inner: optax.GradientTransformation, learning_rate, per_expert: bool) -> optax.GradientTransformation:
    """``inner``'s direction (``eig_muon``) taken as a MuonH Frobenius hyperball step."""

    def update_fn(updates, state, params=None):
        directions, state = inner.update(updates, state, params)
        return _scale_invariant_hyperball_updates(params, directions, learning_rate, per_expert), state

    return optax.GradientTransformation(inner.init, update_fn)


def _sinkhorn_hyperball(momentum: float, iters: int, nesterov: bool, learning_rate) -> optax.GradientTransformation:
    """Sinkhorn-balanced momentum direction with the MuonH/AdamH Frobenius hyperball step."""
    sinkhorn = scale_by_sinkhorn_momentum(momentum=momentum, iters=iters, nesterov=nesterov)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("sinkhornh requires params for the hyperball step")
        directions, state = sinkhorn.update(updates, state, params)
        return _scale_invariant_hyperball_updates(params, directions, learning_rate), state

    return optax.GradientTransformation(sinkhorn.init, update_fn)


# Leaves the trainer writes as data statistics (never trained): the fixed-encoder n-gram table and its code.
LATENT_PROJ_UPDATES = ("muonh", "frozen", "stiefel")
_LATENT_PROJ = re.compile(r"(?:^|\.)w_latent_(down|up)$")
_FROZEN_LEAVES = re.compile(r"(?:^|\.)(ngram_stat_(table|code)|latent_select_mask|embed2_sign_table)$")
# The groups built by ``muonh_transform_at`` (the ones ``muonh_retraction`` applies to).
_MUONH_GROUPS = frozenset({"muonh", "kda_beta", "muonh_attn", "muonh_routed", "muonh_router", "upper_qk", "muonh_qk"})
_HYPERBALL_GROUPS = _MUONH_GROUPS | {"adamh", "sinkhornh"}
_ROUTER_GROUPS = ("adam", "muonh")


class SnooState(NamedTuple):
    count: jax.Array
    slow: optax.Params
    momentum: optax.Updates
    inner: optax.OptState


def _sphere_norm(x: jax.Array) -> jax.Array:
    """Frobenius norm per layer (axis 0 of a stacked leaf) or of the whole 2-D matrix, in float32."""
    axes = tuple(range(x.ndim)) if x.ndim == 2 else tuple(range(1, x.ndim))
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axes, keepdims=True))


def _power_decay_schedule(peak: float, floor: float, warmup_steps: int, total_steps: int, power: float):
    """Linear warmup to ``peak``, then ``floor + (peak - floor) * (1 - p**power)``, ``p`` = post-warmup progress."""

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * step / max(warmup_steps, 1)
        progress = jnp.clip((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0)
        decayed = floor + (peak - floor) * (1.0 - progress**power)
        return jnp.where(step < warmup_steps, warm, decayed)

    return schedule


class MuonMomentumState(NamedTuple):
    count: jax.Array
    buf: optax.Updates
    fast: optax.Updates | None
    slow: optax.Updates | None


@dataclass(frozen=True)
class BiMaxwellRails:
    """Bi-Maxwell's twin rails (ANVIL II): EMA rates of the fast and slow rail, and the slow rail's blend weight."""

    fast_rate: float = 0.15
    slow_rate: float = 0.02
    slow_weight: float = 0.5615
    slow_from_start: bool = False
    """ANVIL II's slow rail: a zero-init EMA accumulating from step 0, kept (not restarted) when the rails engage.
    Off: both rails start from the Nesterov buffer at the switch step."""


DEFAULT_RAILS = BiMaxwellRails()


def _upper_qk_schedule(mult: float, release_step: int, ramp_steps: int):
    """``mult`` until ``release_step``, then a linear ramp to 1 over ``ramp_steps`` (arXiv 2605.10504)."""

    def schedule(step):
        frac = jnp.clip((jnp.asarray(step, jnp.float32) - release_step) / ramp_steps, 0.0, 1.0)
        return mult + (1.0 - mult) * frac

    return schedule


def _momentum_warmup_schedule(start: float, end: float, warmup_steps: int):
    """Linear momentum warmup from ``start`` to ``end`` over the first ``warmup_steps`` steps (modded-nanogpt)."""

    def schedule(step):
        frac = jnp.clip(jnp.asarray(step, jnp.float32) / max(warmup_steps, 1), 0.0, 1.0)
        return (1.0 - frac) * start + frac * end

    return schedule


def scale_by_muon_momentum(
    momentum_schedule, nesterov: bool, switch_step: int | None, rails: BiMaxwellRails = DEFAULT_RAILS
) -> optax.GradientTransformation:
    """MuonH momentum outside Newton-Schulz, with a step-dependent coefficient ``momentum_schedule(step)``
    (``step`` 0-based) and optionally Bi-Maxwell (modded-nanogpt #339) from ``switch_step`` on.

    Before ``switch_step`` (or always, when it is None) it is (Nesterov) momentum: ``buf = m buf + g``, out
    ``m buf + g`` (``buf`` without ``nesterov``). From ``switch_step`` on, a fast and a slow EMA of the gradient
    (``rails``: by default rates 0.15 and 0.02), both started from ``(1 - m) buf``, mix as
    ``M = (1 - w) fast + w slow`` (by default w = 0.5615, mean age about 30 steps) and the output is
    ``g + m (M - g)``. Newton-Schulz is scale-invariant per matrix, so the two regimes' different scales don't
    matter."""

    def init(params):
        zeros = lambda: jax.tree.map(jnp.zeros_like, params)  # noqa: E731
        bimaxwell = switch_step is not None
        return MuonMomentumState(
            jnp.zeros([], jnp.int32), zeros(), zeros() if bimaxwell else None, zeros() if bimaxwell else None
        )

    def update(updates, state, params=None):
        count = state.count + 1
        momentum = momentum_schedule(state.count)

        def early(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            new_buf = jax.tree.map(lambda g, b: (momentum * b + g).astype(b.dtype), g_tree, buf_tree)
            if rails.slow_from_start and slow_tree is not None:
                slow_tree = jax.tree.map(lambda g, s: (s + rails.slow_rate * (g - s)).astype(s.dtype), g_tree, slow_tree)
            if not nesterov:
                return new_buf, new_buf, fast_tree, slow_tree
            out = jax.tree.map(lambda g, b: momentum * b + g, g_tree, new_buf)
            return out, new_buf, fast_tree, slow_tree

        if switch_step is None:
            out, buf, _, _ = early((updates, state.buf, None, None))
            return out, MuonMomentumState(count, buf, None, None)

        def late(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            # At the switch step, start both EMAs from the Nesterov buffer's EMA-scale value.
            first = count == switch_step + 1

            def leaf(g, buf, fast, slow):
                start = (1.0 - momentum) * buf
                fast = jnp.where(first, start, fast)
                if not rails.slow_from_start:
                    slow = jnp.where(first, start, slow)
                fast = fast + rails.fast_rate * (g - fast)
                slow = slow + rails.slow_rate * (g - slow)
                mix = (1.0 - rails.slow_weight) * fast + rails.slow_weight * slow
                return (g + momentum * (mix - g)).astype(g.dtype), fast.astype(buf.dtype), slow.astype(buf.dtype)

            res = jax.tree.map(leaf, g_tree, buf_tree, fast_tree, slow_tree)

            def pick(i):
                return jax.tree.map(lambda _, r: r[i], g_tree, res)

            return pick(0), buf_tree, pick(1), pick(2)

        out, buf, fast, slow = jax.lax.cond(
            count > switch_step, late, early, (updates, state.buf, state.fast, state.slow)
        )
        return out, MuonMomentumState(count, buf, fast, slow)

    return optax.GradientTransformation(init, update)


def _muon_first_moment(
    state: MuonMomentumState, switch_step: int | None, rails: BiMaxwellRails = DEFAULT_RAILS
) -> optax.Updates:
    """The momentum stage's first-moment estimate after its update: ``buf`` (Nesterov phase) or the
    Bi-Maxwell rail mix. Only its direction matters (Magma's cosine)."""
    if switch_step is None:
        return state.buf
    assert state.fast is not None and state.slow is not None
    late = state.count > switch_step
    return jax.tree.map(
        lambda b, f, s: jnp.where(late, (1.0 - rails.slow_weight) * f + rails.slow_weight * s, b),
        state.buf,
        state.fast,
        state.slow,
    )


class MagmaState(NamedTuple):
    count: jax.Array
    scale: optax.Updates
    """Per-block EMA of the alignment score ``sigmoid(cos(mu, g) / tau)``."""


_MAGMA_EMA = 0.9
_MAGMA_TAU = 2.0


def _magma_block_shape(x) -> tuple[int, ...] | None:
    """One Magma block per matrix: the whole 2-D leaf, or each slice of a stacked leaf's leading axis."""
    if x is None or not hasattr(x, "ndim") or x.ndim < 2:
        return None
    return () if x.ndim == 2 else (x.shape[0],) + (1,) * (x.ndim - 1)


def _magma_init(params) -> MagmaState:
    def zeros(p):
        shape = _magma_block_shape(p)
        return None if shape is None else jnp.zeros(shape, jnp.float32)

    return MagmaState(jnp.zeros([], jnp.int32), jax.tree.map(zeros, params))


def _magma_lr_mults(state: MagmaState, grads, first_moment, *, keep_prob: float, seed: int):
    """Magma (arXiv 2602.15322, Algorithm 1): per block, ``s = 0.9 s + 0.1 sigmoid(cos(mu, g) / 2)`` and a
    ``Bernoulli(keep_prob)`` keep mask; the block's step is ``s * mask`` x the base step (no ``1/p``
    rescale: the paper's damping is deliberately biased). The EMA starts at its first sample (the paper
    does not give ``s_0``). Masks come from ``fold_in(PRNGKey(seed), step)``, the same on every host.
    Returns the per-block learning-rate multipliers and the next state."""
    count = state.count + 1
    step_key = jax.random.fold_in(jax.random.PRNGKey(seed), count)
    none_leaf = lambda x: x is None  # noqa: E731
    leaves, treedef = jax.tree.flatten(state.scale, is_leaf=none_leaf)
    g_leaves = treedef.flatten_up_to(grads)
    m_leaves = treedef.flatten_up_to(first_moment)
    scales, mults = [], []
    for i, (s, g, mu) in enumerate(zip(leaves, g_leaves, m_leaves, strict=True)):
        if s is None:
            scales.append(None)
            mults.append(None)
            continue
        g32, mu32 = g.astype(jnp.float32), mu.astype(jnp.float32)
        axes = tuple(range(g.ndim)) if g.ndim == 2 else tuple(range(1, g.ndim))
        dot = jnp.sum(g32 * mu32, axis=axes, keepdims=g.ndim != 2)
        norms = jnp.sqrt(jnp.sum(jnp.square(g32), axis=axes, keepdims=g.ndim != 2)) * jnp.sqrt(
            jnp.sum(jnp.square(mu32), axis=axes, keepdims=g.ndim != 2)
        )
        score = jax.nn.sigmoid(dot / jnp.maximum(norms, 1e-30) / _MAGMA_TAU)
        s = jnp.where(count == 1, score, _MAGMA_EMA * s + (1 - _MAGMA_EMA) * score)
        keep = jax.random.bernoulli(jax.random.fold_in(step_key, i), keep_prob, s.shape).astype(jnp.float32)
        scales.append(s)
        mults.append(s * keep)
    unflatten = functools.partial(jax.tree.unflatten, treedef)
    return unflatten(mults), MagmaState(count, unflatten(scales))


def magma_metrics(opt_state) -> dict[str, jax.Array]:
    """Mean and min of the Magma alignment EMA over every MuonH block (empty without Magma)."""
    states = [
        x for x in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, MagmaState)) if isinstance(x, MagmaState)
    ]
    scales = [s.reshape(-1) for state in states for s in jax.tree.leaves(state.scale)]
    if not scales:
        return {}
    flat = jnp.concatenate(scales)
    return {"train/magma_scale_mean": jnp.mean(flat), "train/magma_scale_min": jnp.min(flat)}


class MuonHState(NamedTuple):
    """MuonH state when the momentum runs outside Newton-Schulz (scheduled momentum, Bi-Maxwell or Magma)."""

    momentum: MuonMomentumState
    core: optax.OptState
    magma: MagmaState | None
    sphere: SpectralSphereState | None
    """MuonSphere state (``retraction="spectral"``)."""


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate=0.02,
    coefficient_type: CoefficientType = "quintic",
    head_dim: int | None = None,
    neuron_norm_beta2: float | None = None,
    hyperball_per_expert: bool = False,
    pre_norm: str = "none",
    top_shrink: float = 0.0,
    precond_beta2: float | None = None,
    truncate_frac: float = 0.0,
    momentum_schedule=None,
    bimaxwell_switch_step: int | None = None,
    bimaxwell_rails: BiMaxwellRails = DEFAULT_RAILS,
    magma_keep_prob: float | None = None,
    magma_seed: int = 0,
    retraction: str = "frobenius",
    spectral_radius_c: float | None = 2.0,
) -> optax.GradientTransformation:
    """MuonH transform for the stacked model: Newton-Schulz direction + Frobenius hyperball step.

    ``neuron_norm_beta2`` adds NorMuon's (arXiv 2510.05491) neuron-wise normalization between the two:
    each output column of the orthogonalized direction is divided by the root of an EMA of its mean
    square (over the input axis); the hyperball step then sets the overall magnitude.

    ``momentum_schedule`` (step -> coefficient; warmup), ``bimaxwell_switch_step`` (Bi-Maxwell from that
    step) or ``magma_keep_prob`` move the momentum to ``scale_by_muon_momentum`` ahead of Newton-Schulz
    (Nesterov momentum there equals the in-Muon one). ``magma_keep_prob`` then applies Magma
    (``_magma_lr_mults``, alignment of that first moment with the gradient) to each matrix's hyperball
    step. ``learning_rate`` may broadcast per stacked slice (``[L, 1, 1]``).

    ``retraction="spectral"`` replaces the Frobenius hyperball with MuonSphere's spectral sphere
    (``_spectral_sphere_updates``, radius from ``spectral_radius_c``); Magma's multipliers then scale the
    spectral step. Without the external momentum stage the state gains a trailing ``SpectralSphereState``
    (``(core, sphere)``); with it the sphere state is ``MuonHState.sphere``.
    """
    if retraction not in MUONH_RETRACTIONS:
        raise ValueError(f"retraction must be one of {MUONH_RETRACTIONS}, got {retraction!r}")
    spectral = retraction == "spectral"
    external_momentum = momentum_schedule is not None or bimaxwell_switch_step is not None or magma_keep_prob is not None
    muon_transform = _grug_scale_with_muon(
        momentum=0.0 if external_momentum else momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        head_dim=head_dim,
        pre_norm=pre_norm,
        top_shrink=top_shrink,
        precond_beta2=precond_beta2,
        truncate_frac=truncate_frac,
    )
    momentum_stage = (
        scale_by_muon_momentum(
            momentum_schedule if momentum_schedule is not None else (lambda _: momentum),
            nesterov,
            bimaxwell_switch_step,
            bimaxwell_rails,
        )
        if external_momentum
        else None
    )

    def _neuron_second_moment(x):
        if x is None or not hasattr(x, "ndim") or x.ndim < 2:
            return None
        return jnp.zeros(x.shape[:-2] + x.shape[-1:], jnp.float32)

    def core_init(params):
        muon_state = muon_transform.init(params)
        if neuron_norm_beta2 is None:
            return muon_state
        return muon_state, jax.tree.map(_neuron_second_moment, params)

    def init_fn(params):
        sphere = _spectral_sphere_init(params, spectral_radius_c) if spectral else None
        if momentum_stage is None:
            return core_init(params) if sphere is None else (core_init(params), sphere)
        magma = _magma_init(params) if magma_keep_prob is not None else None
        return MuonHState(momentum_stage.init(params), core_init(params), magma, sphere)

    def retract(params, directions, lr_mults, sphere):
        if sphere is None:
            updates = _scale_invariant_hyperball_updates(
                params, directions, learning_rate, hyperball_per_expert, lr_mults
            )
            return updates, None
        return _spectral_sphere_updates(params, directions, learning_rate, sphere, lr_mults)

    def core_direction(updates, state, params):
        if neuron_norm_beta2 is None:
            return muon_transform.update(updates, state, params)
        muon_state, second_moment = state
        muon_updates, muon_state = muon_transform.update(updates, muon_state, params)

        def second_moment_update(u, v):
            if u is None or v is None:
                return v
            mean_sq = jnp.mean(jnp.square(u.astype(jnp.float32)), axis=-2)
            return neuron_norm_beta2 * v + (1 - neuron_norm_beta2) * mean_sq

        def normalize(u, v):
            if u is None or v is None:
                return u
            return (u / (jnp.sqrt(v)[..., None, :] + 1e-10)).astype(u.dtype)

        none_leaf = lambda x: x is None  # noqa: E731
        second_moment = jax.tree.map(second_moment_update, muon_updates, second_moment, is_leaf=none_leaf)
        muon_updates = jax.tree.map(normalize, muon_updates, second_moment, is_leaf=none_leaf)
        return muon_updates, (muon_state, second_moment)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")
        if momentum_stage is None:
            core_state, sphere = state if spectral else (state, None)
            directions, core_state = core_direction(updates, core_state, params)
            muonh_updates, sphere = retract(params, directions, None, sphere)
            return muonh_updates, (core_state, sphere) if spectral else core_state
        mixed, momentum_state = momentum_stage.update(updates, state.momentum, params)
        directions, core_state = core_direction(mixed, state.core, params)
        lr_mults, magma_state = None, None
        if state.magma is not None:
            assert magma_keep_prob is not None
            lr_mults, magma_state = _magma_lr_mults(
                state.magma,
                updates,
                _muon_first_moment(momentum_state, bimaxwell_switch_step, bimaxwell_rails),
                keep_prob=magma_keep_prob,
                seed=magma_seed,
            )
        muonh_updates, sphere = retract(params, directions, lr_mults, state.sphere)
        return muonh_updates, MuonHState(momentum_state, core_state, magma_state, sphere)

    return optax.GradientTransformation(init_fn, update_fn)


class RowAdamState(NamedTuple):
    count: jax.Array
    nu: optax.Updates
    """One fp32 second moment per row, ``[..., rows, 1]``."""


def scale_by_row_adam(beta2: float, eps: float) -> optax.GradientTransformation:
    """Momentum-free Adam with one second moment per row (the hashed n-gram table's rule in modded-nanogpt
    record #360): ``v = beta2 v + (1 - beta2) mean_row(g^2)`` and direction ``sqrt(1 - beta2^t) g / (sqrt(v) + eps)``
    before the ``-lr`` scale. A row with zero gradient gets a zero update while its ``v`` decays, so the rule
    only needs the touched rows (this dense form is its loss reference). ``v`` is sliced from the parameter,
    so it keeps the table's row sharding."""

    def init_fn(params):
        nu = jax.tree.map(lambda p: jnp.zeros_like(p[..., :1], dtype=jnp.float32), params)
        return RowAdamState(count=jnp.zeros([], jnp.int32), nu=nu)

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_increment(state.count)
        nu = jax.tree.map(
            lambda g, v: beta2 * v + (1 - beta2) * jnp.mean(jnp.square(g.astype(jnp.float32)), -1, keepdims=True),
            updates,
            state.nu,
        )
        step_scale = jnp.sqrt(1 - beta2 ** count.astype(jnp.float32))
        direction = jax.tree.map(
            lambda g, v: (step_scale * g.astype(jnp.float32) / (jnp.sqrt(v) + eps)).astype(g.dtype), updates, nu
        )
        return direction, RowAdamState(count=count, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def cautious(inner: optax.GradientTransformation) -> optax.GradientTransformation:
    """Cautious optimizer (arXiv 2411.16085): keep only the coordinates of the inner direction whose sign
    agrees with the gradient, rescaled by the kept fraction per leaf. ``inner`` returns the descent
    direction before the ``-lr`` scale."""

    def update(updates, state, params=None):
        direction, state = inner.update(updates, state, params)

        def mask(u, g):
            keep = (u * g > 0).astype(u.dtype)
            return u * keep / jnp.maximum(jnp.mean(keep), 1e-3)

        return jax.tree.map(mask, direction, updates), state

    return optax.GradientTransformation(inner.init, update)


def _is_matrix_stack(x) -> bool:
    return hasattr(x, "ndim") and x.ndim >= 2


def cautious_matrix_deltas(inner: optax.GradientTransformation) -> optax.GradientTransformation:
    """Cautious masking (arXiv 2411.16085) for transforms that emit parameter deltas (MuonH): zero each matrix's
    update coordinates that move with the gradient (i.e. uphill on this batch), then rescale each matrix (the last two
    axes, so each expert separately) by its kept fraction."""

    def update(updates, state, params=None):
        deltas, state = inner.update(updates, state, params)

        def mask(d, g):
            if not (_is_matrix_stack(d) and _is_matrix_stack(g)):
                return d
            keep = (d.astype(jnp.float32) * g.astype(jnp.float32) < 0).astype(jnp.float32)
            kept = jnp.mean(keep, axis=(-2, -1), keepdims=True)
            return (d.astype(jnp.float32) * keep / jnp.maximum(kept, 1e-3)).astype(d.dtype)

        return jax.tree.map(mask, deltas, updates), state

    return optax.GradientTransformation(inner.init, update)


def retract_to_param_sphere(inner: optax.GradientTransformation, per_expert: bool) -> optax.GradientTransformation:
    """Put ``params + delta`` back on the parameter's Frobenius sphere, with the same spheres as the MuonH step (one
    per layer, or with ``per_expert`` one per (layer, expert) of a 4-D stack). Rescaling or masking a MuonH chord
    lands inside the sphere; without this the norm would shrink step after step."""

    def update(updates, state, params=None):
        if params is None:
            raise ValueError("retract_to_param_sphere requires params")
        deltas, state = inner.update(updates, state, params)

        def retract(p, d):
            if not _is_matrix_stack(p):
                return d
            axes = (-2, -1) if per_expert and p.ndim == 4 else tuple(range(1, p.ndim)) if p.ndim > 2 else (0, 1)
            p32 = p.astype(jnp.float32)
            moved = p32 + d.astype(jnp.float32)
            norm = jnp.sqrt(jnp.sum(jnp.square(p32), axis=axes, keepdims=True))
            moved_norm = jnp.sqrt(jnp.sum(jnp.square(moved), axis=axes, keepdims=True))
            return (moved * norm / jnp.maximum(moved_norm, 1e-10) - p32).astype(d.dtype)

        return jax.tree.map(retract, params, deltas), state

    return optax.GradientTransformation(inner.init, update)


class ExpertConsistencyState(NamedTuple):
    inner: optax.OptState
    count: jax.Array
    momentum: optax.Updates
    grad_sq: optax.Updates
    scale: optax.Updates
    """The last step's per-expert multiplier, for logging (``expert_consistency_metrics``)."""


def expert_consistency_metrics(opt_state) -> dict[str, jax.Array]:
    """Distribution of the routed experts' consistency multipliers (empty without ``scale_by_expert_consistency``)."""
    states = [
        x
        for x in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, ExpertConsistencyState))
        if isinstance(x, ExpertConsistencyState)
    ]
    # The multipliers keep the expert axis sharding; replicate them before flattening.
    scales = [
        jax.sharding.reshard(s, jax.sharding.PartitionSpec()).reshape(-1)
        for state in states
        for s in jax.tree.leaves(state.scale)
    ]
    if not scales:
        return {}
    flat = jnp.concatenate(scales)
    return {
        "train/optim/expert_consistency_mean": jnp.mean(flat),
        "train/optim/expert_consistency_p10": jnp.percentile(flat, 10),
        "train/optim/expert_consistency_p50": jnp.percentile(flat, 50),
        "train/optim/expert_consistency_p90": jnp.percentile(flat, 90),
        "train/optim/expert_consistency_zero_frac": jnp.mean((flat == 0).astype(jnp.float32)),
    }


def scale_by_expert_consistency(
    inner: optax.GradientTransformation, momentum: float, beta2: float
) -> optax.GradientTransformation:
    """Scale each matrix's update (per expert for ``[L, E, in, out]`` stacks) by how consistent its gradient is.

    With ``m`` an EMA of the gradient (``momentum``) and ``v`` an EMA of its squared Frobenius norm (``beta2``), both
    bias-corrected, ``r² = ||m||² / v`` is 1 for a gradient that repeats step to step and ``r₀² = (1 - momentum) /
    (1 + momentum)`` for pure noise. The update is multiplied by ``clip((r² - r₀²) / (1 - r₀²), 0, 1)``, so a matrix
    whose gradient keeps changing its mind barely moves. The direction is still the inner transform's.
    """
    floor = (1.0 - momentum) / (1.0 + momentum)

    def init_fn(params):
        def zeros_sq(p):
            return jnp.zeros((*p.shape[:-2], 1, 1), jnp.float32) if _is_matrix_stack(p) else None

        return ExpertConsistencyState(
            inner=inner.init(params),
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32) if _is_matrix_stack(p) else None, params),
            grad_sq=jax.tree.map(zeros_sq, params),
            scale=jax.tree.map(zeros_sq, params),
        )

    def update_fn(updates, state, params=None):
        count = state.count + 1
        is_leaf = lambda x: x is None  # noqa: E731

        def ema_m(m, g):
            return None if m is None else momentum * m + (1.0 - momentum) * g.astype(jnp.float32)

        def ema_v(v, g):
            if v is None:
                return None
            sq = jnp.sum(jnp.square(g.astype(jnp.float32)), axis=(-2, -1), keepdims=True)
            return beta2 * v + (1.0 - beta2) * sq

        momenta = jax.tree.map(ema_m, state.momentum, updates, is_leaf=is_leaf)
        grad_sq = jax.tree.map(ema_v, state.grad_sq, updates, is_leaf=is_leaf)
        m_corr = 1.0 - momentum ** count.astype(jnp.float32)
        v_corr = 1.0 - beta2 ** count.astype(jnp.float32)

        def scale(m, v):
            if m is None:
                return None
            m_sq = jnp.sum(jnp.square(m / m_corr), axis=(-2, -1), keepdims=True)
            r_sq = m_sq / jnp.maximum(v / v_corr, 1e-30)
            return jnp.clip((r_sq - floor) / (1.0 - floor), 0.0, 1.0)

        scales = jax.tree.map(scale, momenta, grad_sq, is_leaf=is_leaf)
        deltas, inner_state = inner.update(updates, state.inner, params)
        deltas = jax.tree.map(lambda d, c: d if c is None else (d * c).astype(d.dtype), deltas, scales, is_leaf=is_leaf)
        return deltas, ExpertConsistencyState(inner_state, count, momenta, grad_sq, scales)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_grad_power(power: float) -> optax.GradientTransformation:
    """GradPower (arXiv 2505.24275; Parameter Golf #1682): ``g <- sign(g) |g|^power`` elementwise."""

    def update(updates, state, params=None):
        return jax.tree.map(lambda g: jnp.sign(g) * jnp.abs(g) ** power, updates), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


def scale_by_mars_correction(gamma: float, momentum: float) -> optax.GradientTransformation:
    """MARS-M (arXiv 2510.21800) gradient correction for the Muon momentum:
    ``C = G + gamma * momentum / (1 - momentum) * (G - G_prev)``, clipped to Frobenius norm 1 per leaf."""
    coef = gamma * momentum / (1.0 - momentum)

    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update(updates, prev, params=None):
        def correct(g, p):
            c = g + coef * (g - p)
            norm = jnp.sqrt(jnp.sum(jnp.square(c.astype(jnp.float32))))
            return (c / jnp.maximum(norm, 1.0)).astype(g.dtype)

        return jax.tree.map(correct, updates, prev), updates

    return optax.GradientTransformation(init, update)


def snoo(
    inner: optax.GradientTransformation,
    labels_fn,
    *,
    period: int,
    outer_lr: float,
    outer_momentum: float,
) -> optax.GradientTransformation:
    """SNOO (arXiv 2510.15830): every ``period`` steps, a Nesterov step on the pseudo-gradient
    ``slow - fast`` moves the slow weights, and the fast weights jump to them. The inner optimizer
    state is never reset.

    Hyperball leaves (``_HYPERBALL_GROUPS`` in ``labels_fn(params)``) are projected back to their fast
    weights' per-layer Frobenius norm, which is the sphere MuonH/AdamH keep them on. ``router_bias`` is
    QB controller state set outside the optimizer, so it passes through untouched."""

    def init(params):
        return SnooState(
            count=jnp.zeros([], jnp.int32),
            slow=jax.tree.map(lambda p: p.astype(jnp.float32), params),
            momentum=jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params),
            inner=inner.init(params),
        )

    def update(updates, state, params):
        inner_updates, inner_state = inner.update(updates, state.inner, params)
        count = state.count + 1
        outer = count % period == 0
        labels = labels_fn(params)
        paths = leaf_key_paths(params)

        def leaf(u, p, slow, b, label, path):
            if "router_bias" in str(path).lower():
                return u, slow, b
            fast = p.astype(jnp.float32) + u.astype(jnp.float32)
            pseudo_grad = slow - fast
            b_new = outer_momentum * b + pseudo_grad
            slow_new = slow - outer_lr * (outer_momentum * b_new + pseudo_grad)
            if label in _HYPERBALL_GROUPS:
                slow_new = slow_new * _sphere_norm(fast) / jnp.maximum(_sphere_norm(slow_new), 1e-10)
            slow_new = _pin_sharding(slow_new, p)
            new_u = jnp.where(outer, slow_new - p.astype(jnp.float32), u.astype(jnp.float32)).astype(u.dtype)
            return new_u, jnp.where(outer, slow_new, slow), jnp.where(outer, b_new, b)

        out = jax.tree.map(leaf, inner_updates, params, state.slow, state.momentum, labels, paths)

        def pick(i):
            return jax.tree.map(lambda _, o: o[i], params, out)

        return pick(0), SnooState(count=count, slow=pick(1), momentum=pick(2), inner=inner_state)

    return optax.GradientTransformation(init, update)


class EmaNesterovState(NamedTuple):
    count: jax.Array
    scale: jax.Array
    """Lookahead coefficient ``s_t`` of the displacement ``s_t e_t`` currently in the parameters."""
    ema: optax.Updates
    inner: optax.OptState


def _matrix_norm(x: jax.Array) -> jax.Array:
    """Frobenius norm of every trailing ``[in, out]`` matrix (the whole array if 1-D), in float32."""
    axes = (-2, -1) if x.ndim >= 2 else (-1,)
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axes, keepdims=True))


def ema_nesterov(
    inner: optax.GradientTransformation,
    labels_fn,
    *,
    scale,
    decay: float,
    start_step: int,
    end_step: int,
    frobenius_groups: frozenset[str],
) -> optax.GradientTransformation:
    """EMA-Nesterov lookahead (arXiv 2605.25395; modded-nanogpt track-3 records #39/#40) around ``inner``.

    The record keeps an EMA ``e`` of the real weights' per-step displacement and evaluates the gradient at
    the lookahead point ``y_t = x_t + s_t e_t``; the inner step is taken from there, ``x_{t+1} = y_t + u_t``,
    so the displacement is ``x_{t+1} - x_t = s_t e_t + u_t`` and ``e_{t+1} = decay e_t + (1 - decay)
    (s_t e_t + u_t)``. The parameters here hold ``y``, so the gradient is taken at the lookahead point with
    no change to the train step, and each update is ``y_{t+1} - y_t = u_t + s_{t+1} e_{t+1}``. ``scale``
    (the caller passes ``s * lr / lr_peak``) is the next step's ``s_{t+1}``, applied for
    ``start_step <= t + 1 < end_step``; outside the window ``y = x``, so a run ending past ``end_step``
    ends (and is evaluated) on the real weights.

    Leaves labelled in ``frobenius_groups`` (the Frobenius hyperball groups) are rescaled so every matrix
    keeps the Frobenius norm the inner step gave it; spectral-sphere leaves are retracted by their next
    MuonSphere step. ``router_bias`` is QB controller state and passes through."""

    def init(params):
        return EmaNesterovState(
            count=jnp.zeros([], jnp.int32),
            scale=jnp.zeros([], jnp.float32),
            ema=jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params),
            inner=inner.init(params),
        )

    def update(updates, state, params):
        inner_updates, inner_state = inner.update(updates, state.inner, params)
        count = state.count + 1
        next_scale = jnp.where((count >= start_step) & (count < end_step), scale, 0.0).astype(jnp.float32)
        labels = labels_fn(params)
        paths = leaf_key_paths(params)

        def leaf(u, p, e, label, path):
            if "router_bias" in str(path).lower():
                return u, e
            u32 = u.astype(jnp.float32)
            e_new = decay * e + (1.0 - decay) * (state.scale * e + u32)
            out = u32 + next_scale * e_new
            if label in frobenius_groups:
                p32 = p.astype(jnp.float32)
                target = _matrix_norm(_pin_sharding(p32 + u32, p))
                new_param = _pin_sharding(p32 + out, p)
                out = new_param * (target / jnp.maximum(_matrix_norm(new_param), 1e-10)) - p32
            return out.astype(u.dtype), e_new

        res = jax.tree.map(leaf, inner_updates, params, state.ema, labels, paths)
        pick = lambda i: jax.tree.map(lambda _, o: o[i], params, res)  # noqa: E731
        return pick(0), EmaNesterovState(count=count, scale=next_scale, ema=pick(1), inner=inner_state)

    return optax.GradientTransformation(init, update)


def optimizer_diagnostics(opt_state) -> dict[str, jax.Array]:
    """``train/optim/`` metrics of the EMA-Nesterov, MuonSphere and row-Adam states found in ``opt_state``."""

    def find(state_type):
        # Searched per type: an EmaNesterovState wraps the MuonSphere states inside its ``inner``.
        is_state = lambda x: isinstance(x, state_type)  # noqa: E731
        return [x for x in jax.tree.leaves(opt_state, is_leaf=is_state) if is_state(x)]

    metrics = {}
    ema_states = find(EmaNesterovState)
    if ema_states:
        ema = ema_states[0]
        ema_norm = jnp.sqrt(sum(jnp.sum(jnp.square(e)) for e in jax.tree.leaves(ema.ema)))
        metrics["train/optim/ema_nesterov_scale"] = ema.scale
        metrics["train/optim/ema_nesterov_ema_norm"] = ema_norm
        metrics["train/optim/ema_nesterov_lookahead_norm"] = ema.scale * ema_norm
    sphere_states = find(SpectralSphereState)
    if sphere_states:
        # sigma_1 / R before each step's retraction: the drift the retraction removes.
        pairs = [
            (s / r, r)
            for st in sphere_states
            for s, r in zip(jax.tree.leaves(st.sigma), jax.tree.leaves(st.radius), strict=True)
        ]
        count = sum(r.size for _, r in pairs)
        metrics["train/optim/spectral_sigma_over_radius_mean"] = sum(jnp.sum(q) for q, _ in pairs) / count
        metrics["train/optim/spectral_sigma_over_radius_max"] = jnp.max(jnp.stack([jnp.max(q) for q, _ in pairs]))
        metrics["train/optim/spectral_radius_mean"] = sum(jnp.sum(r) for _, r in pairs) / count
    row_states = find(RowAdamState)
    if row_states:
        # Per-row RMS gradient scale of the n-gram tables, and the fraction of rows ever touched.
        nus = jax.tree.leaves(row_states[0].nu)
        rows = sum(v.size for v in nus)
        metrics["train/optim/embed2_row_rms_mean"] = sum(jnp.sum(jnp.sqrt(v)) for v in nus) / rows
        metrics["train/optim/embed2_row_touched_frac"] = sum(jnp.sum(v > 0) for v in nus) / rows
    return metrics


@OptimizerConfig.register_subclass("grug_fast_track_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """MuonH optimizer for the EP MoE model. Three LR groups (muonh / adamh / adam):

    - ``muonh``: matrix leaves (attn, MoE MLP, shared) and GatedNorms -- Newton-Schulz
      orthogonalization + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``output_proj`` / ``lm_head``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate`` / 1-D norm gains
      and the tiny SConv kernels.

    The KMA variant adds the Inkling rel-pos weights and the KDA gate / decay parameters to ``adam``,
    and two groups: ``attn_res_query`` (AttnRes pseudo-queries, Adam at ``attn_res_query_lr_scale`` x
    ``adam_lr``) and ``kda_beta`` (the KDA write-strength projection, MuonH at ``kda_beta_lr_mult`` x
    the MuonH LR).
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"
    gate_router_weight_decay: float = 0.02
    gain_weight_decay: float = 0.0
    """Decoupled weight decay (annealed like ``gate_router_weight_decay``) on the zero-centered norm gains'
    ``gamma`` (``zero_centered_gains``), pulling each gain toward 1."""
    attn_res_query_lr_scale: float = 0.1
    kda_beta_lr_mult: float = 2.0
    muon_head_dim: int | None = None
    kda_beta_mlp_group: str = "kda_beta"
    """LR group of the low-rank KDA beta MLP: ``kda_beta`` (MuonH at ``kda_beta_lr_mult``) or ``adam``."""
    kda_decay_lr_mult: float = 1.0
    """Adam LR multiplier for the KDA decay parameters (``_KDA_ADAM_LEAVES``: dt_bias, A_log, the gate projections)."""
    kda_decay_beta1: float | None = None
    """Adam beta1 for the KDA decay parameters (None: ``beta1``)."""
    kda_decay_beta2: float | None = None
    """Adam beta2 for the KDA decay parameters (None: ``beta2``)."""
    muonh_attn_lr_mult: float = 1.0
    """MuonH LR multiplier for the attention-projection family (``_OKLS_FAMILIES['attn']``)."""
    muonh_routed_lr_mult: float = 1.0
    """MuonH LR multiplier for the routed-expert family (``_OKLS_FAMILIES['routed']``)."""
    muonh_qk_lr_mult: float = 1.0
    """MuonH LR multiplier for the query/key projections (``_QK_PROJECTIONS``: KDA ``w_q``/``w_k``, MLA
    ``w_q``/``w_uk``). Tests whether q/k norms make those matrices want a gentler or a bolder step."""
    muonh_qk_momentum: float | None = None
    """MuonH momentum for the query/key projections (None: ``momentum``)."""
    muonh_routed_momentum: float | None = None
    muon_truncate_family: str | None = None
    """Matrix type (``_TRUNCATE_FAMILIES``) whose MuonH updates drop their weakest ``muon_truncate_frac`` of
    singular directions (``_truncate_bottom_directions``). None: no truncation."""
    muon_truncate_frac: float = 0.0
    eig_families: tuple[str, ...] = ()
    """Matrix types (``_TRUNCATE_FAMILIES``) whose MuonH direction is replaced by an eigenbasis variant
    (``eig_muon.scale_by_eig_direction``, mode ``eig_mode``), still taken as a hyperball step."""
    eig_mode: str = "snr"
    eig_lr_mult: float = 1.0
    eig_beta: float = 0.95
    eig_beta2: float = 0.99
    """Second-moment β of ``eig_mode="soap"`` (the other modes use ``eig_beta`` for both moments)."""
    eig_factor_beta: float = 0.95
    eig_refresh_every: int = 10
    eig_whiten_power: float = 0.25
    bimaxwell_slow_from_start: bool = False
    """Every MuonH group's Bi-Maxwell slow rail accumulates from step 0 (``BiMaxwellRails.slow_from_start``)."""
    muonh_routed_slow_rate: float | None = None
    """The routed experts' Bi-Maxwell slow-rail EMA rate (None: ``BiMaxwellRails``'s default, 0.02)."""
    muonh_routed_slow_weight: float | None = None
    """The routed experts' Bi-Maxwell slow-rail blend weight (None: the default, 0.5615)."""
    muonh_routed_bimaxwell_start_step: int | None = None
    """Step at which the routed experts' Bi-Maxwell rails engage (None: ``bimaxwell_start_frac`` of training)."""
    muonh_routed_consistency_beta2: float | None = None
    """Scale each routed expert's MuonH update by its gradient consistency (``scale_by_expert_consistency``), with
    this beta2 for the squared-norm EMA. None: off."""
    muonh_routed_cautious: bool = False
    """Cautious-mask the routed experts' MuonH updates per expert (``cautious_matrix_deltas``)."""
    routed_expert_optimizer: str = "muonh"
    """``muonh`` or ``adam``: which group trains the routed expert matrices."""
    """MuonH momentum for the routed-expert family (None: ``momentum``). Each expert sees ~1/64 of the tokens, so its
    per-step gradient is mostly noise; a longer average may suit it better than the dense matrices."""
    okls_targets: tuple[str, ...] = ()
    """Matrix families (``_OKLS_FAMILIES``) whose direction comes from Online KL-Shampoo whitening instead
    of Newton-Schulz, still taking MuonH's hyperball step at the MuonH LR."""
    okls_beta1: float = 0.9684
    okls_beta2: float = 0.9482
    okls_epsilon: float = 1e-9
    okls_cans_steps: int = 10
    okls_matmul_dtype: str = "float32"
    okls_lr_mult: float = 1.0
    """LR multiplier of the OKLS group relative to the MuonH LR (hyperball mode)."""
    okls_hyperball: bool = True
    """True: OKLS direction + MuonH's norm-preserving hyperball step at the MuonH LR. False: the paper's
    own update (muP shape scale, Nesterov variance correction, AdamC decoupled weight decay) at
    ``okls_peak_lr`` on the same schedule shape."""
    okls_peak_lr: float = 0.09434
    """Paper-mode OKLS peak LR (the release's muP-scaled default)."""
    okls_weight_decay: float = 0.0303
    """Paper-mode AdamC decoupled weight decay."""
    okls_root_every: int = 1
    okls_input_damping: float = 0.0
    """Damping lambda on OKLS's input-side factor before its root (``okls._okls_core_2d``): 0 is plain OKLS, a
    large value leaves output-side-only whitening. Our input-side factors have 1-12% effective rank."""
    latent_proj_update: str = "muonh"
    """How the LatentMoE projections (``w_latent_down`` / ``w_latent_up``) train: ``muonh`` (like every matrix),
    ``frozen`` (kept at init), or ``stiefel`` (Skewon, ``stiefel.py``: stays at its scaled semi-orthogonal init
    point; needs the model's ``latent_orthogonal_init``)."""
    """Recompute the OKLS inverse roots every this many steps (stored in between)."""
    lm_head_group: str = "adamh"
    """LR group of ``output_proj``: ``adamh``, ``muonh`` or ``sinkhornh`` (Sinkhorn-balanced momentum + hyperball)."""
    embed_group: str = "adam"
    embed2_lr_mult: float = 1.0
    """Adam LR multiplier of the second (e.g. bigram) embedding table alone."""
    embed2_row_sparse_adam: bool = False
    """Train the second / third tables with ``scale_by_row_adam`` (beta1 = 0, one fp32 second moment per row, no
    AdEMAMix / cautious masking / grokfast) at ``embed2_lr_mult`` x ``adam_lr``: record #360's row-sparse-compatible
    rule, run densely. Cuts the tables' optimizer state from two (three with AdEMAMix) copies to ``rows`` floats."""
    embed2_beta2: float = 0.95
    """Second-moment decay of ``embed2_row_sparse_adam``."""
    embed2_update: str = "adam"
    """Update rule for the bigram table: ``adam`` (the Adam groups' AdEMAMix) or ``sinkhorn`` (Sinkhorn momentum,
    one buffer, at ``embed2_lr_mult * sinkhorn_lr_mult`` times the Adam LR)."""
    sinkhorn_momentum: float = 0.95
    sinkhorn_iters: int = 5
    sinkhorn_nesterov: bool = True
    sinkhorn_lr_mult: float = 1.0
    """LR multiplier of the Sinkhorn groups: ``sinkhorn`` (embedding, at the Adam LR) and ``sinkhornh`` (lm_head,
    hyperball at the MuonH LR, like AdamH)."""
    """LR group of ``token_embed``: ``adam`` or ``adamh``."""
    muon_free_families: tuple[str, ...] = ()
    """Matrix families (keys of ``_OKLS_FAMILIES``, e.g. ``attn_q``, ``attn_k``) that drop the hyperball:
    MuonH's step size with a free norm and ``muon_free_weight_decay`` (``scale_with_grug_muon_free``)."""
    muon_free_weight_decay: float = 0.0
    hyperball_per_expert: bool = False
    """One MuonH hyperball (Frobenius sphere) per routed expert instead of per layer's expert stack."""
    neuron_norm_beta2: float | None = None
    """NorMuon neuron-wise normalization of the MuonH direction with this second-moment decay (None: off)."""
    """Orthogonalize the attention projections per head of this width (None: whole matrices)."""
    muon_pre_norm: str = "none"
    """MuonH momentum normalization before Newton-Schulz: ``none``, ``out`` or ``in`` (MuonEq)."""
    muon_top_shrink: float = 0.0
    """SAMuon-lite: remove this fraction of the top singular direction from the MuonH direction (``1 - 1/gamma``)."""
    muon_precond_beta2: float | None = None
    """Muon2: Adam second-moment preconditioning of the MuonH momentum before Newton-Schulz (None: off)."""
    muonh_decay_power: float | None = None
    """Decay shape of the MuonH LR after warmup: ``floor + (peak - floor) * (1 - p**power)`` over training
    progress ``p`` (modded-nanogpt MuonH records #345/#351). None keeps the shared schedule (linear = 1)."""
    muon_bimaxwell: bool = False
    bimaxwell_start_frac: float = 1 / 3
    """Fraction of training after which Bi-Maxwell's two-timescale momentum replaces Nesterov momentum."""
    """Bi-Maxwell two-timescale momentum on the MuonH groups from 1/3 of training (modded-nanogpt #339)."""
    adam_cautious: bool = False
    """Cautious masking (arXiv 2411.16085) on the plain-Adam groups."""
    muon_grad_power: float = 1.0
    """GradPower exponent on the MuonH-group gradients before momentum (1: off; Parameter Golf #1682 used 0.9)."""
    muon_mars_gamma: float = 0.0
    """MARS-M variance-reduction strength for the MuonH groups (0: off; the paper uses 0.025)."""
    adam_ademamix_alpha: float = 0.0
    """AdEMAMix (arXiv 2409.03152) on the plain-Adam groups: weight of a slow gradient EMA added to Adam's
    first moment, warmed up linearly over the run (0: off; the paper uses 5-8)."""
    adam_ademamix_beta3: float = 0.999
    """Decay of the AdEMAMix slow EMA, warmed up over ``adam_ademamix_warmup`` as in the paper."""
    adam_ademamix_warmup: float = 1.0
    """Fraction of the run over which AdEMAMix's alpha and beta3 warm up (the paper: the whole run)."""
    adam_ademamix_cooldown: float = 0.0
    """Fraction at the end of the run over which alpha decays linearly back to 0 (0: none)."""
    grokfast_lambda: float = 0.0
    """Grokfast-EMA (arXiv 2405.20233): every gradient gets ``lambda`` x its EMA added before the optimizer
    (0: off; the paper uses 2)."""
    grokfast_alpha: float = 0.98
    value_embed_lr_mult: float = 1.0
    """Adam LR multiplier of the value-embedding tables (``value_embeds``); like the bigram table, a sparse
    token table may want a hotter LR than the dense Adam leaves."""
    ple_lr_mult: float = 1.0
    """Adam LR multiplier of the per-layer embedding table (``ple_dim``)."""
    memory_lr_mult: float = 1.0
    """Adam LR multiplier of the product-key memory value tables (``memory_layers``)."""
    output_bigram_lr_mult: float = 1.0
    """Adam LR multiplier of the output bigram prior's ``U`` and ``W`` (``output_bigram_rank``)."""
    grokfast_adam_only: bool = False
    """Apply Grokfast to the plain-Adam groups only (on MuonH it acts as extra momentum)."""
    router_group: str = "adam"
    """Update rule of the MoE router weights: ``adam`` or ``muonh`` (Newton-Schulz + hyperball per layer, as the
    routers of DeepSeek-V4 arXiv 2606.19348 and Kimi K3 are on Muon). MuonH pins each router's Frobenius norm,
    and with it the logit temperature; pair it with the model's ``router_logit_scale``."""
    router_lr_mult: float = 1.0
    """LR multiplier of the router weights alone (of ``adam_lr`` or, with ``router_group=muonh``, the MuonH LR)."""
    router_weight_decay: float | None = None
    """Decoupled weight decay of the Adam router weights alone (annealed like ``gate_router_weight_decay``);
    None: the shared ``gate_router_weight_decay``."""
    snoo_period: int = 0
    """SNOO outer step every this many inner steps (0: off)."""
    snoo_lr: float = 0.5
    snoo_momentum: float = 0.5
    ema_nesterov: bool = False
    """EMA-Nesterov lookahead (arXiv 2605.25395; modded-nanogpt track-3 #40) around the whole optimizer:
    the gradient is taken at ``W + s e`` with ``e`` an EMA of the weight displacement (``ema_nesterov``)."""
    ema_nesterov_scale: float = 0.3
    """Peak lookahead ``s``, scheduled as ``s * lr / lr_peak`` (the MuonH LR); record #40 uses 0.3."""
    ema_nesterov_decay: float = 0.99
    ema_nesterov_start_frac: float = 300 / 2900
    ema_nesterov_end_frac: float = 1950 / 2900
    """Lookahead window as fractions of training (record #40: steps 300 to 1950 of 2900)."""
    muonh_retraction: str = "frobenius"
    """MuonH retraction: ``frobenius`` (hyperball) or ``spectral`` (MuonSphere, arXiv 2601.08393: spectral
    sphere per matrix, ``lr`` the relative step in the spectral norm)."""
    spectral_radius_c: float | None = 2.0
    """MuonSphere radius ``R = c sqrt(fan_out / fan_in)`` (the paper's best c is 2); None keeps each matrix's
    initial spectral norm."""
    magma: bool = False
    """Magma (arXiv 2602.15322) on every MuonH group: each matrix (stacked slice) keeps its hyperball step
    with probability ``magma_keep_prob``, scaled by an EMA(0.9) of ``sigmoid(cos(momentum, grad) / 2)``.
    The mean step shrinks to about ``0.3x``; the paper keeps the base LR."""
    magma_keep_prob: float = 0.5
    muon_momentum_warmup_steps: int = 0
    """Ramp the MuonH momentum linearly from ``muon_momentum_warmup_start`` to ``momentum`` over this many
    steps (modded-nanogpt; 0: off). Bi-Maxwell uses the ramped value too, so a warmup longer than
    ``bimaxwell_start_frac`` of training carries into its mix."""
    muon_momentum_warmup_start: float = 0.85
    cautious_weight_decay: bool = False
    """Cautious weight decay (arXiv 2510.12402) on the Adam group's decoupled decay (router / attn_gate, and
    the gains with ``gain_weight_decay``): decay only where the update and the parameter agree in sign."""
    upper_qk_lr_mult: float = 1.0
    """Early-training LR multiplier on the upper-layer softmax-attention q/k projections (arXiv 2605.10504:
    0.25; 1: off). Held until ``upper_qk_slow_frac`` of training, then ramped linearly to 1 over
    ``upper_qk_ramp_frac``."""
    upper_qk_slow_frac: float = 0.25
    """Release point of ``upper_qk_lr_mult`` (the paper's maturity rule fires at 3-6%; fixed 3% keeps most of it)."""
    upper_qk_ramp_frac: float = 0.01
    upper_qk_slice_mask: tuple[bool, ...] = ()
    """Per ``stacked_blocks`` slice, whether that softmax layer is in the upper half (``layer >= num_layers // 2``);
    filled from the model config by the launcher (``upper_softmax_slice_mask``)."""

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        if self.muonh_decay_power is not None:
            learning_rate_schedule = _power_decay_schedule(
                self.learning_rate,
                self.learning_rate * self.min_lr_ratio,
                _convert_frac_or_steps(self.warmup, num_train_steps),
                num_train_steps,
                self.muonh_decay_power,
            )
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        slow_upper_qk = self.upper_qk_lr_mult != 1.0
        if slow_upper_qk and not any(self.upper_qk_slice_mask):
            raise ValueError("upper_qk_lr_mult needs upper_qk_slice_mask (no upper softmax layer given)")
        momentum_schedule = (
            _momentum_warmup_schedule(self.muon_momentum_warmup_start, self.momentum, self.muon_momentum_warmup_steps)
            if self.muon_momentum_warmup_steps > 0
            else None
        )

        def optimizer(learning_rate, adam_lr, upper_qk_mult=None):
            default_switch = int(self.bimaxwell_start_frac * num_train_steps) if self.muon_bimaxwell else None

            def muonh_transform_at(
                lr,
                magma_seed: int,
                momentum: float | None = None,
                rails: BiMaxwellRails | None = None,
                switch_step: int | None = None,
                truncate_frac: float = 0.0,
            ):
                momentum = self.momentum if momentum is None else momentum
                switch_step = default_switch if switch_step is None else switch_step
                rails = self._base_rails() if rails is None else rails
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                if self.muon_grad_power != 1.0:
                    components.append(scale_by_grad_power(self.muon_grad_power))
                if self.muon_mars_gamma:
                    components.append(scale_by_mars_correction(self.muon_mars_gamma, momentum))
                components.append(
                    scale_with_grug_muonh(
                        momentum=momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=lr,
                        coefficient_type=self.coefficient_type,
                        head_dim=self.muon_head_dim,
                        neuron_norm_beta2=self.neuron_norm_beta2,
                        hyperball_per_expert=self.hyperball_per_expert,
                        pre_norm=self.muon_pre_norm,
                        top_shrink=self.muon_top_shrink,
                        precond_beta2=self.muon_precond_beta2,
                        truncate_frac=truncate_frac,
                        retraction=self.muonh_retraction,
                        spectral_radius_c=self.spectral_radius_c,
                        momentum_schedule=momentum_schedule,
                        bimaxwell_switch_step=switch_step,
                        bimaxwell_rails=rails,
                        magma_keep_prob=self.magma_keep_prob if self.magma else None,
                        magma_seed=magma_seed,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                return optax.chain(*components)

            def adam_core(beta1, beta2):
                core = adam_moments(beta1, beta2)
                if self.grokfast_lambda and self.grokfast_adam_only:
                    return optax.chain(scale_by_grokfast_ema(self.grokfast_alpha, self.grokfast_lambda), core)
                return core

            def adam_moments(beta1, beta2):
                if not self.adam_ademamix_alpha:
                    return optax.scale_by_adam(beta1, beta2, self.epsilon)
                warmup = max(1, int(self.adam_ademamix_warmup * num_train_steps))
                cooldown = int(self.adam_ademamix_cooldown * num_train_steps)
                alpha = optax.join_schedules(
                    [
                        optax.linear_schedule(0.0, self.adam_ademamix_alpha, warmup),
                        optax.constant_schedule(self.adam_ademamix_alpha),
                        optax.linear_schedule(self.adam_ademamix_alpha, 0.0, cooldown),
                    ],
                    [warmup, max(warmup, num_train_steps - cooldown)],
                )
                ademamix = optax.contrib.scale_by_ademamix(
                    beta1,
                    beta2,
                    _ademamix_beta3_schedule(beta1, self.adam_ademamix_beta3, warmup),
                    alpha,
                    self.epsilon,
                )
                return ademamix

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                adam = adam_core(self.beta1, self.beta2)
                if self.gate_router_weight_decay > 0.0 or self.gain_weight_decay > 0.0:
                    adam = _scale_by_adam_decoupled_decay(
                        adam,
                        self.gate_router_weight_decay,
                        self.gain_weight_decay,
                        num_train_steps,
                        cautious_decay=self.cautious_weight_decay,
                    )
                components.append(cautious(adam) if self.adam_cautious else adam)
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            def plain_adam_at(lr, beta1=None, beta2=None):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                adam = adam_core(self.beta1 if beta1 is None else beta1, self.beta2 if beta2 is None else beta2)
                components.append(cautious(adam) if self.adam_cautious else adam)
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            def row_adam_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_row_adam(self.embed2_beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            def router_adam_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                adam = adam_core(self.beta1, self.beta2)
                decay = self.gate_router_weight_decay if self.router_weight_decay is None else self.router_weight_decay
                if decay > 0.0:
                    adam = _scale_by_adam_decoupled_decay(adam, decay, 0.0, num_train_steps)
                components.append(cautious(adam) if self.adam_cautious else adam)
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform_at(learning_rate, 0),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
                "attn_res_query": plain_adam_at(adam_lr * self.attn_res_query_lr_scale),
                "kda_beta": muonh_transform_at(learning_rate * self.kda_beta_lr_mult, 1),
                "okls": optax.chain(
                    scale_with_grug_okls(
                        beta1=self.okls_beta1,
                        beta2=self.okls_beta2,
                        eps=self.okls_epsilon,
                        weight_decay=0.0 if self.okls_hyperball else self.okls_weight_decay,
                        cans_steps=self.okls_cans_steps,
                        matmul_dtype=OKLS_MATMUL_DTYPES[self.okls_matmul_dtype],
                        # Paper mode rescales the MuonH schedule to its own peak (same warmup/decay shape).
                        learning_rate=learning_rate
                        * (self.okls_lr_mult if self.okls_hyperball else self.okls_peak_lr / self.learning_rate),
                        lr_peak=self.learning_rate * self.okls_lr_mult if self.okls_hyperball else self.okls_peak_lr,
                        hyperball=self.okls_hyperball,
                        root_every=self.okls_root_every,
                        input_damping=self.okls_input_damping,
                    ),
                    _match_named_update_sharding(),
                ),
                "muonh_attn": muonh_transform_at(learning_rate * self.muonh_attn_lr_mult, 2),
                "muonh_trunc": muonh_transform_at(learning_rate, 5, truncate_frac=self.muon_truncate_frac),
                "eig": optax.chain(
                    _eig_hyperball(
                        scale_by_eig_direction(
                            self.eig_mode,
                            beta=self.eig_beta,
                            beta2=self.eig_beta2,
                            factor_beta=self.eig_factor_beta,
                            refresh_every=self.eig_refresh_every,
                            whiten_power=self.eig_whiten_power,
                            ns_steps=self.backend_steps,
                        ),
                        learning_rate * self.eig_lr_mult,
                        self.hyperball_per_expert,
                    ),
                    _match_named_update_sharding(),
                ),
                "muonh_qk": muonh_transform_at(
                    learning_rate * self.muonh_qk_lr_mult, 4, momentum=self.muonh_qk_momentum
                ),
                "muonh_routed": self._routed_transform(
                    muonh_transform_at(
                        learning_rate * self.muonh_routed_lr_mult,
                        3,
                        momentum=self.muonh_routed_momentum,
                        rails=self._routed_rails(),
                        switch_step=self.muonh_routed_bimaxwell_start_step,
                    )
                ),
                "muon_free": optax.chain(
                    scale_with_grug_muon_free(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        head_dim=self.muon_head_dim,
                        weight_decay=self.muon_free_weight_decay,
                    ),
                    _match_named_update_sharding(),
                ),
                "sinkhorn": optax.chain(
                    scale_by_sinkhorn_momentum(
                        momentum=self.sinkhorn_momentum, iters=self.sinkhorn_iters, nesterov=self.sinkhorn_nesterov
                    ),
                    optax.scale(-adam_lr * self.sinkhorn_lr_mult),
                ),
                "sinkhornh": _sinkhorn_hyperball(
                    self.sinkhorn_momentum,
                    self.sinkhorn_iters,
                    self.sinkhorn_nesterov,
                    learning_rate * self.sinkhorn_lr_mult,
                ),
                "embed2": (
                    row_adam_at(adam_lr * self.embed2_lr_mult)
                    if self.embed2_row_sparse_adam
                    else (
                        optax.chain(
                            scale_by_sinkhorn_momentum(
                                momentum=self.sinkhorn_momentum,
                                iters=self.sinkhorn_iters,
                                nesterov=self.sinkhorn_nesterov,
                            ),
                            optax.scale(-adam_lr * self.embed2_lr_mult * self.sinkhorn_lr_mult),
                        )
                        if self.embed2_update == "sinkhorn"
                        else plain_adam_at(adam_lr * self.embed2_lr_mult)
                    )
                ),
                # The n-gram statistic table and its code are data statistics written by the trainer, not trained.
                "frozen": optax.set_to_zero(),
                "stiefel": scale_with_stiefel_muon(
                    momentum=self.momentum, nesterov=self.nesterov, learning_rate=learning_rate
                ),
                "ple": plain_adam_at(adam_lr * self.ple_lr_mult),
                "value_embed": plain_adam_at(adam_lr * self.value_embed_lr_mult),
                "memory": plain_adam_at(adam_lr * self.memory_lr_mult),
                "output_bigram": plain_adam_at(adam_lr * self.output_bigram_lr_mult),
                "kda_decay": plain_adam_at(adam_lr * self.kda_decay_lr_mult, self.kda_decay_beta1, self.kda_decay_beta2),
                "router": router_adam_at(adam_lr * self.router_lr_mult),
                "muonh_router": muonh_transform_at(learning_rate * self.router_lr_mult, 5),
            }
            if slow_upper_qk:
                # One LR per stacked slice: the upper softmax layers take the scheduled multiplier.
                upper = jnp.asarray(self.upper_qk_slice_mask)
                slice_lr = jnp.where(upper, upper_qk_mult, 1.0)[:, None, None]
                transforms["upper_qk"] = muonh_transform_at(learning_rate * self.muonh_attn_lr_mult * slice_lr, 4)
            inner = optax.multi_transform(transforms, self.create_mask)
            if self.grokfast_lambda and not self.grokfast_adam_only:
                inner = optax.chain(scale_by_grokfast_ema(self.grokfast_alpha, self.grokfast_lambda), inner)
            if self.snoo_period > 0:
                inner = snoo(
                    inner,
                    self.create_mask,
                    period=self.snoo_period,
                    outer_lr=self.snoo_lr,
                    outer_momentum=self.snoo_momentum,
                )
            if not self.ema_nesterov:
                return inner
            spectral = _MUONH_GROUPS if self.muonh_retraction == "spectral" else frozenset()
            okls = frozenset({"okls"}) if self.okls_hyperball else frozenset()
            return ema_nesterov(
                inner,
                self.create_mask,
                scale=self.ema_nesterov_scale * learning_rate / self.learning_rate,
                decay=self.ema_nesterov_decay,
                start_step=round(self.ema_nesterov_start_frac * num_train_steps),
                end_step=round(self.ema_nesterov_end_frac * num_train_steps),
                frobenius_groups=(_HYPERBALL_GROUPS | okls) - spectral,
            )

        schedules = {"learning_rate": learning_rate_schedule, "adam_lr": adam_lr_schedule}
        if slow_upper_qk:
            schedules["upper_qk_mult"] = _upper_qk_schedule(
                self.upper_qk_lr_mult,
                int(self.upper_qk_slow_frac * num_train_steps),
                max(1, int(self.upper_qk_ramp_frac * num_train_steps)),
            )
        return optax.inject_hyperparams(optimizer)(**schedules)

    def _base_rails(self) -> BiMaxwellRails:
        return dataclasses.replace(DEFAULT_RAILS, slow_from_start=self.bimaxwell_slow_from_start)

    def _routed_rails(self) -> BiMaxwellRails:
        rails = self._base_rails()
        if self.muonh_routed_slow_rate is not None:
            rails = dataclasses.replace(rails, slow_rate=self.muonh_routed_slow_rate)
        if self.muonh_routed_slow_weight is not None:
            rails = dataclasses.replace(rails, slow_weight=self.muonh_routed_slow_weight)
        return rails

    def _routed_transform(self, inner: optax.GradientTransformation) -> optax.GradientTransformation:
        if self.muonh_routed_cautious:
            inner = cautious_matrix_deltas(inner)
        if self.muonh_routed_consistency_beta2 is not None:
            momentum = self.momentum if self.muonh_routed_momentum is None else self.muonh_routed_momentum
            inner = scale_by_expert_consistency(inner, momentum, self.muonh_routed_consistency_beta2)
        if self.muonh_routed_cautious or self.muonh_routed_consistency_beta2 is not None:
            inner = retract_to_param_sphere(inner, self.hyperball_per_expert)
        return inner

    def __post_init__(self):
        if self.muon_truncate_family is not None and self.muon_truncate_family not in _TRUNCATE_FAMILIES:
            raise ValueError(f"muon_truncate_family must be one of {sorted(_TRUNCATE_FAMILIES)}")
        if (self.muon_truncate_family is None) != (
            self.muon_truncate_frac == 0.0
        ) or not 0.0 <= self.muon_truncate_frac < 1:
            raise ValueError("muon_truncate_family and a muon_truncate_frac in (0, 1) go together")
        unknown_eig = set(self.eig_families) - set(_TRUNCATE_FAMILIES)
        if unknown_eig:
            raise ValueError(f"unknown eig_families {sorted(unknown_eig)}; choose from {sorted(_TRUNCATE_FAMILIES)}")
        if self.eig_mode not in EIG_MODES:
            raise ValueError(f"eig_mode must be one of {EIG_MODES}, got {self.eig_mode!r}")
        if self.routed_expert_optimizer not in ("muonh", "adam"):
            raise ValueError(f"routed_expert_optimizer must be muonh or adam, got {self.routed_expert_optimizer!r}")
        if self.embed2_update not in ("adam", "sinkhorn"):
            raise ValueError(f"embed2_update must be adam or sinkhorn, got {self.embed2_update!r}")
        if self.embed2_update == "sinkhorn" and self.embed2_row_sparse_adam:
            raise ValueError("embed2_update=sinkhorn and embed2_row_sparse_adam are exclusive")
        if self.latent_proj_update not in LATENT_PROJ_UPDATES:
            raise ValueError(f"latent_proj_update must be one of {LATENT_PROJ_UPDATES}, got {self.latent_proj_update!r}")
        if self.lm_head_group not in ("adamh", "muonh", "sinkhornh"):
            raise ValueError(f"lm_head_group must be adamh, muonh or sinkhornh, got {self.lm_head_group!r}")
        if self.kda_beta_mlp_group not in ("kda_beta", "adam"):
            raise ValueError(f"kda_beta_mlp_group must be kda_beta or adam, got {self.kda_beta_mlp_group!r}")
        if self.embed_group not in ("adam", "adamh", "sinkhorn"):
            raise ValueError(f"embed_group must be adam, adamh or sinkhorn, got {self.embed_group!r}")
        if self.router_group not in _ROUTER_GROUPS:
            raise ValueError(f"router_group must be one of {_ROUTER_GROUPS}, got {self.router_group!r}")
        if self.router_group == "muonh" and self.router_weight_decay is not None:
            raise ValueError("router_weight_decay applies to the Adam router; MuonH keeps the router norm fixed")
        if self.muonh_retraction not in MUONH_RETRACTIONS:
            raise ValueError(f"muonh_retraction must be one of {MUONH_RETRACTIONS}, got {self.muonh_retraction!r}")

    def create_mask(self, params):
        paths = leaf_key_paths(params)
        unknown = (set(self.okls_targets) | set(self.muon_free_families)) - set(_OKLS_FAMILIES)
        if unknown:
            raise ValueError(f"unknown matrix families {sorted(unknown)}; choose from {sorted(_OKLS_FAMILIES)}")

        def mask_fn(param, path):
            group = _base_group(param, path)
            if group == "muonh":
                path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
                if self.muon_truncate_family is not None and _TRUNCATE_FAMILIES[self.muon_truncate_family].search(
                    path_lower
                ):
                    return "muonh_trunc"
                if any(_TRUNCATE_FAMILIES[f].search(path_lower) for f in self.eig_families):
                    return "eig"
                if any(_OKLS_FAMILIES[f].search(path_lower) for f in self.okls_targets):
                    return "okls"
                if any(_OKLS_FAMILIES[f].search(path_lower) for f in self.muon_free_families):
                    return "muon_free"
                if self.upper_qk_lr_mult != 1.0 and _SOFTMAX_QK.search(path_lower):
                    return "upper_qk"
                qk_own_group = self.muonh_qk_lr_mult != 1.0 or self.muonh_qk_momentum is not None
                if qk_own_group and _QK_PROJECTIONS.search(path_lower):
                    return "muonh_qk"
                if self.muonh_attn_lr_mult != 1.0 and _OKLS_FAMILIES["attn"].search(path_lower):
                    return "muonh_attn"
                routed_own_group = (
                    self.muonh_routed_lr_mult != 1.0
                    or self.muonh_routed_momentum is not None
                    or self.muonh_routed_consistency_beta2 is not None
                    or self.muonh_routed_cautious
                    or self.muonh_routed_slow_rate is not None
                    or self.muonh_routed_slow_weight is not None
                    or self.muonh_routed_bimaxwell_start_step is not None
                )
                if routed_own_group and _OKLS_FAMILIES["routed"].search(path_lower):
                    return "muonh_routed"
            return group

        def _base_group(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _FROZEN_LEAVES.search(path_lower):
                return "frozen"
            if self.latent_proj_update != "muonh" and _LATENT_PROJ.search(path_lower):
                return "frozen" if self.latent_proj_update == "frozen" else "stiefel"
            kda_leaf = _kda_leaf(path_lower)
            if kda_leaf == _KDA_BETA_LEAF:
                return "kda_beta"
            if kda_leaf in _KDA_BETA_MLP_LEAVES:
                return self.kda_beta_mlp_group
            if kda_leaf in _KDA_ADAM_LEAVES:
                return "kda_decay"
            # AttnRes pseudo-queries are per-layer vectors (2D once stacked, which would route to MuonH).
            if "attn_res_query" in path_lower:
                return "attn_res_query"
            # Inkling rel-pos weights (r_proj and the shared bias bank); value embeddings and their mixing weights.
            if path_lower.endswith(".value_embed") and self.value_embed_lr_mult != 1.0:
                return "value_embed"
            if ".rel_pos." in path_lower or re.search(
                r"(?:^|\.)(value_embed|ve_lambda|ve_gate|xsa_scale|xsa_gate|head_mix|ssmax_scale|shared_gate|laurel_[ab]_\w+|ple_up|moe_out_gate_[wb]|bigram_gate_[wb]|bigram_gate_[ab]_lr|trigram_gate_[wb]|trigram_gate_[ab]_lr|bank_scale|router_logit_scale|expert_output_gain|bias_\w+|dyt_alpha|dyt_beta|qk_mult|diff_lambda|diff_lambda_init|vres_lambda|rot_scale|null_const_[vw]|comba_d|v_filter_[wb]|gamma|ngram_stat_gate_[wb]|ngram_stat_up|forget_gate_[wb]|lm_head_bias|router_tok_[ab]|router_tie_alpha|router_hist_w|router_mlp_[ab]|expert_router_alpha)$",
                path_lower,
            ):
                return "adam"
            # Product-key memory: sparse value tables at their own LR; codebooks and the zero-init output on Adam
            # (MuonH cannot move a zero matrix); the query and gate projections fall through to MuonH.
            if _MEMORY_VALUES.fullmatch(path_lower):
                return "memory"
            if _MEMORY_ADAM.fullmatch(path_lower):
                return "adam"
            # Output bigram prior: a sparse token table and a zero-init read-out (MuonH cannot move a zero matrix).
            if _OUTPUT_BIGRAM.search(path_lower):
                return "output_bigram"
            if "token_embed_ple" in path_lower:
                return "ple"
            if re.search(r"token_embed(2|3)", path_lower) and (
                self.embed2_lr_mult != 1.0 or self.embed2_row_sparse_adam or self.embed2_update != "adam"
            ):
                return "embed2"
            if "token_embed" in path_lower:
                return self.embed_group
            if _is_router_weight(path_lower):
                if self.router_group == "muonh":
                    return "muonh_router" if self.router_lr_mult != 1.0 else "muonh"
                if self.router_lr_mult != 1.0 or self.router_weight_decay is not None:
                    return "router"
                return "adam"
            if "router_bias" in path_lower or _is_gate_or_router_weight(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return self.lm_head_group
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            # Scanning prepends a layer axis, so norm gains / SConv kernels stay named ``.weight``
            # (route to Adam) while expert matrices become 4D and other matmuls 3D (route to MuonH).
            if path_lower.endswith(".weight"):
                return "adam"
            if hasattr(param, "ndim") and param.ndim in (2, 3, 4):
                if self.routed_expert_optimizer == "adam" and _OKLS_FAMILIES["routed"].search(path_lower):
                    return "adam"
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeMuonHConfig",
    "magma_metrics",
    "optimizer_diagnostics",
    "scale_with_grug_muonh",
]
