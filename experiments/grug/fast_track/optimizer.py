# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon, _target_named_sharding
from experiments.grug.fast_track.okls import OKLS_MATMUL_DTYPES, scale_with_grug_okls


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


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float, per_expert: bool = False):
    """MuonH hyperball step: move along the orthogonalized direction, then project back to the
    parameter's Frobenius sphere (scale-invariant update). Stacked leaves take one sphere per layer, and
    with ``per_expert`` the 4-D expert stacks ``[L, E, in, out]`` take one sphere per (layer, expert)."""
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            # jnp.linalg.norm over a sharded matrix mis-lowers under SPMD and over-counts (issue #8073);
            # sum-of-squares in float32 plus a same-layout reshard of the intermediate reduces correctly.
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param.astype(jnp.float32))))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update.astype(jnp.float32))))
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param = _pin_sharding(new_param, param)
            new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param.astype(jnp.float32))))
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = (2, 3) if per_expert and param.ndim == 4 else tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param = _pin_sharding(new_param, param)  # correct the sharded norm reduction (issue #8073)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(scale_invariant_update, params, direction_updates, is_leaf=lambda x: x is None)


# KDA-layer leaves (``kda_blocks.stacked.attn.<leaf>``) and their update rules. The q/k/v/o and
# output-gate matrices and the random-init ``kda_dd_rope`` angle projections (w_rot_down/w_rot_up) take the
# MuonH catch-all, the ShortConv kernels and output-norm scale are ``.weight`` leaves (Adam), and the zero-init
# angle amplitude ``rot_scale`` is on the generic Adam list.
_KDA_ATTN_LEAF = re.compile(r"kda_blocks\.stacked\.attn\.(\w+)")
# Low-rank forget gate, per-head A_log, per-channel dt_bias and the zero-init push / erase-gate
# projections (MuonH cannot move a zero matrix): Adam (no weight decay).
_KDA_ADAM_LEAVES = frozenset({"w_a_down", "w_a_up", "a_log", "dt_bias", "push_decay", "w_push", "w_erase"})
# Write-strength projection: MuonH at ``kda_beta_lr_mult`` x the MuonH LR.
_KDA_BETA_LEAF = "w_beta"
# Low-rank write-strength MLP (``kda_beta_rank``): LR group chosen by ``kda_beta_mlp_group``.
_KDA_BETA_MLP_LEAVES = frozenset({"w_beta_down", "w_beta_up"})


# Matrix families that ``okls_targets`` can move from MuonH to the OKLS direction.
_OKLS_FAMILIES: dict[str, re.Pattern] = {
    "attn": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(q|k|v|o|g|dkv|uk|uv|q2|uk2)$"),
    "routed": re.compile(r"\.mlp\.expert_mlp\.w_(gate|up|down)$"),
    "shared": re.compile(r"\.shared\.\d+\.w_(gate|up|down)$"),
    "latent": re.compile(r"\.mlp\.w_latent_(down|up)$"),
    "gated_norm": re.compile(r"gated_norm\.w_(down|up)$"),
    # Per-projection subsets of "attn" (KDA and MLA together).
    "attn_q": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_q2?$"),
    "attn_k": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(k|uk|uk2)$"),
    "attn_v": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(v|uv)$"),
    "attn_o": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_o$"),
    "attn_other": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(g|dkv)$"),
    # The softmax (MLA) layers' q and k: KDA L2-normalizes q and k, so their scale is inert there.
    "mla_qk": re.compile(r"stacked_blocks\.stacked\.attn\.w_(q|uk|q2|uk2)$"),
}


_MEMORY_VALUES = re.compile(r"memory\.\d+\.values")
_MEMORY_ADAM = re.compile(r"memory\.\d+\.(keys|w_out)")


def _kda_leaf(path_lower: str) -> str | None:
    match = _KDA_ATTN_LEAF.fullmatch(path_lower)
    return None if match is None else match.group(1)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for exactly the ``attn_gate`` and MoE ``router`` weight leaves.

    Matches the leaf attribute name at the end of the path, so it selects ``...attn.attn_gate`` and
    ``...mlp.router`` but not the separate ``...mlp.router_bias`` leaf.
    """
    return path_lower.endswith((".attn_gate", ".router", ".router_down", ".router_up"))


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
    adam: optax.GradientTransformation, weight_decay: float, gain_weight_decay: float, total_steps: int
) -> optax.GradientTransformation:
    """``scale_by_adam`` plus decoupled weight decay on ``attn_gate`` and the ``router`` weight (and
    ``gain_weight_decay`` on the zero-centered norm gains), annealed linearly to 0 over ``total_steps``.
    The coefficient reads the Adam ``count`` and the state stays ``ScaleByAdamState``, so a checkpoint
    written without decay resumes at the right step with its moments intact."""

    def init_fn(params):
        return adam.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_scale_by_adam_decoupled_decay requires params for decoupled decay")
        step = optax.tree_utils.tree_get(state, "count")
        updates, next_state = adam.update(updates, state, params)
        anneal = jnp.clip(1.0 - step / total_steps, 0.0, None)
        coefficients = _adam_decay_coefficients(params, weight_decay, gain_weight_decay)
        updates = jax.tree.map(lambda u, p, c: u + anneal * c * p if c else u, updates, params, coefficients)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    head_dim: int | None = None,
    neuron_norm_beta2: float | None = None,
    hyperball_per_expert: bool = False,
    pre_norm: str = "none",
    top_shrink: float = 0.0,
    precond_beta2: float | None = None,
) -> optax.GradientTransformation:
    """MuonH transform for the stacked model: Newton-Schulz direction + Frobenius hyperball step.

    ``neuron_norm_beta2`` adds NorMuon's (arXiv 2510.05491) neuron-wise normalization between the two:
    each output column of the orthogonalized direction is divided by the root of an EMA of its mean
    square (over the input axis); the hyperball step then sets the overall magnitude.
    """
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        head_dim=head_dim,
        pre_norm=pre_norm,
        top_shrink=top_shrink,
        precond_beta2=precond_beta2,
    )

    def _neuron_second_moment(x):
        if x is None or not hasattr(x, "ndim") or x.ndim < 2:
            return None
        return jnp.zeros(x.shape[:-2] + x.shape[-1:], jnp.float32)

    def init_fn(params):
        muon_state = muon_transform.init(params)
        if neuron_norm_beta2 is None:
            return muon_state
        return muon_state, jax.tree.map(_neuron_second_moment, params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")
        if neuron_norm_beta2 is None:
            muon_updates, next_state = muon_transform.update(updates, state, params)
        else:
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
            next_state = (muon_state, second_moment)
        muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate, hyperball_per_expert)
        return muonh_updates, next_state

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
_FROZEN_LEAVES = re.compile(r"(?:^|\.)(ngram_stat_(table|code)|latent_select_idx)$")
_HYPERBALL_GROUPS = frozenset({"muonh", "adamh", "kda_beta", "muonh_attn", "muonh_routed", "sinkhornh"})


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


class BiMaxwellState(NamedTuple):
    count: jax.Array
    buf: optax.Updates
    fast: optax.Updates
    slow: optax.Updates


def scale_by_bimaxwell_momentum(momentum: float, switch_step: int) -> optax.GradientTransformation:
    """Bi-Maxwell momentum (modded-nanogpt #339) for Muon-family groups whose own momentum is 0.

    Before ``switch_step`` it is Nesterov momentum (``buf = m buf + g``, out ``m buf + g``). From
    ``switch_step`` on, a fast (0.15) and a slow (0.02) EMA of the gradient, both started from
    ``(1 - m) buf``, mix as ``M = 0.4385 fast + 0.5615 slow`` (mean age about 30 steps) and the output
    is ``g + m (M - g)``. Newton-Schulz is scale-invariant per matrix, so the two regimes' different
    scales don't matter."""

    def init(params):
        zeros = lambda: jax.tree.map(jnp.zeros_like, params)  # noqa: E731
        return BiMaxwellState(jnp.zeros([], jnp.int32), zeros(), zeros(), zeros())

    def update(updates, state, params=None):
        count = state.count + 1

        def early(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            new_buf = jax.tree.map(lambda g, b: momentum * b + g, g_tree, buf_tree)
            out = jax.tree.map(lambda g, b: momentum * b + g, g_tree, new_buf)
            return out, new_buf, fast_tree, slow_tree

        def late(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            # At the switch step, start both EMAs from the Nesterov buffer's EMA-scale value.
            first = count == switch_step + 1

            def leaf(g, buf, fast, slow):
                start = (1.0 - momentum) * buf
                fast = jnp.where(first, start, fast)
                slow = jnp.where(first, start, slow)
                fast = fast + 0.15 * (g - fast)
                slow = slow + 0.02 * (g - slow)
                return g + momentum * (0.4385 * fast + 0.5615 * slow - g), fast, slow

            res = jax.tree.map(leaf, g_tree, buf_tree, fast_tree, slow_tree)

            def pick(i):
                return jax.tree.map(lambda _, r: r[i], g_tree, res)

            return pick(0), buf_tree, pick(1), pick(2)

        out, buf, fast, slow = jax.lax.cond(
            count > switch_step, late, early, (updates, state.buf, state.fast, state.slow)
        )
        return out, BiMaxwellState(count, buf, fast, slow)

    return optax.GradientTransformation(init, update)


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
    """Recompute the OKLS inverse roots every this many steps (stored in between)."""
    lm_head_group: str = "adamh"
    """LR group of ``output_proj``: ``adamh``, ``muonh`` or ``sinkhornh`` (Sinkhorn-balanced momentum + hyperball)."""
    embed_group: str = "adam"
    embed2_lr_mult: float = 1.0
    """Adam LR multiplier of the second (e.g. bigram) embedding table alone."""
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
    grokfast_adam_only: bool = False
    """Apply Grokfast to the plain-Adam groups only (on MuonH it acts as extra momentum)."""
    snoo_period: int = 0
    """SNOO outer step every this many inner steps (0: off)."""
    snoo_lr: float = 0.5
    snoo_momentum: float = 0.5

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

        def optimizer(learning_rate, adam_lr):
            def muonh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                if self.muon_grad_power != 1.0:
                    components.append(scale_by_grad_power(self.muon_grad_power))
                if self.muon_mars_gamma:
                    components.append(scale_by_mars_correction(self.muon_mars_gamma, self.momentum))
                if self.muon_bimaxwell:
                    components.append(
                        scale_by_bimaxwell_momentum(self.momentum, int(self.bimaxwell_start_frac * num_train_steps))
                    )
                components.append(
                    scale_with_grug_muonh(
                        momentum=0.0 if self.muon_bimaxwell else self.momentum,
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
                        adam, self.gate_router_weight_decay, self.gain_weight_decay, num_train_steps
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

            transforms = {
                "muonh": muonh_transform_at(learning_rate),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
                "attn_res_query": plain_adam_at(adam_lr * self.attn_res_query_lr_scale),
                "kda_beta": muonh_transform_at(learning_rate * self.kda_beta_lr_mult),
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
                    ),
                    _match_named_update_sharding(),
                ),
                "muonh_attn": muonh_transform_at(learning_rate * self.muonh_attn_lr_mult),
                "muonh_routed": muonh_transform_at(learning_rate * self.muonh_routed_lr_mult),
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
                "embed2": plain_adam_at(adam_lr * self.embed2_lr_mult),
                # The n-gram statistic table and its code are data statistics written by the trainer, not trained.
                "frozen": optax.set_to_zero(),
                "ple": plain_adam_at(adam_lr * self.ple_lr_mult),
                "value_embed": plain_adam_at(adam_lr * self.value_embed_lr_mult),
                "memory": plain_adam_at(adam_lr * self.memory_lr_mult),
                "kda_decay": plain_adam_at(adam_lr * self.kda_decay_lr_mult, self.kda_decay_beta1, self.kda_decay_beta2),
            }
            inner = optax.multi_transform(transforms, self.create_mask)
            if self.grokfast_lambda and not self.grokfast_adam_only:
                inner = optax.chain(scale_by_grokfast_ema(self.grokfast_alpha, self.grokfast_lambda), inner)
            if self.snoo_period <= 0:
                return inner
            return snoo(
                inner,
                self.create_mask,
                period=self.snoo_period,
                outer_lr=self.snoo_lr,
                outer_momentum=self.snoo_momentum,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def __post_init__(self):
        if self.lm_head_group not in ("adamh", "muonh", "sinkhornh"):
            raise ValueError(f"lm_head_group must be adamh, muonh or sinkhornh, got {self.lm_head_group!r}")
        if self.kda_beta_mlp_group not in ("kda_beta", "adam"):
            raise ValueError(f"kda_beta_mlp_group must be kda_beta or adam, got {self.kda_beta_mlp_group!r}")
        if self.embed_group not in ("adam", "adamh", "sinkhorn"):
            raise ValueError(f"embed_group must be adam, adamh or sinkhorn, got {self.embed_group!r}")

    def create_mask(self, params):
        paths = leaf_key_paths(params)
        unknown = (set(self.okls_targets) | set(self.muon_free_families)) - set(_OKLS_FAMILIES)
        if unknown:
            raise ValueError(f"unknown matrix families {sorted(unknown)}; choose from {sorted(_OKLS_FAMILIES)}")

        def mask_fn(param, path):
            group = _base_group(param, path)
            if group == "muonh":
                path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
                if any(_OKLS_FAMILIES[f].search(path_lower) for f in self.okls_targets):
                    return "okls"
                if any(_OKLS_FAMILIES[f].search(path_lower) for f in self.muon_free_families):
                    return "muon_free"
                if self.muonh_attn_lr_mult != 1.0 and _OKLS_FAMILIES["attn"].search(path_lower):
                    return "muonh_attn"
                if self.muonh_routed_lr_mult != 1.0 and _OKLS_FAMILIES["routed"].search(path_lower):
                    return "muonh_routed"
            return group

        def _base_group(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _FROZEN_LEAVES.search(path_lower):
                return "frozen"
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
                r"(?:^|\.)(value_embed|ve_lambda|ve_gate|xsa_scale|xsa_gate|head_mix|ssmax_scale|shared_gate|laurel_[ab]_\w+|ple_up|moe_out_gate_[wb]|bigram_gate_[wb]|bigram_gate_[ab]_lr|trigram_gate_[wb]|trigram_gate_[ab]_lr|bank_scale|bias_\w+|dyt_alpha|dyt_beta|qk_mult|diff_lambda|diff_lambda_init|vres_lambda|rot_scale|null_const_[vw]|comba_d|v_filter_[wb]|gamma|ngram_stat_gate_[wb]|ngram_stat_up)$",
                path_lower,
            ):
                return "adam"
            # Product-key memory: sparse value tables at their own LR; codebooks and the zero-init output on Adam
            # (MuonH cannot move a zero matrix); the query and gate projections fall through to MuonH.
            if _MEMORY_VALUES.fullmatch(path_lower):
                return "memory"
            if _MEMORY_ADAM.fullmatch(path_lower):
                return "adam"
            if "token_embed_ple" in path_lower:
                return "ple"
            if re.search(r"token_embed(2|3)", path_lower) and self.embed2_lr_mult != 1.0:
                return "embed2"
            if "token_embed" in path_lower:
                return self.embed_group
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
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeMuonHConfig",
    "scale_with_grug_muonh",
]
