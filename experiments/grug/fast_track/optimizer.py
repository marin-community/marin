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


def _named_scope_transform(name: str, tx: optax.GradientTransformation) -> optax.GradientTransformation:
    """``tx`` with its update traced under ``jax.named_scope(name)`` (attributes the group's kernels in profiles)."""

    def update_fn(updates, state, params=None):
        with jax.named_scope(name):
            return tx.update(updates, state, params)

    return optax.GradientTransformation(tx.init, update_fn)


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


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate):
    """MuonH hyperball step: move along the orthogonalized direction, then project back to the
    parameter's Frobenius sphere (scale-invariant update). Stacked leaves take one sphere per layer."""
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

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param = _pin_sharding(new_param, param)  # correct the sharded norm reduction (issue #8073)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(scale_invariant_update, params, direction_updates, is_leaf=lambda x: x is None)


# Power iterations per step for sigma_1 (warm-started from the previous step's vector), and at init (cold).


# KDA-layer leaves (``kda_blocks.stacked.attn.<leaf>``) and their update rules. The q/k/v/o and
# output-gate matrices and the random-init ``kda_dd_rope`` angle projections (w_rot_down/w_rot_up) take the
# MuonH catch-all, the ShortConv kernels and output-norm scale are ``.weight`` leaves (Adam), and the zero-init
# angle amplitude ``rot_scale`` is on the generic Adam list.
_KDA_ATTN_LEAF = re.compile(r"kda_blocks(?:_tail|_rest)?\.stacked\.attn\.(\w+)")
# Low-rank forget gate, per-head A_log, per-channel dt_bias and the zero-init push / erase-gate
# projections (MuonH cannot move a zero matrix): Adam (no weight decay).
_KDA_ADAM_LEAVES = frozenset({"w_a_down", "w_a_up", "a_log", "dt_bias", "push_decay", "w_push", "w_erase", "beta_scale"})
# Write-strength projection: MuonH at ``kda_beta_lr_mult`` x the MuonH LR.
_KDA_BETA_LEAF = "w_beta"
# Low-rank write-strength MLP (``kda_beta_rank``): LR group chosen by ``kda_beta_mlp_group``.
_KDA_BETA_MLP_LEAVES = frozenset({"w_beta_down", "w_beta_up"})


def _kda_leaf(path_lower: str) -> str | None:
    match = _KDA_ATTN_LEAF.fullmatch(path_lower)
    return None if match is None else match.group(1)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for exactly the ``attn_gate``, MoE ``router`` and SwitchHead gate weight leaves.

    Matches the leaf attribute name at the end of the path, so it selects ``...attn.attn_gate`` and
    ``...mlp.router`` but not the separate ``...mlp.router_bias`` leaf.
    """
    return path_lower.endswith(
        (
            ".attn_gate",
            ".attn_gate_up",
            ".router",
            ".router_down",
            ".router_up",
            ".switch_v_gate",
            ".switch_o_gate",
            ".mix_gate",
            ".kv_mix_gate",
            ".latent_up_gate",
            ".latent_width_gate",
            ".latent_mix_in_gate",
            ".latent_mix_out_gate",
        )
    )


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


def _scale_by_adam_decoupled_decay(
    adam: optax.GradientTransformation, weight_decay: float, gain_weight_decay: float, total_steps: int
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
            return u + anneal * c * p

        updates = jax.tree.map(decay, updates, params, coefficients)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def _power_decay_schedule(peak: float, floor: float, warmup_steps: int, total_steps: int, power: float):
    """Linear warmup to ``peak``, then ``floor + (peak - floor) * (1 - p**power)``, ``p`` = post-warmup progress."""

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * step / max(warmup_steps, 1)
        progress = jnp.clip((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0)
        decayed = floor + (peak - floor) * (1.0 - progress**power)
        return jnp.where(step < warmup_steps, warm, decayed)

    return schedule


BIMAXWELL_FAST_RATE = 0.15
BIMAXWELL_SLOW_RATE = 0.02
BIMAXWELL_SLOW_WEIGHT = 0.5615
"""Bi-Maxwell's twin rails (ANVIL II): EMA rates of the fast and slow rail, and the slow rail's blend weight."""


class MuonMomentumState(NamedTuple):
    count: jax.Array
    buf: optax.Updates
    fast: optax.Updates | None
    slow: optax.Updates | None


def scale_by_muon_momentum(
    momentum: float,
    nesterov: bool,
    switch_step: int | None,
    warmup_start: float | None,
    warmup_steps: int,
) -> optax.GradientTransformation:
    """MuonH momentum outside Newton-Schulz, with optional Bi-Maxwell (modded-nanogpt #339) from ``switch_step`` on.

    Before ``switch_step`` (always, when it is None) it is (Nesterov) momentum: ``buf = m buf + g``, out
    ``m buf + g`` (``buf`` without ``nesterov``). With ``warmup_start``, ``m`` rises linearly from ``warmup_start``
    to ``momentum`` over the first ``warmup_steps`` steps (modded-nanogpt's momentum warmup), then stays at
    ``momentum``. From ``switch_step`` on, a fast and a slow EMA of the gradient (rates ``BIMAXWELL_FAST_RATE`` and
    ``BIMAXWELL_SLOW_RATE``), both started from ``(1 - m) buf``, mix as ``M = (1 - w) fast + w slow``
    (``w = BIMAXWELL_SLOW_WEIGHT``, mean age about 30 steps) and the output is ``g + m (M - g)``. Newton-Schulz is
    scale-invariant per matrix, so the regimes' different scales don't matter."""

    def momentum_at(count):
        if warmup_start is None:
            return momentum
        frac = jnp.minimum((count - 1).astype(jnp.float32) / max(1, warmup_steps), 1.0)
        return warmup_start + (momentum - warmup_start) * frac

    def init(params):
        zeros = lambda: jax.tree.map(jnp.zeros_like, params)  # noqa: E731
        if switch_step is None:
            return MuonMomentumState(jnp.zeros([], jnp.int32), zeros(), None, None)
        return MuonMomentumState(jnp.zeros([], jnp.int32), zeros(), zeros(), zeros())

    def update(updates, state, params=None):
        count = state.count + 1
        m = momentum_at(count)

        def early(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            new_buf = jax.tree.map(lambda g, b: (m * b + g).astype(b.dtype), g_tree, buf_tree)
            if not nesterov:
                return new_buf, new_buf, fast_tree, slow_tree
            out = jax.tree.map(lambda g, b: m * b + g, g_tree, new_buf)
            return out, new_buf, fast_tree, slow_tree

        if switch_step is None:
            out, buf, _, _ = early((updates, state.buf, None, None))
            return out, MuonMomentumState(count, buf, None, None)

        def late(args):
            g_tree, buf_tree, fast_tree, slow_tree = args
            # At the switch step, start both EMAs from the Nesterov buffer's EMA-scale value.
            first = count == switch_step + 1

            def leaf(g, buf, fast, slow):
                start = (1.0 - m) * buf
                fast = jnp.where(first, start, fast)
                slow = jnp.where(first, start, slow)
                fast = fast + BIMAXWELL_FAST_RATE * (g - fast)
                slow = slow + BIMAXWELL_SLOW_RATE * (g - slow)
                mix = (1.0 - BIMAXWELL_SLOW_WEIGHT) * fast + BIMAXWELL_SLOW_WEIGHT * slow
                return (g + m * (mix - g)).astype(g.dtype), fast.astype(buf.dtype), slow.astype(buf.dtype)

            res = jax.tree.map(leaf, g_tree, buf_tree, fast_tree, slow_tree)

            def pick(i):
                return jax.tree.map(lambda _, r: r[i], g_tree, res)

            return pick(0), buf_tree, pick(1), pick(2)

        out, buf, fast, slow = jax.lax.cond(
            count > switch_step, late, early, (updates, state.buf, state.fast, state.slow)
        )
        return out, MuonMomentumState(count, buf, fast, slow)

    return optax.GradientTransformation(init, update)


class MuonHState(NamedTuple):
    """MuonH state when the momentum runs outside Newton-Schulz (Bi-Maxwell)."""

    momentum: MuonMomentumState
    core: optax.OptState


def scale_with_grug_muonh(
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    learning_rate,
    coefficient_type: CoefficientType,
    pre_norm: str,
    bimaxwell_switch_step: int | None,
    momentum_warmup_start: float | None,
    momentum_warmup_steps: int,
) -> optax.GradientTransformation:
    """MuonH transform for the stacked model: Newton-Schulz direction + Frobenius hyperball step.

    ``bimaxwell_switch_step`` (Bi-Maxwell from that step) or ``momentum_warmup_start`` (momentum warmup) moves the
    momentum to ``scale_by_muon_momentum`` ahead of Newton-Schulz (Nesterov momentum there equals the in-Muon one).
    """
    external_momentum = bimaxwell_switch_step is not None or momentum_warmup_start is not None
    muon_transform = _grug_scale_with_muon(
        momentum=0.0 if external_momentum else momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        head_dim=None,
        pre_norm=pre_norm,
        top_shrink=0.0,
        precond_beta2=None,
        truncate_frac=0.0,
        post_blend=0.0,
    )
    momentum_stage = (
        scale_by_muon_momentum(momentum, nesterov, bimaxwell_switch_step, momentum_warmup_start, momentum_warmup_steps)
        if external_momentum
        else None
    )

    def init_fn(params):
        if momentum_stage is None:
            return muon_transform.init(params)
        return MuonHState(momentum_stage.init(params), muon_transform.init(params))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")
        if momentum_stage is None:
            directions, core_state = muon_transform.update(updates, state, params)
            return _scale_invariant_hyperball_updates(params, directions, learning_rate), core_state
        mixed, momentum_state = momentum_stage.update(updates, state.momentum, params)
        directions, core_state = muon_transform.update(mixed, state.core, params)
        return _scale_invariant_hyperball_updates(params, directions, learning_rate), MuonHState(
            momentum_state, core_state
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _is_matrix_stack(x) -> bool:
    return hasattr(x, "ndim") and x.ndim >= 2


BIMAXWELL_START_FRAC = 1 / 3
"""Fraction of training after which Bi-Maxwell's two-timescale momentum replaces Nesterov momentum."""
ATTN_RES_QUERY_LR_SCALE = 0.1
"""Adam LR of the AttnRes pseudo-queries as a multiple of ``adam_lr``."""
KDA_BETA_LR_MULT = 2.0
"""MuonH LR multiple of the ``kda_beta`` group (the KDA write-strength projection)."""
# MSAR (``attn_res_streams``) conv taps (one leaf per stream) and stream mixers ride along: the ``[m * D, m]`` mixer
# and the ``[N, K, D]`` taps must not fall into the MuonH matrix catch-all.
_ADAM_LEAVES = re.compile(
    r"(?:^|\.)(value_embed|ve_lambda|ve_gate|bigram_gate_[wb]|bigram_gate_[ab]_lr|qk_mult|attn_res_partial_scale"
    r"|attn_res_stream_mix_(?:w|alpha|b)|attn_res_stream_conv(?:_scalar)?\.\d+|attn_res_embed_scale)$"
)


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
    kda_beta_group: str = "kda_beta"
    embed2_lr_mult: float = 1.0
    """Adam LR multiplier of the second (e.g. bigram) embedding table alone."""
    embed2_beta2: float = 0.95
    """Second-moment decay of ``embed2_row_sparse_adam``."""
    muon_pre_norm: str = "none"
    """MuonH momentum normalization before Newton-Schulz: ``none``, ``out`` or ``in`` (MuonEq)."""
    muonh_decay_power: float | None = None
    """Decay shape of the MuonH LR after warmup: ``floor + (peak - floor) * (1 - p**power)`` over training
    progress ``p`` (modded-nanogpt MuonH records #345/#351). None keeps the shared schedule (linear = 1)."""
    muon_bimaxwell: bool = False
    momentum_warmup_start: float | None = None
    """MuonH Nesterov momentum at step 0, rising linearly to ``momentum`` over ``momentum_warmup`` (a fraction of
    training, or steps); None keeps ``momentum`` constant."""
    momentum_warmup: float = 0.1
    adam_ademamix_alpha: float = 0.0
    """AdEMAMix (arXiv 2409.03152) on the plain-Adam groups: weight of a slow gradient EMA added to Adam's
    first moment, warmed up linearly over the run (0: off; the paper uses 5-8)."""
    adam_ademamix_beta3: float = 0.999
    """Decay of the AdEMAMix slow EMA, warmed up over ``adam_ademamix_warmup`` as in the paper."""
    router_lr_mult: float = 1.0
    """LR multiplier of the router weights alone (of ``adam_lr`` or, with ``router_group=muonh``, the MuonH LR)."""

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
        bimaxwell_switch = int(BIMAXWELL_START_FRAC * num_train_steps) if self.muon_bimaxwell else None

        def clipped(*components):
            if self.max_grad_norm:
                components = (optax.clip_by_global_norm(self.max_grad_norm), *components)
            return optax.chain(*components)

        def muonh_at(lr):
            return clipped(
                scale_with_grug_muonh(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=lr,
                    coefficient_type=self.coefficient_type,
                    pre_norm=self.muon_pre_norm,
                    bimaxwell_switch_step=bimaxwell_switch,
                    momentum_warmup_start=self.momentum_warmup_start,
                    momentum_warmup_steps=_convert_frac_or_steps(self.momentum_warmup, num_train_steps),
                ),
                _match_named_update_sharding(),
            )

        def adam_moments(beta1, beta2):
            if not self.adam_ademamix_alpha:
                return optax.scale_by_adam(beta1, beta2, self.epsilon)
            # AdEMAMix: alpha and beta3 warm up linearly over the whole run, as in the paper.
            warmup = max(1, num_train_steps)
            alpha = optax.join_schedules(
                [
                    optax.linear_schedule(0.0, self.adam_ademamix_alpha, warmup),
                    optax.constant_schedule(self.adam_ademamix_alpha),
                    optax.linear_schedule(self.adam_ademamix_alpha, 0.0, 0),
                ],
                [warmup, max(warmup, num_train_steps)],
            )
            return optax.contrib.scale_by_ademamix(
                beta1, beta2, _ademamix_beta3_schedule(beta1, self.adam_ademamix_beta3, warmup), alpha, self.epsilon
            )

        def adam_at(lr, *, decay: bool = False):
            """Adam at ``lr``; ``decay`` adds the annealed decoupled ``gate_router_weight_decay``."""
            adam = adam_moments(self.beta1, self.beta2)
            if decay and self.gate_router_weight_decay > 0.0:
                adam = _scale_by_adam_decoupled_decay(adam, self.gate_router_weight_decay, 0.0, num_train_steps)
            return clipped(adam, optax.scale(-lr))

        def optimizer(learning_rate, adam_lr):
            transforms = {
                "muonh": muonh_at(learning_rate),
                "adamh": clipped(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)),
                "adam": adam_at(adam_lr, decay=True),
                "attn_res_query": adam_at(adam_lr * ATTN_RES_QUERY_LR_SCALE),
                "kda_beta": muonh_at(learning_rate * KDA_BETA_LR_MULT),
                "kda_beta_adam": adam_at(adam_lr, decay=True),
                "kda_decay": adam_at(adam_lr),
                # Learned output gains (moe_null_learned_gain, extra_mlp_gain) and the entropy head: no decay pull.
                "null_gain": adam_at(adam_lr),
                "embed2": adam_at(adam_lr * self.embed2_lr_mult),
                "router": adam_at(adam_lr * self.router_lr_mult, decay=True),
            }
            return optax.multi_transform(
                {name: _named_scope_transform(f"opt_{name}", tx) for name, tx in transforms.items()}, self.create_mask
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def __post_init__(self):
        if self.kda_beta_group not in ("kda_beta", "adam"):
            raise ValueError(f"kda_beta_group must be kda_beta or adam, got {self.kda_beta_group!r}")

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
            kda_leaf = _kda_leaf(path_lower)
            if kda_leaf == _KDA_BETA_LEAF:
                return "kda_beta_adam" if self.kda_beta_group == "adam" else self.kda_beta_group
            if kda_leaf in _KDA_BETA_MLP_LEAVES:
                return "kda_beta"
            if kda_leaf in _KDA_ADAM_LEAVES:
                return "kda_decay"
            if re.search(
                r"null_gain|extra_mlp_gain|entropy_head|mid_conf|final_conf|skip_probe|skip_router", path_lower
            ):
                return "null_gain"
            # AttnRes pseudo-queries are per-layer vectors (2D once stacked, which would route to MuonH).
            if "attn_res_query" in path_lower:
                return "attn_res_query"
            # Inkling rel-pos weights; value embeddings and their mixing weights; small per-layer gates.
            if ".rel_pos." in path_lower or _ADAM_LEAVES.search(path_lower):
                return "adam"
            if re.search(r"token_embed(2|3)", path_lower) and self.embed2_lr_mult != 1.0:
                return "embed2"
            if "token_embed" in path_lower:
                return "adam"
            if _is_router_weight(path_lower):
                return "router" if self.router_lr_mult != 1.0 else "adam"
            if "router_bias" in path_lower or _is_gate_or_router_weight(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
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
