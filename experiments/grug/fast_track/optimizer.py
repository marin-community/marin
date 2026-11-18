# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig, _convert_frac_or_steps
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths
from optax import tree_utils as otu

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
# output-gate matrices take the MuonH catch-all, and the ShortConv kernels and output-norm scale are ``.weight``
# leaves (Adam).
_KDA_ATTN_LEAF = re.compile(r"kda_blocks\.stacked\.attn\.(\w+)")
# Low-rank forget gate, per-head A_log and per-channel dt_bias: Adam (no weight decay).
_KDA_ADAM_LEAVES = frozenset({"w_a_down", "w_a_up", "a_log", "dt_bias"})
# Write-strength projection: MuonH at ``KDA_BETA_LR_MULT`` x the MuonH LR, or Adam (``kda_beta_group``).
_KDA_BETA_LEAF = "w_beta"


def _kda_leaf(path_lower: str) -> str | None:
    match = _KDA_ATTN_LEAF.fullmatch(path_lower)
    return None if match is None else match.group(1)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for exactly the ``attn_gate`` and MoE ``router`` weight leaves.

    Matches the leaf attribute name at the end of the path, so it selects ``...attn.attn_gate`` and
    ``...mlp.router`` but not the separate ``...mlp.router_bias`` leaf.
    """
    return path_lower.endswith((".attn_gate", ".router"))


def _is_router_weight(path_lower: str) -> bool:
    return path_lower.endswith(".mlp.router")


def _adam_decay_coefficients(params, gate_router_weight_decay: float):
    """Per-leaf decoupled weight decay of the ``adam`` group: ``gate_router_weight_decay`` on the
    ``attn_gate`` and ``router`` weight leaves, 0 everywhere else."""
    paths = leaf_key_paths(params)

    def coefficient(_, path):
        path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
        return gate_router_weight_decay if _is_gate_or_router_weight(path_lower) else 0.0

    return jax.tree.map(coefficient, params, paths)


def _ademamix_beta3_schedule(beta1: float, beta3: float, total_steps: int):
    """AdEMAMix's beta3 warmup: linear in half-life from ``beta1``'s to ``beta3``'s over ``total_steps``."""
    log_b1, log_b3 = jnp.log(beta1), jnp.log(beta3)

    def schedule(step):
        frac = jnp.clip(step / total_steps, 0.0, 1.0)
        return jnp.exp(log_b1 * log_b3 / ((1 - frac) * log_b3 + frac * log_b1))

    return schedule


def _scale_by_adam_decoupled_decay(
    adam: optax.GradientTransformation, weight_decay: float, total_steps: int
) -> optax.GradientTransformation:
    """``scale_by_adam`` plus decoupled weight decay on ``attn_gate`` and the ``router`` weight, annealed
    linearly to 0 over ``total_steps``. The coefficient reads the Adam ``count`` and the state stays
    ``ScaleByAdamState``, so a checkpoint written without decay resumes at the right step with its moments
    intact."""

    def init_fn(params):
        return adam.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_scale_by_adam_decoupled_decay requires params for decoupled decay")
        step = optax.tree_utils.tree_get(state, "count")
        updates, next_state = adam.update(updates, state, params)
        anneal = jnp.clip(1.0 - step / total_steps, 0.0, None)
        coefficients = _adam_decay_coefficients(params, weight_decay)

        def decay(u, p, c):
            if not c:
                return u
            return u + anneal * c * p

        updates = jax.tree.map(decay, updates, params, coefficients)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def _fixed_decoupled_decay(weight_decay: float) -> optax.GradientTransformation:
    """``u + weight_decay * p`` on every leaf, not annealed. Followed by ``scale(-lr)`` this is AdamW's decoupled
    decay, ``p <- p (1 - lr_t weight_decay) - lr_t u``."""

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_fixed_decoupled_decay requires params")
        return jax.tree.map(lambda u, p: u + weight_decay * p, updates, params), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update_fn)


def _row_norms(x: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True))


def scale_by_row_sphere_adam(b1: float, b2: float, eps: float, learning_rate: jax.Array) -> optax.GradientTransformation:
    """Row-wise AdamH for an embedding table ``[rows, D]``: the plain Adam step ``p - lr u``, then each row
    rescaled back to its norm before the step, so every row keeps its init norm. Rows whose gradient is zero this
    step (tokens absent from the batch) are left untouched, so no row drifts; their moments still decay. Returns
    the full parameter delta (no ``scale(-lr)`` after it)."""
    adam = optax.scale_by_adam(b1, b2, eps)

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("scale_by_row_sphere_adam requires params")
        updates, state = adam.update(grads, state, params)

        def leaf(g, u, p):
            if p.ndim != 2:
                raise ValueError(f"row-wise AdamH needs a [rows, D] table, got shape {p.shape}")
            active = jnp.any(g != 0, axis=-1, keepdims=True)
            p32 = p.astype(jnp.float32)
            stepped = p32 - learning_rate * u.astype(jnp.float32)
            renormed = stepped * (_row_norms(p32) / jnp.maximum(_row_norms(stepped), 1e-10))
            return jnp.where(active, renormed - p32, 0.0).astype(p.dtype)

        return jax.tree.map(leaf, grads, updates, params), state

    return optax.GradientTransformation(adam.init, update_fn)


class EmbedOptimizer(StrEnum):
    """How ``token_embed`` is optimized (``GrugMoeMuonHConfig.embed_optimizer``)."""

    ADAM_GROUP = "adam_group"
    """In the shared ``adam`` group (AdEMAMix when on; no decay on ``token_embed``)."""
    ADEMAMIX = "ademamix"
    """Own ``embed`` group: the ``adam`` group's moments (AdEMAMix when on) plus fixed ``embed_weight_decay``."""
    ADAM = "adam"
    """Own ``embed`` group: plain Adam moments (no AdEMAMix) plus fixed ``embed_weight_decay``."""
    ROW_ADAMH = "row_adamh"
    """Own ``embed`` group: ``scale_by_row_sphere_adam`` at the Adam LR (no AdEMAMix, no decay)."""


_TOKEN_EMBED = re.compile(r"(?:^|\.)token_embed$")


def _power_decay_schedule(peak: float, floor: float, warmup_steps: int, total_steps: int, power: float):
    """Linear warmup to ``peak``, then ``floor + (peak - floor) * (1 - p**power)``, ``p`` = post-warmup progress."""

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * step / max(warmup_steps, 1)
        progress = jnp.clip((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0)
        decayed = floor + (peak - floor) * (1.0 - progress**power)
        return jnp.where(step < warmup_steps, warm, decayed)

    return schedule


def scale_with_grug_muonh(
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    learning_rate,
    coefficient_type: CoefficientType,
    pre_norm: str,
) -> optax.GradientTransformation:
    """MuonH transform for the stacked model: Newton-Schulz direction + Frobenius hyperball step."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        pre_norm=pre_norm,
    )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")
        directions, core_state = muon_transform.update(updates, state, params)
        return _scale_invariant_hyperball_updates(params, directions, learning_rate), core_state

    return optax.GradientTransformation(muon_transform.init, update_fn)


class RadialMuonHState(NamedTuple):
    muon: optax.OptState
    count: jax.Array
    init_radius: optax.Params
    log_radius: optax.Params
    mu: optax.Params
    nu: optax.Params


def _slice_sum(x: jax.Array) -> jax.Array:
    """Float32 sum over each hyperball sphere: the whole 2-D matrix, or each slice along a stack's leading axis."""
    axes = tuple(range(x.ndim)) if x.ndim == 2 else tuple(range(1, x.ndim))
    return jnp.sum(x.astype(jnp.float32), axis=axes, keepdims=True)


def _slice_norm(x: jax.Array) -> jax.Array:
    return jnp.sqrt(_slice_sum(jnp.square(x.astype(jnp.float32))))


def scale_with_grug_muonh_radial(
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    learning_rate,
    coefficient_type: CoefficientType,
    pre_norm: str,
    softcap: float,
    b1: float,
    b2: float,
    eps: float,
) -> optax.GradientTransformation:
    """MuonH with a learned, bounded radius per sphere: ``W = r0 exp(s) W_hat``.

    ``W_hat`` takes the MuonH hyperball step. The log-radius ``s`` (zero at init, ``r0`` the init norm) takes an Adam
    step at ``learning_rate`` on the radial gradient ``<dL/dW, W>``; a step away from ``s = 0`` is scaled by
    ``1 - (s / softcap)**2`` and a step toward it is taken in full, and ``s`` is clipped to ``[-softcap, softcap]``.
    """
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        pre_norm=pre_norm,
    )

    def init_fn(params):
        init_radius = jax.tree.map(_slice_norm, params)
        zeros = jax.tree.map(jnp.zeros_like, init_radius)
        return RadialMuonHState(
            muon=muon_transform.init(params),
            count=jnp.zeros([], jnp.int32),
            init_radius=init_radius,
            log_radius=zeros,
            mu=zeros,
            nu=zeros,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh_radial requires params")
        directions, muon_state = muon_transform.update(updates, state.muon, params)
        sphere_steps = _scale_invariant_hyperball_updates(params, directions, learning_rate)
        grads = _match_named_sharding_to_params(updates, params)
        radial = jax.tree.map(lambda g, p: _slice_sum(g.astype(jnp.float32) * p.astype(jnp.float32)), grads, params)
        count = optax.safe_increment(state.count)
        mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, radial)
        nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.nu, radial)
        mu_hat = otu.tree_bias_correction(mu, b1, count)
        nu_hat = otu.tree_bias_correction(nu, b2, count)

        def step_log_radius(s, m, v):
            step = -learning_rate * m / (jnp.sqrt(v) + eps)
            outward = step * s > 0
            step = jnp.where(outward, step * jnp.maximum(1.0 - jnp.square(s / softcap), 0.0), step)
            return jnp.clip(s + step, -softcap, softcap)

        log_radius = jax.tree.map(step_log_radius, state.log_radius, mu_hat, nu_hat)

        def rescale(p, sphere_step, r0, s):
            stepped = _pin_sharding(p.astype(jnp.float32) + sphere_step.astype(jnp.float32), p)
            target = r0 * jnp.exp(s)
            return (stepped * (target / jnp.maximum(_slice_norm(stepped), 1e-10)) - p).astype(p.dtype)

        new_updates = jax.tree.map(rescale, params, sphere_steps, state.init_radius, log_radius)
        return new_updates, RadialMuonHState(muon_state, count, state.init_radius, log_radius, mu, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def _multi_transform_when_used(
    transforms: dict[str, optax.GradientTransformation],
    optional: dict[str, optax.GradientTransformation],
    labels_fn,
) -> optax.GradientTransformationExtraArgs:
    """``optax.multi_transform`` over ``transforms`` plus each ``optional`` group that labels some parameter, so a model
    without that parameter keeps its optimizer state unchanged."""

    def build(params) -> optax.GradientTransformationExtraArgs:
        used = set(jax.tree.leaves(labels_fn(params)))
        groups = {**transforms, **{name: tx for name, tx in optional.items() if name in used}}
        return optax.multi_transform(
            {name: _named_scope_transform(f"opt_{name}", tx) for name, tx in groups.items()}, labels_fn
        )

    def init_fn(params):
        return build(params).init(params)

    def update_fn(updates, state, params=None, **extra_args):
        return build(params).update(updates, state, params, **extra_args)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _is_matrix_stack(x) -> bool:
    return hasattr(x, "ndim") and x.ndim >= 2


ATTN_RES_QUERY_LR_SCALE = 0.1
"""Adam LR of the AttnRes pseudo-queries as a multiple of ``adam_lr``."""
KDA_BETA_LR_MULT = 2.0
"""MuonH LR multiple of the ``kda_beta`` group (the KDA write-strength projection)."""
_ADAM_LEAVES = re.compile(
    r"(?:^|\.)(value_embed|ve_lambda|ve_gate|head_ve_lambda|head_ve_gate|head_ve_gain|qk_mult"
    r"|pointer_w|pointer_b|pointer_open)$"
)
# ``full_bandwidth_passes``' zero-initialised W_fuse: plain Muon at the MuonH LR (a hyperball step keeps a zero matrix at
# zero), in its own group present only when the model has it.
_FULL_BANDWIDTH_FUSE = re.compile(r"(?:^|\.)full_bandwidth_fuse$")
FULL_BANDWIDTH_MUON_GROUP = "full_bandwidth_muon"
# Matrices whose output passes through a scale-removing norm, so their radial gradient is zero: the softmax layers'
# q/k projections under the q/k norms (``qk_norm``, or ``mla_q_norm``/``mla_k_norm``), the MLA KV down-projection
# (``kv_latent_norm``) and the MoE latent down-projection (``latent_norm``). They keep plain MuonH under
# ``muonh_radial_softcap``.
_RADIUS_FROZEN = re.compile(r"(?:^|\.)(?:stacked_blocks\.stacked\.attn\.(?:w_q|w_k|w_uk|w_dkv)|mlp\.w_latent_down)$")


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
    coefficient_type: CoefficientType = "quintic"
    gate_router_weight_decay: float = 0.02
    kda_beta_group: str = "kda_beta"
    muon_pre_norm: str = "none"
    """MuonH momentum normalization before Newton-Schulz: ``none``, ``out`` or ``in`` (MuonEq)."""
    muonh_decay_power: float | None = None
    """Decay shape of the MuonH LR after warmup: ``floor + (peak - floor) * (1 - p**power)`` over training
    progress ``p`` (modded-nanogpt MuonH records #345/#351). None keeps the shared schedule (linear = 1)."""
    adam_ademamix_alpha: float = 0.0
    """AdEMAMix (arXiv 2409.03152) on the plain-Adam groups: weight of a slow gradient EMA added to Adam's
    first moment, warmed up linearly over the run (0: off; the paper uses 5-8)."""
    adam_ademamix_beta3: float = 0.999
    """Decay of the AdEMAMix slow EMA, warmed up over ``adam_ademamix_warmup`` as in the paper."""
    router_lr_mult: float = 1.0
    """LR multiplier of the router weights alone (of ``adam_lr`` or, with ``router_group=muonh``, the MuonH LR)."""
    embed_optimizer: EmbedOptimizer = EmbedOptimizer.ADAM_GROUP
    """How ``token_embed`` (the input table alone) is optimized, at the Adam LR (``EmbedOptimizer``)."""
    embed_weight_decay: float = 0.0
    """Fixed (not annealed) decoupled decay of ``token_embed``, ``p <- p (1 - adam_lr_t wd) - adam_lr_t u``; needs
    ``embed_optimizer`` ``ademamix`` or ``adam``."""
    embed_lr_mult: float | None = None
    """``token_embed`` (the input table alone) in its own ``embed_lr`` group, identical to the ``adam`` group it
    otherwise shares (schedule, moments, decoupled decay) at ``embed_lr_mult`` x ``adam_lr``. None: off. Needs
    ``embed_optimizer`` ``adam_group``."""
    muonh_radial_softcap: float = 0.0
    """Softcap radial MuonH: each MuonH sphere learns a bounded log-radius ``s``, ``|s| <= muonh_radial_softcap``
    (``scale_with_grug_muonh_radial``), at the group's MuonH LR, in the ``muonh_radial`` and ``kda_beta_radial``
    groups. The ``_RADIUS_FROZEN`` matrices keep plain MuonH. 0: off."""

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

        def muonh_radial_at(lr):
            return optax.chain(
                scale_with_grug_muonh_radial(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=lr,
                    coefficient_type=self.coefficient_type,
                    pre_norm=self.muon_pre_norm,
                    softcap=self.muonh_radial_softcap,
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                ),
                _match_named_update_sharding(),
            )

        def muonh_at(lr):
            return optax.chain(
                scale_with_grug_muonh(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=lr,
                    coefficient_type=self.coefficient_type,
                    pre_norm=self.muon_pre_norm,
                ),
                _match_named_update_sharding(),
            )

        def muon_at(lr):
            return optax.chain(
                _grug_scale_with_muon(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    coefficient_type=self.coefficient_type,
                    pre_norm=self.muon_pre_norm,
                ),
                optax.scale(-lr),
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
                adam = _scale_by_adam_decoupled_decay(adam, self.gate_router_weight_decay, num_train_steps)
            return optax.chain(adam, optax.scale(-lr))

        def embed_at(lr):
            if self.embed_optimizer == EmbedOptimizer.ROW_ADAMH:
                return optax.chain(scale_by_row_sphere_adam(self.beta1, self.beta2, self.epsilon, lr))
            if self.embed_optimizer == EmbedOptimizer.ADAM:
                moments = optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)
            else:
                moments = adam_moments(self.beta1, self.beta2)
            decay = (_fixed_decoupled_decay(self.embed_weight_decay),) if self.embed_weight_decay else ()
            return optax.chain(moments, *decay, optax.scale(-lr))

        def optimizer(learning_rate, adam_lr):
            transforms = {
                "muonh": muonh_at(learning_rate),
                "adamh": optax.chain(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)),
                "adam": adam_at(adam_lr, decay=True),
                "attn_res_query": adam_at(adam_lr * ATTN_RES_QUERY_LR_SCALE),
                "kda_beta": muonh_at(learning_rate * KDA_BETA_LR_MULT),
                "kda_beta_adam": adam_at(adam_lr, decay=True),
                "kda_decay": adam_at(adam_lr),
                # The entropy head and the skip router: no decay pull. (Group names stay as the checkpoints
                # of earlier runs store them; "embed2" holds no parameter.)
                "null_gain": adam_at(adam_lr),
                "embed2": adam_at(adam_lr),
                "router": adam_at(adam_lr * self.router_lr_mult, decay=True),
                "embed": embed_at(adam_lr),
            }
            # Only when on, so the optimizer state of runs without it keeps its groups.
            if self.embed_lr_mult is not None:
                transforms["embed_lr"] = adam_at(adam_lr * self.embed_lr_mult, decay=True)
            if self.muonh_radial_softcap:
                transforms["muonh_radial"] = muonh_radial_at(learning_rate)
                transforms["kda_beta_radial"] = muonh_radial_at(learning_rate * KDA_BETA_LR_MULT)
            return _multi_transform_when_used(
                transforms, {FULL_BANDWIDTH_MUON_GROUP: muon_at(learning_rate)}, self.create_mask
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def __post_init__(self):
        if self.kda_beta_group not in ("kda_beta", "adam"):
            raise ValueError(f"kda_beta_group must be kda_beta or adam, got {self.kda_beta_group!r}")
        if self.embed_weight_decay and self.embed_optimizer not in (EmbedOptimizer.ADEMAMIX, EmbedOptimizer.ADAM):
            raise ValueError("embed_weight_decay needs embed_optimizer ademamix or adam")
        if self.embed_lr_mult is not None and (
            self.embed_lr_mult <= 0 or self.embed_optimizer != EmbedOptimizer.ADAM_GROUP
        ):
            raise ValueError("embed_lr_mult needs a positive multiplier and embed_optimizer adam_group")
        if self.muonh_radial_softcap < 0:
            raise ValueError("muonh_radial_softcap must be non-negative")

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            label = group_fn(param, path)
            if not self.muonh_radial_softcap or label not in ("muonh", "kda_beta") or param.ndim < 2:
                return label
            path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
            return label if _RADIUS_FROZEN.search(path_lower) else f"{label}_radial"

        def group_fn(param, path):
            path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
            if _FULL_BANDWIDTH_FUSE.search(path_lower):
                return FULL_BANDWIDTH_MUON_GROUP
            kda_leaf = _kda_leaf(path_lower)
            if kda_leaf == _KDA_BETA_LEAF:
                return "kda_beta_adam" if self.kda_beta_group == "adam" else self.kda_beta_group
            if kda_leaf in _KDA_ADAM_LEAVES:
                return "kda_decay"
            if re.search(r"entropy_head|skip_router", path_lower):
                return "null_gain"
            # AttnRes pseudo-queries are per-layer vectors (2D once stacked, which would route to MuonH).
            if "attn_res_query" in path_lower:
                return "attn_res_query"
            # Inkling rel-pos weights; value embeddings and their mixing weights; small per-layer gates; the pointer
            # mixture's weights (zero-initialised).
            if ".rel_pos." in path_lower or _ADAM_LEAVES.search(path_lower):
                return "adam"
            if _TOKEN_EMBED.search(path_lower) and self.embed_lr_mult is not None:
                return "embed_lr"
            if _TOKEN_EMBED.search(path_lower) and self.embed_optimizer != EmbedOptimizer.ADAM_GROUP:
                return "embed"
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
