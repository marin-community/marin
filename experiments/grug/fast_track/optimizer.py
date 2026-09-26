# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
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
# output-gate matrices take the MuonH catch-all, the ShortConv kernels and output-norm scale are
# ``.weight`` leaves (Adam).
_KDA_ATTN_LEAF = re.compile(r"kda_blocks\.stacked\.attn\.(\w+)")
# Low-rank forget gate, per-head A_log and per-channel dt_bias: Adam (no weight decay).
_KDA_ADAM_LEAVES = frozenset({"w_a_down", "w_a_up", "a_log", "dt_bias", "push_decay", "w_push"})
# Write-strength projection: MuonH at ``kda_beta_lr_mult`` x the MuonH LR.
_KDA_BETA_LEAF = "w_beta"
# Low-rank write-strength MLP (``kda_beta_rank``): LR group chosen by ``kda_beta_mlp_group``.
_KDA_BETA_MLP_LEAVES = frozenset({"w_beta_down", "w_beta_up"})


# Matrix families that ``okls_targets`` can move from MuonH to the OKLS direction.
_OKLS_FAMILIES: dict[str, re.Pattern] = {
    "attn": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(q|k|v|o|g|dkv|uk|uv)$"),
    "routed": re.compile(r"\.mlp\.expert_mlp\.w_(gate|up|down)$"),
    "shared": re.compile(r"\.shared\.\d+\.w_(gate|up|down)$"),
    "latent": re.compile(r"\.mlp\.w_latent_(down|up)$"),
    "gated_norm": re.compile(r"gated_norm\.w_(down|up)$"),
    # Per-projection subsets of "attn" (KDA and MLA together).
    "attn_q": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_q$"),
    "attn_k": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(k|uk)$"),
    "attn_v": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(v|uv)$"),
    "attn_o": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_o$"),
    "attn_other": re.compile(r"(stacked_blocks|kda_blocks)\.stacked\.attn\.w_(g|dkv)$"),
}


def _kda_leaf(path_lower: str) -> str | None:
    match = _KDA_ATTN_LEAF.fullmatch(path_lower)
    return None if match is None else match.group(1)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for exactly the ``attn_gate`` and MoE ``router`` weight leaves.

    Matches the leaf attribute name at the end of the path, so it selects ``...attn.attn_gate`` and
    ``...mlp.router`` but not the separate ``...mlp.router_bias`` leaf.
    """
    return path_lower.endswith((".attn_gate", ".router", ".router_down", ".router_up"))


def _gate_router_decay_mask(params):
    """Boolean pytree that is True on the ``attn_gate`` and ``router`` weight leaves -- the ones that
    receive decoupled weight decay -- and False everywhere else."""
    paths = leaf_key_paths(params)

    def is_target(_, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        return _is_gate_or_router_weight(path_str.lower())

    return jax.tree.map(is_target, params, paths)


def _scale_by_adam_gate_router_decay(
    b1: float, b2: float, eps: float, weight_decay: float, total_steps: int
) -> optax.GradientTransformation:
    """``scale_by_adam`` plus decoupled weight decay on ``attn_gate`` and the ``router`` weight,
    annealed linearly to 0 over ``total_steps``. The coefficient reads the Adam ``count`` and the
    state stays ``ScaleByAdamState``, so a checkpoint written without decay resumes at the right step
    with its moments intact."""
    adam = optax.scale_by_adam(b1, b2, eps)

    def init_fn(params):
        return adam.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_scale_by_adam_gate_router_decay requires params for decoupled decay")
        step = state.count
        updates, next_state = adam.update(updates, state, params)
        wd = weight_decay * jnp.clip(1.0 - step / total_steps, 0.0, None)
        mask = _gate_router_decay_mask(params)
        updates = jax.tree.map(lambda u, p, keep: u + wd * p if keep else u, updates, params, mask)
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
    """LR group of ``output_proj``: ``adamh`` or ``muonh``."""
    embed_group: str = "adam"
    """LR group of ``token_embed``: ``adam`` or ``adamh``."""
    hyperball_per_expert: bool = False
    """One MuonH hyperball (Frobenius sphere) per routed expert instead of per layer's expert stack."""
    neuron_norm_beta2: float | None = None
    """NorMuon neuron-wise normalization of the MuonH direction with this second-moment decay (None: off)."""
    """Orthogonalize the attention projections per head of this width (None: whole matrices)."""

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=lr,
                        coefficient_type=self.coefficient_type,
                        head_dim=self.muon_head_dim,
                        neuron_norm_beta2=self.neuron_norm_beta2,
                        hyperball_per_expert=self.hyperball_per_expert,
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

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                if self.gate_router_weight_decay > 0.0:
                    components.append(
                        _scale_by_adam_gate_router_decay(
                            self.beta1, self.beta2, self.epsilon, self.gate_router_weight_decay, num_train_steps
                        )
                    )
                else:
                    components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            def plain_adam_at(lr, beta1=None, beta2=None):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    optax.scale_by_adam(
                        self.beta1 if beta1 is None else beta1, self.beta2 if beta2 is None else beta2, self.epsilon
                    )
                )
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
                "kda_decay": plain_adam_at(adam_lr * self.kda_decay_lr_mult, self.kda_decay_beta1, self.kda_decay_beta2),
            }
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def __post_init__(self):
        if self.lm_head_group not in ("adamh", "muonh"):
            raise ValueError(f"lm_head_group must be adamh or muonh, got {self.lm_head_group!r}")
        if self.kda_beta_mlp_group not in ("kda_beta", "adam"):
            raise ValueError(f"kda_beta_mlp_group must be kda_beta or adam, got {self.kda_beta_mlp_group!r}")
        if self.embed_group not in ("adam", "adamh"):
            raise ValueError(f"embed_group must be adam or adamh, got {self.embed_group!r}")

    def create_mask(self, params):
        paths = leaf_key_paths(params)
        unknown = set(self.okls_targets) - set(_OKLS_FAMILIES)
        if unknown:
            raise ValueError(f"unknown okls_targets {sorted(unknown)}; choose from {sorted(_OKLS_FAMILIES)}")

        def mask_fn(param, path):
            group = _base_group(param, path)
            if group == "muonh":
                path_lower = (".".join(path) if isinstance(path, (list, tuple)) else str(path)).lower()
                if any(_OKLS_FAMILIES[f].search(path_lower) for f in self.okls_targets):
                    return "okls"
                if self.muonh_attn_lr_mult != 1.0 and _OKLS_FAMILIES["attn"].search(path_lower):
                    return "muonh_attn"
                if self.muonh_routed_lr_mult != 1.0 and _OKLS_FAMILIES["routed"].search(path_lower):
                    return "muonh_routed"
            return group

        def _base_group(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
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
            if ".rel_pos." in path_lower or re.search(r"\.(value_embed|ve_lambda|ve_gate|bias_\w+)$", path_lower):
                return "adam"
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
