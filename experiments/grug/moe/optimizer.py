# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe.klsoaph import scale_by_klsoaph

# softplus^{-1}(1): raw gain init so gamma = softplus(ghat) = 1.0 at start (Häggström decoupling).
_GAIN_INIT = math.log(math.e - 1.0)


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
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


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(
        scale_invariant_update,
        params,
        direction_updates,
        is_leaf=lambda x: x is None,
    )


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        muon_updates, next_state = muon_transform.update(updates, state, params)
        muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate)
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def _is_decouple_matrix(p) -> bool:
    return p is not None and hasattr(p, "ndim") and p.ndim >= 2


class _DecoupledGainState(NamedTuple):
    """State for magnitude-direction decoupled MuonH.

    ``muon`` holds the direction (W_hat) Muon momentum buffer. ``ghat_row``/``ghat_col`` are the
    RAW per-row/per-col gains (gamma = softplus(ghat)); ``gm_*``/``gv_*`` are their Adam moments.
    All gain pytrees mirror the params tree, with non-matrix leaves = None.
    """

    count: jax.Array
    muon: optax.OptState
    ghat_row: optax.Updates
    ghat_col: optax.Updates
    gm_row: optax.Updates
    gv_row: optax.Updates
    gm_col: optax.Updates
    gv_col: optax.Updates


def scale_with_grug_muonh_decoupled(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    gain_lr: float = 1e-3,
    gain_beta1: float = 0.9,
    gain_beta2: float = 0.95,
    gain_eps: float = 1e-8,
    gain_mode: str = "both",
    coefficient_type: CoefficientType = "quintic",
) -> optax.GradientTransformation:
    """Magnitude-direction decoupled MuonH (Häggström): W = diag(gamma_row) W_hat diag(gamma_col).

    Each matrix is held fused as W. Per step: recover the on-sphere direction W_hat = W / gain;
    split the gradient into a direction part (gain * G, stepped by MuonH's NS + Frobenius-hyperball
    step, which preserves ||W_hat|| -> stays on the sphere) and per-row/col gain parts
    (reduce(W_hat * G), stepped by Adam at its own LR through a softplus reparam); reassemble
    W = gain * W_hat. ``gain_mode`` in {both,row,col}. No weight decay / warmup needed.
    """
    if gain_mode not in ("both", "row", "col"):
        raise ValueError(f"gain_mode must be both|row|col, got {gain_mode!r}")
    use_row = gain_mode in ("both", "row")
    use_col = gain_mode in ("both", "col")
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def none_leaf(x):
        return x is None

    def _row_init(p):
        return jnp.full(p.shape[:-1], _GAIN_INIT, jnp.float32) if _is_decouple_matrix(p) else None

    def _col_init(p):
        return jnp.full((*p.shape[:-2], p.shape[-1]), _GAIN_INIT, jnp.float32) if _is_decouple_matrix(p) else None

    def _zeros_like_tree(t):
        return jax.tree.map(lambda x: None if x is None else jnp.zeros_like(x), t, is_leaf=none_leaf)

    def init_fn(params):
        ghat_row = jax.tree.map(_row_init, params, is_leaf=none_leaf)
        ghat_col = jax.tree.map(_col_init, params, is_leaf=none_leaf)
        return _DecoupledGainState(
            count=jnp.zeros([], jnp.int32),
            muon=muon_transform.init(params),
            ghat_row=ghat_row,
            ghat_col=ghat_col,
            gm_row=_zeros_like_tree(ghat_row),
            gv_row=_zeros_like_tree(ghat_row),
            gm_col=_zeros_like_tree(ghat_col),
            gv_col=_zeros_like_tree(ghat_col),
        )

    def _gain_from_raw(ghr, ghc, ref):
        # gain[..., r, c] = (use_row ? softplus(ghr)[...,:,None] : 1) * (use_col ? softplus(ghc)[...,None,:] : 1)
        if not _is_decouple_matrix(ref):
            return None
        g = jnp.ones(ref.shape, jnp.float32)
        if use_row:
            g = g * jax.nn.softplus(ghr)[..., :, None]
        if use_col:
            g = g * jax.nn.softplus(ghc)[..., None, :]
        return g

    def _adam_gain(ghat, g_gamma, m, v, ref_use, count):
        # Step the raw gain by Adam through the softplus reparam: grad_ghat = sigmoid(ghat) * dL/dgamma.
        if ghat is None or not ref_use:
            return ghat, m, v
        grad = jax.nn.sigmoid(ghat) * g_gamma
        new_m = gain_beta1 * m + (1.0 - gain_beta1) * grad
        new_v = gain_beta2 * v + (1.0 - gain_beta2) * jnp.square(grad)
        mhat = new_m / (1.0 - gain_beta1**count)
        vhat = new_v / (1.0 - gain_beta2**count)
        new_ghat = ghat - gain_lr * mhat / (jnp.sqrt(vhat) + gain_eps)
        return new_ghat, new_m, new_v

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh_decoupled requires params")
        count = optax.safe_increment(state.count)
        gain = jax.tree.map(_gain_from_raw, state.ghat_row, state.ghat_col, params, is_leaf=none_leaf)
        w_hat = jax.tree.map(lambda w, g: None if g is None else w / g, params, gain, is_leaf=none_leaf)
        # direction gradient G_hat = gain * G; gain gradients dL/dgamma = reduce(W_hat * G)
        g_hat = jax.tree.map(lambda gr, g: None if g is None else gr * g, updates, gain, is_leaf=none_leaf)
        wg = jax.tree.map(lambda wh, gr: None if wh is None else wh * gr, w_hat, updates, is_leaf=none_leaf)
        g_gamma_row = jax.tree.map(lambda x: None if x is None else jnp.sum(x, axis=-1), wg, is_leaf=none_leaf)
        g_gamma_col = jax.tree.map(lambda x: None if x is None else jnp.sum(x, axis=-2), wg, is_leaf=none_leaf)
        # MuonH direction step on W_hat (NS-orthogonalize the momentum of g_hat, then norm-preserving hyperball)
        direction, new_muon = muon_transform.update(g_hat, state.muon, w_hat)
        hb = _scale_invariant_hyperball_updates(w_hat, direction, learning_rate)
        w_hat_new = jax.tree.map(lambda wh, d: None if wh is None else wh + d, w_hat, hb, is_leaf=none_leaf)
        # Adam step the gains (own LR, softplus reparam)
        row_out = jax.tree.map(
            lambda gh, gg, m, v: _adam_gain(gh, gg, m, v, use_row, count),
            state.ghat_row,
            g_gamma_row,
            state.gm_row,
            state.gv_row,
            is_leaf=none_leaf,
        )
        col_out = jax.tree.map(
            lambda gh, gg, m, v: _adam_gain(gh, gg, m, v, use_col, count),
            state.ghat_col,
            g_gamma_col,
            state.gm_col,
            state.gv_col,
            is_leaf=none_leaf,
        )

        def _is_adam_triple(x):
            # The (ghat, m, v) tuples from _adam_gain. Length-3 guard avoids matching optax's
            # MaskedNode (an empty NamedTuple) for leaves outside the muonh group under multi_transform.
            return x is None or (isinstance(x, tuple) and len(x) == 3)

        def unpack(tree, i):
            return jax.tree.map(lambda x: None if x is None else x[i], tree, is_leaf=_is_adam_triple)

        ghat_row_new, gm_row_new, gv_row_new = unpack(row_out, 0), unpack(row_out, 1), unpack(row_out, 2)
        ghat_col_new, gm_col_new, gv_col_new = unpack(col_out, 0), unpack(col_out, 1), unpack(col_out, 2)
        # reassemble W = gain_new * W_hat_new; emit delta = W_new - W
        gain_new = jax.tree.map(_gain_from_raw, ghat_row_new, ghat_col_new, params, is_leaf=none_leaf)
        w_new = jax.tree.map(lambda g, wh: None if g is None else g * wh, gain_new, w_hat_new, is_leaf=none_leaf)
        delta = jax.tree.map(lambda wn, w: None if wn is None else wn - w, w_new, params, is_leaf=none_leaf)
        return delta, _DecoupledGainState(
            count, new_muon, ghat_row_new, ghat_col_new, gm_row_new, gv_row_new, gm_col_new, gv_col_new
        )

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate_up,
      mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer: 3 LR groups (muonh / adamh / adam).

    Three LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all GatedNorms.
      Newton-Schulz orthogonalisation + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate``
      / 1-D norm weights.

    ``max_grad_norm`` defaults to ``None`` here (no clipping) for the 1pct-noclip
    schedule used by the May Recipe baseline.
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
    # Magnitude-direction decoupling (Häggström): factorize each muonh matrix as W = diag(gamma_row) W_hat
    # diag(gamma_col) with W_hat on a fixed-norm sphere and learnable per-row/col gains (own Adam LR).
    decouple_gains: bool = False
    gain_lr: float = 1e-3
    gain_mode: str = "both"  # both | row | col

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                if self.decouple_gains:
                    components.append(
                        scale_with_grug_muonh_decoupled(
                            momentum=self.momentum,
                            nesterov=self.nesterov,
                            steps=self.backend_steps,
                            muon_eps=self.muon_epsilon,
                            learning_rate=learning_rate,
                            gain_lr=self.gain_lr,
                            gain_beta1=self.beta1,
                            gain_beta2=self.beta2,
                            gain_eps=self.epsilon,
                            gain_mode=self.gain_mode,
                            coefficient_type=self.coefficient_type,
                        )
                    )
                else:
                    components.append(
                        scale_with_grug_muonh(
                            momentum=self.momentum,
                            nesterov=self.nesterov,
                            steps=self.backend_steps,
                            muon_eps=self.muon_epsilon,
                            learning_rate=learning_rate,
                            coefficient_type=self.coefficient_type,
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
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "router_bias" in path_lower
                or path_lower.endswith(".attn_gate")
                or ".router" in path_lower
            ):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "scale_with_grug_muonh",
]


def _uses_klsoaph_baseline_adam_group(path_lower: str) -> bool:
    """KL Soap H variant override: route attn_gate into the matrix group.

    Same as the MuonH baseline-Adam predicate but drops "attn_gate" — the
    attention gate (hidden_dim, num_heads) is kept under the SOAP preconditioner.
    """
    return "token_embed" in path_lower or "router_bias" in path_lower or ".router" in path_lower


def scale_with_grug_klsoaph(
    beta1: float = 0.95,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 1,
    init_factor: float = 0.1,
    identity_init: bool = False,
    reparam_eig: bool = False,
    nesterov: bool = False,
    soap_muon: bool = False,
    kl: bool = True,
    block_size: int = 0,
    use_scqr: bool = False,
    learning_rate: float = 0.018,
) -> optax.GradientTransformation:
    """KL Soap H: full-matrix SOAP-eigenbasis Adam direction + hyperball post-step.

    Reproduces KLSOAPH from KellerJordan/modded-nanogpt PR #290 on the full
    per-leaf matrix (no block tiling). The scale-invariant ("hyperball")
    post-step normalizes the full update. Default (beta1, beta2, shampoo_beta)
    = (0.95, 0.9, 0.9) matches upstream's passing tuple; precond_freq=1.

    ``kl=False`` gives non-KL SOAP-H: the Gram is accumulated from RAW gradient
    outer products (G Gᵀ, Gᵀ G) as in canonical SOAP (Vyas et al.), with no ESI
    eigenvalue-whitening of the Gram. The hyperball ("H") post-step is unchanged.
    """
    soap_transform = scale_by_klsoaph(
        beta1=beta1,
        beta2=beta2,
        shampoo_beta=shampoo_beta,
        eps=eps,
        precond_freq=precond_freq,
        init_factor=init_factor,
        identity_init=identity_init,
        reparam_eig=reparam_eig,
        nesterov=nesterov,
        soap_muon=soap_muon,
        kl=kl,
        block_size=block_size,
        use_scqr=use_scqr,
    )

    def init_fn(params):
        return soap_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_klsoaph requires params for norm-preserving updates")
        direction, next_state = soap_transform.update(updates, state, params)
        klsoaph_updates = _scale_invariant_hyperball_updates(params, direction, learning_rate)
        return klsoaph_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_klsoaph_v1")
@dataclass(frozen=True)
class GrugMoeKLSoapHConfig(OptimizerConfig):
    """KL Soap H for Grug MoE (full-matrix SOAP + hyperball).

    Reproduces KLSOAPH from KellerJordan/modded-nanogpt PR #290. "KL" names the
    scale-invariant ("hyperball") post-step. Routing mirrors MuonH but routes
    attn_gate into the SOAP matrix group:
    - klsoaph: matrix leaves (incl. attn_gate, gated_norm, experts)
    - adamh: lm_head / output_proj
    - adam: token_embed / router / router_bias
    """

    adam_lr: float = 6e-4
    # SOAP eigenbasis: (beta1, beta2, shampoo_beta) = upstream "passing" tuple from PR #290.
    # beta1 = projected-momentum EMA, beta2 = projected-Adam 2nd-moment, shampoo_beta = Gram EMA.
    beta1: float = 0.95
    beta2: float = 0.9
    shampoo_beta: float = 0.9
    epsilon: float = 1e-8  # SOAP eigenbasis Adam eps (upstream PR #290)
    precond_freq: int = 1
    init_factor: float = 0.1
    identity_init: bool = False  # skip eigh at step 1 (identity eigenbasis) -> faster compile
    reparam_eig: bool = False  # eigenvalues from fresh-basis Gram diag at refresh -> high precond_freq loss-neutral
    nesterov: bool = False  # Nadam-style look-ahead: precond numerator = b1*m_t+(1-b1)*g_t instead of plain EMA
    soap_muon: bool = False  # SOAP-Muon (modded-nanogpt PR #278/#321): msign the Adam-precond update after rotate-back
    kl: bool = True  # False -> non-KL SOAP-H: raw Gram (G Gᵀ, Gᵀ G), no ESI whitening (canonical SOAP) + hyperball
    block_size: int = 0  # >0 -> block-wise SOAP: per bxb tile preconditioner (higher MFU, lower memory)
    use_scqr: bool = False  # refresh eigenbasis via matmul Cholesky-QR (MXU-fast) instead of jnp.linalg.qr
    # Real-Adam settings for the NON-SOAP groups (adamh: lm_head/output_proj; adam: embeddings/router),
    # kept SEPARATE from the SOAP eigenbasis betas/eps above so the SOAP group is the only variable
    # vs the MuonH baseline. Defaults match the d512 MuonH run (heuristic beta1=0.9062, beta2=0.999,
    # eps=1.01e-15); the launcher pins them from the MuonH heuristic to stay apples-to-apples.
    adam_beta1: float = 0.9062
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-15
    max_grad_norm: float | None = None
    # SOAP-group warmup, SEPARATE from the inherited ``warmup`` (which governs the adamh/adam groups).
    # The SOAP group has an early preconditioner-estimation lag, so it may want a longer warmup than the
    # Adam groups. Defaults to 0.01 (= MuonH) so behavior matches the Adam warmup unless overridden.
    klsoaph_warmup: float = 0.01

    def build(self, num_train_steps):
        # adamh/adam groups use the inherited ``warmup``; the klsoaph (SOAP) group gets its own warmup.
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        klsoaph_lr_schedule = replace(self, warmup=self.klsoaph_warmup).lr_scheduler(num_train_steps)

        def optimizer(klsoaph_lr, learning_rate, adam_lr):
            def klsoaph_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_klsoaph(
                        beta1=self.beta1,
                        beta2=self.beta2,
                        shampoo_beta=self.shampoo_beta,
                        eps=self.epsilon,
                        precond_freq=self.precond_freq,
                        init_factor=self.init_factor,
                        identity_init=self.identity_init,
                        reparam_eig=self.reparam_eig,
                        nesterov=self.nesterov,
                        soap_muon=self.soap_muon,
                        kl=self.kl,
                        block_size=self.block_size,
                        use_scqr=self.use_scqr,
                        learning_rate=klsoaph_lr,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.adam_beta1, self.adam_beta2, self.adam_epsilon, learning_rate))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.adam_beta1, self.adam_beta2, self.adam_epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "klsoaph": klsoaph_transform(),
                    "adamh": adamh_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            klsoaph_lr=klsoaph_lr_schedule,
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_klsoaph_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "klsoaph"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)
