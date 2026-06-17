# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delayed-gradient optimizer wrappers for studying pipeline-parallel staleness.

Pipeline-parallel training with a throughput-optimal async schedule applies a
gradient that was computed ``tau`` weight-versions ago. We study that regime
*without building pipeline parallelism* by injecting a controlled gradient delay
in software: the wrapper keeps a depth-``tau`` FIFO of past gradients (and the
weights they were computed at) inside the optimizer state, and feeds the inner
optimizer the *stale* gradient each step. This exactly reproduces constant-delay
asynchronous SGD.

The FIFO and any correction statistics live in ``opt_state`` — they are the
"O(weights) extra optimizer state" we are budgeting for — so the canonical grug
train loop needs no changes; we only swap the optimizer config. At ``tau == 0``
the wrapper is a pass-through and is bit-identical to the inner optimizer.

Correctors (applied to the stale gradient before the inner optimizer sees it):

- ``none``        — naive async SGD; apply the stale gradient as-is.
- ``dc_asgd``     — DC-ASGD delay compensation (Zheng et al. 2017):
                    ``g + lambda * (g (.) g) (.) (w_t - w_stale)`` using the
                    instantaneous squared stale gradient as the diagonal-Hessian
                    proxy.
- ``dc_asgd_ema`` — the same correction but with the diagonal curvature read from
                    an EMA of the squared gradient (i.e. Adam/RMSProp's second
                    moment ``v_t``). This tests the "reuse the preconditioner
                    state as the curvature term, near-free" hypothesis.

For Muon the corrected gradient is fed *before* Newton-Schulz orthogonalization,
so the correction acts on the momentum/direction and orthogonalization then
renormalizes the magnitude.

``weight_pred`` is different in kind: it is a *forward-side* corrector, not a
gradient correction. Instead of patching the stale gradient, it asks the train
step to evaluate the gradient at *predicted* weights ``W_hat = w - tau*lr*dW``
(extrapolating the most recent applied update ``dW`` forward by ``tau`` steps) so
that, once the gradient is delayed and applied, it lands on the weights it was
meant for. For Muon ``dW`` is the orthogonalized update (post-orthogonalization
prediction). The optimizer cannot do this alone — it exposes the predicted offset
via :meth:`make_forward_predictor` and the grug train step computes the forward
there; see ``experiments/grug/moe/train.py``.

Muon-specific forward predictors (``wp_*``) — a pipeline-parallel Muon variant.
Newton-Schulz decouples the update magnitude from the gradient and makes the
orthogonalized *direction* noisy step-to-step, so the post-orthogonalization
``weight_pred`` extrapolates a jittery signal. These predict instead along the
*smoothed raw momentum* ``m_raw`` (an EMA of the stale gradient, maintained in
the wrapper — pre-orthogonalization), pointed in the descent direction ``-m_raw``
and rescaled to the realized per-leaf step RMS:

- ``wp_preorth``    — raw-momentum-direction prediction (the base PP-Muon mode).
- ``wp_cautious``   — ``wp_preorth`` with a Cautious-Optimizer sign gate: zero the
                      coordinates where the predicted descent ``-m_raw`` disagrees
                      with the realized update, rescaling surviving mass.
- ``wp_trust``      — ``wp_preorth`` with a LARS-style per-leaf trust-ratio clamp:
                      cap the offset RMS at ``trust * rms(param)`` (scale-free,
                      since Muon's magnitude is normalized).
- ``wp_confidence`` — ``wp_preorth`` scaled by ``relu(cos(-m_raw, last_update))``:
                      a soft agreement gate that shrinks the prediction when the
                      momentum and the realized update directions diverge.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import GrugMuonConfig
from optax import tree_utils as otu

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig

CORRECTORS = (
    "none",
    "dc_asgd",
    "dc_asgd_ema",
    "weight_pred",
    "lr_damp",
    "wp_preorth",
    "wp_cautious",
    "wp_trust",
    "wp_confidence",
)

# Forward-side correctors: they emit a predicted-weight offset and leave the stale
# gradient untouched (the correction happens in the forward evaluation, not the
# gradient). The ``wp_*`` family also maintains the raw-momentum buffer ``m_raw``.
_FORWARD_PRED_CORRECTORS = ("weight_pred", "wp_preorth", "wp_cautious", "wp_trust", "wp_confidence")
_PREORTH_CORRECTORS = ("wp_preorth", "wp_cautious", "wp_trust", "wp_confidence")
_EPS = 1e-12

# Grug Transformer fields on the input side of the pipeline (forward first -> first
# PP stage -> stalest) and the output side (forward last -> last stage -> fresh).
_INPUT_FIELDS = ("token_embed", "embed_norm", "embed_gated_norm")
_OUTPUT_FIELDS = ("output_proj", "final_norm", "final_gated_norm")


def grug_stage_tau(num_layers: int, num_stages: int) -> Callable[[tuple], int]:
    """Per-leaf delay for the realistic pipeline-parallel staleness profile.

    Splits the grug Transformer into ``num_stages`` contiguous pipeline stages
    over its ``num_layers`` blocks (plus the input embedding group and the output
    projection group). A 1F1B/async pipeline applies a gradient ``(P-1-stage)``
    weight-versions late, so the last stage is fresh (τ=0) and τ increases toward
    the first stage. ``num_stages == num_layers`` gives one stage per layer
    (τ=0 for the last block, τ=1 for the next-to-last, ...).
    """
    if num_stages < 1 or num_layers < 1:
        raise ValueError(f"num_layers and num_stages must be >=1, got {num_layers}, {num_stages}")

    def leaf_tau(path: tuple) -> int:
        names = [getattr(k, "name", None) for k in path]
        if "blocks" in names:
            block_idx = getattr(path[names.index("blocks") + 1], "idx", 0)
            stage = (block_idx * num_stages) // num_layers
            return (num_stages - 1) - stage
        if any(n in _INPUT_FIELDS for n in names):
            return num_stages - 1
        if any(n in _OUTPUT_FIELDS for n in names):
            return 0
        return 0

    return leaf_tau


class DelayState(NamedTuple):
    """State for the delayed-gradient wrapper.

    ``grad_buf`` / ``param_buf`` are length-``tau`` tuples (oldest first) holding
    past gradients and the parameters they were computed at. ``v_ema`` is the
    optional EMA-of-g^2 curvature buffer (``None`` unless ``corrector`` needs it).
    ``m_raw`` is the optional EMA of the (stale) gradient — the raw, pre-Newton-
    Schulz momentum the ``wp_*`` predictors extrapolate (``None`` otherwise).
    ``last_update`` is the most recent applied update ``dW`` (used by the
    forward predictors; zeros otherwise). ``inner`` is the wrapped optimizer's
    state.
    """

    grad_buf: tuple
    param_buf: tuple
    v_ema: optax.Updates | None
    m_raw: optax.Updates | None
    last_update: optax.Updates
    inner: optax.OptState


def wrap_delayed(
    inner: optax.GradientTransformation,
    *,
    tau: int,
    corrector: str = "none",
    dc_lambda: float = 1.0,
    dc_beta2: float = 0.99,
    lr_damp: float = 1.0,
    pred_beta: float = 0.95,
    leaf_tau: Callable[[tuple], int] | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Wrap ``inner`` so it receives gradients delayed per parameter.

    With ``leaf_tau=None`` every parameter is delayed by the same ``tau`` steps
    (a uniform global delay; faithful to PipeDream-2BW). With ``leaf_tau`` set,
    each parameter leaf is delayed by ``leaf_tau(path)`` steps — the realistic
    pipeline-parallel profile where the last stage is fresh (τ=0) and τ grows
    toward the first stage. The FIFO depth is the maximum τ over the tree; each
    leaf reads the gradient from its own τ steps ago.

    Args:
        inner: the optimizer to feed stale (optionally corrected) gradients to.
        tau: uniform gradient delay in steps when ``leaf_tau`` is None. ``0`` is a
            pass-through.
        corrector: one of :data:`CORRECTORS`.
        dc_lambda: DC-ASGD correction strength.
        dc_beta2: EMA decay for the ``dc_asgd_ema`` curvature buffer.
        lr_damp: step multiplier for the ``lr_damp`` corrector (PipeMare-style
            staleness damping); ``<1`` shrinks the applied update.
        pred_beta: EMA decay for the raw-momentum buffer ``m_raw`` that the
            ``wp_*`` forward predictors extrapolate. Approximates Muon's internal
            momentum without reaching into the inner optimizer's opaque state.
        leaf_tau: optional per-leaf delay map (path -> τ) for the per-stage PP
            profile; overrides ``tau`` when provided.
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if corrector not in CORRECTORS:
        raise ValueError(f"unknown corrector {corrector!r}; expected one of {CORRECTORS}")
    needs_v = corrector == "dc_asgd_ema"
    needs_w = corrector in ("dc_asgd", "dc_asgd_ema")
    needs_mraw = corrector in _PREORTH_CORRECTORS
    damp = lr_damp if corrector == "lr_damp" else 1.0

    def _tau(path) -> int:
        return tau if leaf_tau is None else leaf_tau(path)

    def _tau_max(params) -> int:
        if leaf_tau is None:
            return tau
        taus = [leaf_tau(path) for path, _ in jax.tree_util.tree_leaves_with_path(params)]
        return max(taus) if taus else 0

    def _scale(updates):
        if damp == 1.0:
            return updates
        return jax.tree.map(lambda u: damp * u, updates)

    def _select(history):
        # history[d] is the tree from d steps ago (history[0] = current); pick,
        # per leaf, the entry from that leaf's own delay.
        return jax.tree_util.tree_map_with_path(lambda path, *hist: hist[_tau(path)], *history)

    def init_fn(params):
        tau_max = _tau_max(params)
        grad_buf = tuple(otu.tree_zeros_like(params) for _ in range(tau_max))
        # Seed weight snapshots with the initial params so DC-ASGD's (w_t -
        # w_stale) term is ~0 during the FIFO fill rather than differencing
        # against zeros. Only the DC correctors need the weight history.
        param_buf = tuple(params for _ in range(tau_max)) if needs_w else ()
        v_ema = otu.tree_zeros_like(params) if needs_v else None
        m_raw = otu.tree_zeros_like(params) if needs_mraw else None
        last_update = otu.tree_zeros_like(params)
        return DelayState(grad_buf, param_buf, v_ema, m_raw, last_update, inner.init(params))

    def _correct(g_stale, v_ema, params, w_stale):
        if corrector not in ("dc_asgd", "dc_asgd_ema"):
            # none / lr_damp / forward-predictor modes leave the gradient as-is.
            return g_stale, v_ema
        delta_w = jax.tree.map(lambda a, b: a - b, params, w_stale)
        if corrector == "dc_asgd":
            corrected = jax.tree.map(lambda g, dw: g + dc_lambda * (g * g) * dw, g_stale, delta_w)
            return corrected, v_ema
        # dc_asgd_ema: curvature from an EMA of g^2 (reused second moment).
        new_v = jax.tree.map(lambda v, g: dc_beta2 * v + (1.0 - dc_beta2) * (g * g), v_ema, g_stale)
        corrected = jax.tree.map(lambda g, v, dw: g + dc_lambda * v * dw, g_stale, new_v, delta_w)
        return corrected, new_v

    def update_fn(grads, state, params=None):
        tau_max = _tau_max(params)
        g_stale = _select((grads, *reversed(state.grad_buf)))
        new_grad_buf = (*state.grad_buf[1:], grads) if tau_max else ()
        if needs_w:
            w_stale = _select((params, *reversed(state.param_buf)))
            new_param_buf = (*state.param_buf[1:], params) if tau_max else ()
        else:
            w_stale, new_param_buf = params, ()

        corrected, new_v = _correct(g_stale, state.v_ema, params, w_stale)
        # Raw-momentum buffer (pre-orthogonalization): EMA the *stale* gradient
        # the optimizer actually applies, so the predictor extrapolates the same
        # signal that drives the step.
        if needs_mraw:
            new_m_raw = jax.tree.map(lambda m, g: pred_beta * m + (1.0 - pred_beta) * g, state.m_raw, g_stale)
        else:
            new_m_raw = None
        updates, new_inner = inner.update(corrected, state.inner, params=params)
        updates = _scale(updates)
        return updates, DelayState(new_grad_buf, new_param_buf, new_v, new_m_raw, updates, new_inner)

    return optax.with_extra_args_support(optax.GradientTransformation(init_fn, update_fn))


def _rms(x: jax.Array) -> jax.Array:
    """Root-mean-square of a leaf (scalar), with an epsilon floor."""
    return jnp.sqrt(jnp.mean(jnp.square(x)) + _EPS)


def _preorth_leaf_offset(
    corrector: str,
    horizon: float,
    last_update: jax.Array,
    m_raw: jax.Array,
    param: jax.Array,
    trust: float,
) -> jax.Array:
    """Predicted-weight offset for one leaf under a ``wp_*`` (pre-orth) corrector.

    The base prediction points ``horizon`` steps along the *descent* direction of
    the smoothed raw momentum ``-m_raw``, rescaled to the realized update RMS so
    its magnitude matches Muon's spectrally-normalized step (which ``last_update``
    carries) rather than the raw-gradient scale. The ``wp_cautious`` /
    ``wp_confidence`` / ``wp_trust`` variants then gate or clamp that offset.
    """
    descent = -m_raw / _rms(m_raw)  # unit-RMS descent direction (pre-orthogonalization)
    base = horizon * descent * _rms(last_update)
    if corrector == "wp_preorth":
        return base
    if corrector == "wp_cautious":
        # Cautious gate: keep coords where the predicted descent agrees with the
        # realized update, rescale surviving mass to preserve the step size.
        keep = (descent * last_update > 0).astype(base.dtype)
        return base * keep / (jnp.mean(keep) + _EPS)
    if corrector == "wp_confidence":
        # Soft gate: shrink by the (clipped) cosine between the predicted descent
        # and the realized update direction.
        cos = jnp.vdot(descent, last_update) / (_rms(descent) * _rms(last_update) * descent.size)
        return base * jnp.maximum(0.0, cos)
    if corrector == "wp_trust":
        # LARS-style clamp: cap the offset RMS at a trust fraction of the param
        # scale (scale-free, since Muon's update magnitude is normalized).
        cap = trust * _rms(param)
        return base * jnp.minimum(1.0, cap / _rms(base))
    raise ValueError(f"not a pre-orth corrector: {corrector!r}")


@dataclass(frozen=True)
class _DelayMixin:
    """Config knobs shared by the delayed optimizer configs."""

    tau: int = 0
    corrector: str = "none"
    dc_lambda: float = 1.0
    dc_beta2: float = 0.99
    # weight_pred: how many steps ahead to extrapolate the last update, as a
    # multiple of tau. 1.0 predicts exactly tau steps forward (the delay depth);
    # <1 under-predicts, >1 over-predicts (for ablating prediction horizon).
    pred_scale: float = 1.0
    # lr_damp: step multiplier for the lr_damp corrector (1.0 = no damping).
    lr_damp: float = 1.0
    # wp_*: EMA decay for the raw-momentum predictor buffer (approximates Muon's
    # internal momentum) and the LARS-style trust fraction for wp_trust's clamp.
    pred_beta: float = 0.95
    trust: float = 0.01
    # Per-stage PP profile: split the model into num_stages stages over
    # num_layers blocks, delaying each leaf by its stage's τ. num_stages == 0
    # keeps the uniform global `tau`. num_layers is the model's block count.
    num_stages: int = 0
    num_layers: int = 0

    def _leaf_tau(self) -> Callable[[tuple], int] | None:
        if self.num_stages <= 0:
            return None
        return grug_stage_tau(self.num_layers, self.num_stages)

    def _wrap(self, inner: optax.GradientTransformation) -> optax.GradientTransformationExtraArgs:
        return wrap_delayed(
            inner,
            tau=self.tau,
            corrector=self.corrector,
            dc_lambda=self.dc_lambda,
            dc_beta2=self.dc_beta2,
            lr_damp=self.lr_damp,
            pred_beta=self.pred_beta,
            leaf_tau=self._leaf_tau(),
        )

    def make_forward_predictor(self) -> Callable[[optax.OptState, optax.Params], optax.Updates] | None:
        """Forward-weight predictor for the forward-side correctors; else ``None``.

        Returns a function mapping ``(opt_state, params)`` to a per-leaf parameter
        offset, so the train step evaluates each parameter's gradient at
        ``w + offset`` (predicted weights) while still applying the update at the
        real weights. The horizon is ``τ_leaf * pred_scale`` — with a per-stage
        profile τ_leaf is the leaf's own stage delay, otherwise the uniform
        ``tau``.

        ``weight_pred`` extrapolates the post-orthogonalization update
        ``last_update`` directly. The ``wp_*`` family instead points the offset
        along the smoothed raw momentum and applies its gate/clamp; see
        :func:`_preorth_leaf_offset`.
        """
        if self.corrector not in _FORWARD_PRED_CORRECTORS:
            return None
        leaf_tau = self._leaf_tau()
        ps = self.pred_scale
        corrector = self.corrector
        trust = self.trust
        uniform_tau = float(self.tau)

        def horizon(path) -> float:
            return (leaf_tau(path) if leaf_tau is not None else uniform_tau) * ps

        if corrector == "weight_pred":

            def predict(opt_state: optax.OptState, params: optax.Params) -> optax.Updates:
                return jax.tree_util.tree_map_with_path(lambda path, u: horizon(path) * u, opt_state.last_update)

            return predict

        def predict(opt_state: optax.OptState, params: optax.Params) -> optax.Updates:
            return jax.tree_util.tree_map_with_path(
                lambda path, u, m, p: _preorth_leaf_offset(corrector, horizon(path), u, m, p, trust),
                opt_state.last_update,
                opt_state.m_raw,
                params,
            )

        return predict


@OptimizerConfig.register_subclass("grug_muon_delayed")
@dataclass(frozen=True)
class DelayedGrugMuonConfig(_DelayMixin, GrugMuonConfig):
    """`grug_muon` with a delayed-gradient wrapper for staleness experiments."""

    def build(self, num_train_steps):
        return self._wrap(super().build(num_train_steps))


@OptimizerConfig.register_subclass("grug_moe_adamh_delayed")
@dataclass(frozen=True)
class DelayedGrugMoeAdamHConfig(_DelayMixin, GrugMoeAdamHConfig):
    """`grug_moe_adamh_v2` with a delayed-gradient wrapper for staleness experiments."""

    def build(self, num_train_steps):
        return self._wrap(super().build(num_train_steps))


__all__ = [
    "CORRECTORS",
    "DelayState",
    "DelayedGrugMoeAdamHConfig",
    "DelayedGrugMuonConfig",
    "wrap_delayed",
]
