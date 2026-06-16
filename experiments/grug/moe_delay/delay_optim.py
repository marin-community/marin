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
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import GrugMuonConfig
from optax import tree_utils as otu

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig

CORRECTORS = ("none", "dc_asgd", "dc_asgd_ema")


class DelayState(NamedTuple):
    """State for the delayed-gradient wrapper.

    ``grad_buf`` / ``param_buf`` are length-``tau`` tuples (oldest first) holding
    past gradients and the parameters they were computed at. ``v_ema`` is the
    optional EMA-of-g^2 curvature buffer (``None`` unless ``corrector`` needs it).
    ``inner`` is the wrapped optimizer's state.
    """

    grad_buf: tuple
    param_buf: tuple
    v_ema: optax.Updates | None
    inner: optax.OptState


def wrap_delayed(
    inner: optax.GradientTransformation,
    *,
    tau: int,
    corrector: str = "none",
    dc_lambda: float = 1.0,
    dc_beta2: float = 0.99,
) -> optax.GradientTransformation:
    """Wrap ``inner`` so it receives gradients delayed by ``tau`` steps.

    Args:
        inner: the optimizer to feed stale (optionally corrected) gradients to.
        tau: gradient delay in steps. ``0`` is a pass-through.
        corrector: one of :data:`CORRECTORS`.
        dc_lambda: DC-ASGD correction strength.
        dc_beta2: EMA decay for the ``dc_asgd_ema`` curvature buffer.
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if corrector not in CORRECTORS:
        raise ValueError(f"unknown corrector {corrector!r}; expected one of {CORRECTORS}")
    needs_v = corrector == "dc_asgd_ema"

    def init_fn(params):
        grad_buf = tuple(otu.tree_zeros_like(params) for _ in range(tau))
        # Seed the weight snapshots with the initial params so the DC-ASGD
        # (w_t - w_stale) term is ~0 during the tau-step FIFO fill rather than
        # differencing against zeros.
        param_buf = tuple(params for _ in range(tau))
        v_ema = otu.tree_zeros_like(params) if needs_v else None
        return DelayState(grad_buf, param_buf, v_ema, inner.init(params))

    def _correct(g_stale, v_ema, params, w_stale):
        if corrector == "none":
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
        if tau == 0:
            corrected, new_v = _correct(grads, state.v_ema, params, params)
            updates, new_inner = inner.update(corrected, state.inner, params=params)
            return updates, DelayState((), (), new_v, new_inner)

        g_stale = state.grad_buf[0]
        w_stale = state.param_buf[0]
        new_grad_buf = (*state.grad_buf[1:], grads)
        new_param_buf = (*state.param_buf[1:], params)

        corrected, new_v = _correct(g_stale, state.v_ema, params, w_stale)
        updates, new_inner = inner.update(corrected, state.inner, params=params)
        return updates, DelayState(new_grad_buf, new_param_buf, new_v, new_inner)

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass(frozen=True)
class _DelayMixin:
    """Config knobs shared by the delayed optimizer configs."""

    tau: int = 0
    corrector: str = "none"
    dc_lambda: float = 1.0
    dc_beta2: float = 0.99

    def _wrap(self, inner: optax.GradientTransformation) -> optax.GradientTransformation:
        return wrap_delayed(
            inner,
            tau=self.tau,
            corrector=self.corrector,
            dc_lambda=self.dc_lambda,
            dc_beta2=self.dc_beta2,
        )


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
