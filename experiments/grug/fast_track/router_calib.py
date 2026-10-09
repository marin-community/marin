# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc router calibration of a trained MoE checkpoint: does eval loss improve with other routing biases or
per-expert output gains? A diagnostic (``--router-calib``), never a change to the run.

Everything runs dropless (no capacity drops, whatever the load), with the run's weights frozen. Biases are edited as
the QB betas (``[L, E]``, layer order; ``_apply_qb_betas`` maps them onto the stacks). Fits use held-out training
batches (``fit``); every candidate is also scored on validation sequences (``val``), which never steer a choice.

1. ``scale``: the QB biases times ``s`` for each ``SCALES`` entry.
2. ``layer_scale``: per-layer bias multipliers by coordinate search on ``fit``, starting from the best ``s``.
3. ``gains``: per-expert output gains ``exp(theta)`` (``Router.expert_output_gain``, the combine weight of expert e
   times ``g_e``) fit by Adam on fresh training batches, at the base and at the best biases.
4. ``soft_bias``: the biases themselves, fit by Adam through a smoothed top-K (``router_soft_select``) over
   ``SOFT_CANDIDATES`` candidates with an annealed temperature, then scored with the hard top-K.
"""

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Sequence

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.grug.fast_track.model import Transformer, _stack_layer_indices, _tail_stack_layer_indices
from experiments.grug.fast_track.router_soft_select import soft_select

logger = logging.getLogger(__name__)

SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LAYER_MULTS = (0.5, 0.75, 1.25, 1.5)
LAYER_PASSES = 2
FIT_STEPS = 200
GAIN_LR = 0.02
BIAS_LR = 0.02
SOFT_CANDIDATES = 16
TAU_START, TAU_END = 1.0, 0.05

# loss(model, betas, batch) -> scalar mean next-token loss
LossFn = Callable[[Transformer, jax.Array, object], jax.Array]


def with_expert_gains(model: Transformer, gains: jax.Array) -> Transformer:
    """``model`` with ``Router.expert_output_gain`` set from ``gains`` ``[L, E]`` (layer order) in every MoE stack."""
    cfg = model.config
    softmax, kda = _stack_layer_indices(cfg)
    softmax_tail, kda_tail = _tail_stack_layer_indices(cfg)
    for name, layers in (
        ("stacked_blocks", softmax),
        ("kda_blocks", kda),
        ("stacked_blocks_tail", softmax_tail),
        ("kda_blocks_tail", kda_tail),
    ):
        if not layers:
            continue
        rows = gains[jnp.asarray(layers)]
        model = eqx.tree_at(
            lambda m, name=name: getattr(m, name).stacked.mlp.expert_output_gain,
            model,
            rows,
            is_leaf=lambda x: x is None,
        )
    return model


def with_candidates(model: Transformer, candidates: int) -> Transformer:
    """The same weights with every MoE layer selecting ``candidates`` experts (a config-only change)."""
    cfg16 = dataclasses.replace(model.config, num_experts_per_token=candidates)
    shapes = eqx.filter_eval_shape(Transformer.init, cfg16, key=jax.random.PRNGKey(0))
    return jax.tree.unflatten(jax.tree.structure(shapes), jax.tree.leaves(model))


def run(
    model: Transformer,
    base_betas: jax.Array,
    loss_fn: LossFn,
    fit: Sequence,
    val: Sequence,
    stream: Iterator,
    path: str,
) -> dict:
    """All four calibrations; writes and returns the JSON summary."""
    num_layers = base_betas.shape[0]
    jit_loss = eqx.filter_jit(loss_fn)

    def mean_loss(m, betas, batches) -> float:
        return float(np.mean([float(jit_loss(m, betas, b)) for b in batches]))

    def score(m, betas) -> dict:
        return {"fit": mean_loss(m, betas, fit), "val": mean_loss(m, betas, val)}

    out: dict = {"base": score(model, base_betas)}
    logger.info("router_calib base %s", out["base"])

    out["scale"] = {}
    for s in SCALES:
        out["scale"][f"{s:g}"] = score(model, base_betas * s)
        logger.info("router_calib scale %g %s", s, out["scale"][f"{s:g}"])
    best_s = min(SCALES, key=lambda s: out["scale"][f"{s:g}"]["fit"])

    mults = np.full((num_layers,), best_s, np.float64)
    best_fit = out["scale"][f"{best_s:g}"]["fit"]
    for _ in range(LAYER_PASSES):
        for layer in range(num_layers):
            for m_ in LAYER_MULTS:
                trial = mults.copy()
                trial[layer] *= m_
                fit_loss = mean_loss(model, base_betas * jnp.asarray(trial, jnp.float32)[:, None], fit)
                if fit_loss < best_fit:
                    best_fit, mults = fit_loss, trial
    layer_betas = base_betas * jnp.asarray(mults, jnp.float32)[:, None]
    out["layer_scale"] = {"multipliers": mults.tolist(), **score(model, layer_betas)}
    logger.info("router_calib layer_scale %s", out["layer_scale"])

    def fit_gains(betas) -> tuple[jax.Array, list]:
        theta = jnp.zeros(base_betas.shape, jnp.float32)
        opt = optax.adam(GAIN_LR)
        state = opt.init(theta)

        @eqx.filter_jit
        def step(theta, state, batch):
            loss, grad = jax.value_and_grad(lambda t: loss_fn(with_expert_gains(model, jnp.exp(t)), betas, batch))(theta)
            updates, state = opt.update(grad, state)
            return optax.apply_updates(theta, updates), state, loss

        trace = []
        for i in range(FIT_STEPS):
            theta, state, loss = step(theta, state, next(stream))
            if i % 20 == 0:
                trace.append(float(loss))
        return theta, trace

    out["gains"] = {}
    for label, betas in (("base_bias", base_betas), ("layer_scale_bias", layer_betas)):
        theta, trace = fit_gains(betas)
        gained = with_expert_gains(model, jnp.exp(theta))
        out["gains"][label] = {
            **score(gained, betas),
            "train_trace": trace,
            "log_gain_std": float(jnp.std(theta)),
            "log_gain_absmax": float(jnp.max(jnp.abs(theta))),
        }
        logger.info(
            "router_calib gains %s %s", label, {k: v for k, v in out["gains"][label].items() if k != "train_trace"}
        )

    wide = with_candidates(model, SOFT_CANDIDATES)
    k = model.config.num_experts_per_token
    delta = jnp.zeros(base_betas.shape, jnp.float32)
    opt = optax.adam(BIAS_LR)
    state = opt.init(delta)

    def soft_step(delta, state, batch, tau):
        def objective(d):
            with soft_select(k, tau):
                return loss_fn(wide, base_betas + d, batch)

        loss, grad = jax.value_and_grad(objective)(delta)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(delta, updates), state, loss

    soft_step = eqx.filter_jit(soft_step)
    trace = []
    for i in range(FIT_STEPS):
        tau = TAU_START * (TAU_END / TAU_START) ** (i / max(1, FIT_STEPS - 1))
        delta, state, loss = soft_step(delta, state, next(stream), jnp.float32(tau))
        if i % 20 == 0:
            trace.append(float(loss))
    out["soft_bias"] = {
        **score(model, base_betas + delta),
        "train_trace": trace,
        "delta_std": float(jnp.std(delta)),
        "base_beta_std": float(jnp.std(base_betas)),
    }
    logger.info("router_calib soft_bias %s", {k_: v for k_, v in out["soft_bias"].items() if k_ != "train_trace"})
    if jax.process_index() == 0:
        with fsspec.open(path, "w") as f:
            json.dump(out, f, indent=1)
    return out
