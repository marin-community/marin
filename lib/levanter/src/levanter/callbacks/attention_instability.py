# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Attention instability metrics for training diagnostics.

Provides two complementary signals:

1. **QK weight-norm product** (all backends): upper bound on max attention logit
   derived from Q/K projection weight norms.  O(params), runs inside JIT.
2. **Max pre-softmax attention logit** (vanilla backend only): exact value captured
   via ``jax.debug.callback`` during the forward pass.  Fused backends (SPLASH,
   NVTE, JAX_FLASH) never materialise the logit tensor so this metric is
   unavailable — the weight-norm proxy covers that case.
"""

import math
from dataclasses import dataclass
from typing import TypeVar

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import PyTree

import haliax as hax
import levanter.tracker
from levanter.callbacks._core import JitCallback, StepInfo
from levanter.layers.attention import Attention, get_max_attn_logit, reset_max_attn_logit
from levanter.trainer_state import InsideJitInfo, TrainerState

S = TypeVar("S", bound=TrainerState)
M = TypeVar("M", bound=PyTree)


def _frobenius_norm(x: hax.NamedArray) -> jax.Array:
    """Frobenius norm of a NamedArray, reduced to a scalar."""
    return jnp.sqrt(jnp.sum(x.array**2))


def _frobenius_norm_per_layer(x: hax.NamedArray, layer_axis: str) -> jax.Array:
    """Frobenius norm along all axes except the layer axis, returning per-layer norms."""
    raw = x.array
    ax = x.axis_index(layer_axis)
    # Move layer axis to front, flatten the rest, take norm
    raw = jnp.moveaxis(raw, ax, 0)
    shape = raw.shape
    flat = raw.reshape(shape[0], -1)
    return jnp.sqrt(jnp.sum(flat**2, axis=1))


def _find_attention_modules(model: PyTree) -> list[Attention]:
    """Walk an Equinox model tree and collect all ``Attention`` instances."""
    found: list[Attention] = []

    def _walk(node):
        if isinstance(node, Attention):
            found.append(node)
            return
        # Recurse into eqx.Module fields
        if isinstance(node, eqx.Module):
            for val in vars(node).values():
                _walk(val)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)

    _walk(model)
    return found


def compute_attention_instability_stats(
    model: PyTree,
) -> dict[str, jax.Array]:
    """Compute attention instability metrics from model weights.

    For each ``Attention`` module found in the model tree, computes the product
    of Q and K projection weight Frobenius norms scaled by ``1/sqrt(head_size)``.
    This upper-bounds the magnitude of any single pre-softmax attention logit and
    serves as a backend-agnostic instability proxy.

    Returns a dict of metric names to scalar JAX arrays.
    """
    attn_modules = _find_attention_modules(model)
    if not attn_modules:
        return {}

    metrics: dict[str, jax.Array] = {}
    global_max = jnp.array(-jnp.inf)

    for i, attn in enumerate(attn_modules):
        scale = 1.0 / math.sqrt(attn.config.head_size)
        q_weight = attn.q_proj.weight
        k_weight = attn.k_proj.weight

        # Check if weights have a layer/block axis (Stacked modules)
        layer_axis = None
        for ax in q_weight.axes:
            if ax.name in ("layer", "block", "layers", "blocks"):
                layer_axis = ax.name
                break

        if layer_axis is not None:
            q_norms = _frobenius_norm_per_layer(q_weight, layer_axis)
            k_norms = _frobenius_norm_per_layer(k_weight, layer_axis)
            products = q_norms * k_norms * scale
            module_max = jnp.max(products)
        else:
            q_norm = _frobenius_norm(q_weight)
            k_norm = _frobenius_norm(k_weight)
            module_max = q_norm * k_norm * scale

        metrics[f"attention/qk_norm_product/{i}"] = module_max
        global_max = jnp.maximum(global_max, module_max)

    metrics["attention/qk_norm_product_max"] = global_max
    return metrics


@dataclass(frozen=True)
class AttentionInstabilityConfig:
    """Configuration for attention instability monitoring.

    Attributes:
        interval: Compute metrics every N training steps.  0 disables.
        track_vanilla_max_logit: When True and the model uses the vanilla
            attention backend, capture the exact max pre-softmax logit per step
            via ``jax.debug.callback``.  Has no effect on fused backends
            (SPLASH, NVTE, JAX_FLASH) which never materialise the logit tensor.
    """

    interval: int = 0
    track_vanilla_max_logit: bool = False

    @property
    def is_enabled(self) -> bool:
        return self.interval > 0

    def build(self) -> "AttentionInstabilityCallback":
        return AttentionInstabilityCallback(
            track_vanilla_max_logit=self.track_vanilla_max_logit,
        )


class AttentionInstabilityCallback(JitCallback[S, M, dict[str, jax.Array]]):
    """JitCallback that logs attention instability metrics.

    Inside JIT: computes QK weight-norm products from ``state.model``.
    Outside JIT: logs them and, when enabled, also logs the exact max
    pre-softmax logit captured from the vanilla attention path.
    """

    def __init__(self, track_vanilla_max_logit: bool = False):
        self.track_vanilla_max_logit = track_vanilla_max_logit

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]) -> dict[str, jax.Array]:
        return compute_attention_instability_stats(state.model)

    def on_step(self, step_info: StepInfo[S], cb_info: dict[str, jax.Array]):
        to_log: dict[str, float | jax.Array] = dict(cb_info)

        if self.track_vanilla_max_logit:
            max_logit = get_max_attn_logit()
            if max_logit > float("-inf"):
                to_log["attention/max_attn_logit"] = max_logit
            reset_max_attn_logit()

        levanter.tracker.log(to_log, step=step_info.step)
