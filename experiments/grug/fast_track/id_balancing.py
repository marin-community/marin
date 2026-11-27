# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ID Balancing (arXiv 2609.39137): an integral-derivative controller for the MoE router bias.

QB sets each step's bias to the negated per-expert quantile of the previous step's margins, a proportional target
with no memory of past load errors. ID Balancing integrates the normalized load error instead. Per layer, with
``n_i`` the global routing counts and ``n̄`` their mean:

    e_i = (n̄ - n_i) / n̄,   Δe_i = e_i - e_i^prev,   g_i = 1[e_i^prev Δe_i > 0]
    b_i <- b_i + gain e_i + gain g_i Δe_i,   b <- b - mean(b)

The derivative gate is open only while an expert's imbalance grows without crossing zero, and is closed at the
first step (``e^prev = 0``). The bias reaches the router with the same one-step lag as QB's.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def id_balance_update(
    bias: Float[Array, "L E"],
    prev_error: Float[Array, "L E"],
    counts: Float[Array, "L E"],
    gain: float,
) -> tuple[Float[Array, "L E"], Float[Array, "L E"], Float[Array, "L E"]]:
    """One ID step from this step's per-layer routing ``counts``; returns ``(bias, error, gate)``."""
    counts = counts.astype(jnp.float32)
    mean = jnp.mean(counts, axis=-1, keepdims=True)
    error = (mean - counts) / jnp.maximum(mean, 1.0)
    delta = error - prev_error
    gate = (prev_error * delta > 0).astype(jnp.float32)
    bias = bias + gain * error + gain * gate * delta
    bias = bias - jnp.mean(bias, axis=-1, keepdims=True)
    return bias, error, gate


def load_violation_metrics(counts: Float[Array, "L E"], gate: Float[Array, "L E"]) -> dict[str, jax.Array]:
    """Per-layer MaxVio ``(max n - n̄) / n̄`` and MinVio ``(n̄ - min n) / n̄``, and the derivative gate's open share."""
    counts = counts.astype(jnp.float32)
    mean = jnp.maximum(jnp.mean(counts, axis=-1), 1.0)
    max_vio = (jnp.max(counts, axis=-1) - mean) / mean
    min_vio = (mean - jnp.min(counts, axis=-1)) / mean
    out = {
        "train/router/id_balance/max_vio_mean": jnp.mean(max_vio),
        "train/router/id_balance/max_vio_max": jnp.max(max_vio),
        "train/router/id_balance/min_vio_mean": jnp.mean(min_vio),
        "train/router/id_balance/gate_open_frac": jnp.mean(gate),
    }
    for layer in range(counts.shape[0]):
        out[f"train/router/layer_{layer}/max_vio"] = max_vio[layer]
        out[f"train/router/layer_{layer}/min_vio"] = min_vio[layer]
    return out
