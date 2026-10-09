# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``soft_select``: a smoothed top-K selection, so the router biases get a gradient (``router_calib.py``).

Inside the context the model must run with ``num_experts_per_token`` candidates above the hard ``k``. Each
candidate's combine weight is multiplied by ``sigmoid((b - t) / tau)``, with ``b`` its biased logit (the bias
without the stop-gradient) and ``t`` the midpoint of the ``k``-th and ``k+1``-th biased logits, then the weights are
renormalized to their original sum. As ``tau -> 0`` this is exactly the hard top-``k`` routing."""

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp

SOFT_SELECT: tuple[int, float] | None = None
"""``(k, tau)`` while ``soft_select`` is active; read by the router at trace time."""


@contextmanager
def soft_select(k: int, tau) -> Iterator[None]:
    global SOFT_SELECT
    previous, SOFT_SELECT = SOFT_SELECT, (k, tau)
    try:
        yield
    finally:
        SOFT_SELECT = previous


def soft_combine(combine: jax.Array, biased: jax.Array) -> jax.Array:
    """``combine`` ``[T, K']`` reweighted by the smoothed selection of the ``biased`` ``[T, K']`` logits."""
    assert SOFT_SELECT is not None
    k, tau = SOFT_SELECT
    top = jax.lax.top_k(jax.lax.stop_gradient(biased), k + 1)[0]
    threshold = 0.5 * (top[:, k - 1] + top[:, k])
    keep = jax.nn.sigmoid((biased - threshold[:, None]) / tau)
    weighted = combine * keep
    return weighted * (jnp.sum(combine, axis=-1, keepdims=True) / (jnp.sum(weighted, axis=-1, keepdims=True) + 1e-9))
