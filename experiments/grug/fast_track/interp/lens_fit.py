# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``attrib``'s head / sublayer means and tuned lens, fit in-job (for studies that run before ``attrib.npz`` exists)."""

import logging

import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import attrib, common

logger = logging.getLogger(__name__)


def fit(model, mesh, num_sequences: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(head_mean, sub_mean, tuned)`` as ``attrib.main`` fits them, on ``num_sequences`` of its fit sequences."""
    _, means_fn, tuned_fn, _ = attrib.make_fns()
    head_sum = sub_sum = xtx = xty = None
    n_sum = 0.0
    with set_mesh(mesh):
        for b in common.train_batches(attrib.BATCH, num_sequences, seed=1):
            tokens, seg, w = common.device_batch(b, mesh)
            h, s, n = means_fn(model, tokens, seg, w)
            a, c = tuned_fn(model, tokens, seg, w)
            head_sum = h if head_sum is None else head_sum + h
            sub_sum = s if sub_sum is None else sub_sum + s
            xtx = a if xtx is None else xtx + a
            xty = c if xty is None else xty + c
            n_sum += float(n)
    xtx_h, xty_h = np.asarray(xtx, np.float64), np.asarray(xty, np.float64)
    tuned = np.stack(
        [
            np.linalg.solve(xtx_h[k] + attrib.TUNED_RIDGE * n_sum * np.eye(xtx_h.shape[1]), xty_h[k])
            for k in range(len(attrib.STAGES))
        ]
    ).astype(np.float32)
    logger.info("means and tuned lens fit on %d tokens", n_sum)
    return head_sum / n_sum, sub_sum / n_sum, jnp.asarray(tuned)
