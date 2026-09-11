# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inference-time forward pass for the pooled fast-transformer.

Split out from ``train.py`` so the scorer (and the production scoring / calibration
paths) can run a forward pass without importing the training loop (optax, the
optimizer, the fit machinery). Both training and inference share
``data_parallel_shardings``; only ``predict`` is inference-specific.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer

# Tokens per inference batch; bounds the [B, T, E] embedding activation so long
# context (T up to 16k) auto-shrinks the batch instead of OOMing one v6e chip.
_PREDICT_TOKEN_BUDGET = 262_144


def data_parallel_shardings():
    """(num_devices, replicated, batch-sharded) shardings over all chips.

    Data parallelism: the model + optimizer state are replicated on every chip and
    the batch is split across them, so all of a v6e slice's chips are used instead
    of one. With one device this is a no-op.
    """
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("dp",))
    return len(devices), NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("dp"))


@eqx.filter_jit
def predict_batch(model: FastTransformer, ids, doc_embed=None):
    """Sigmoid quality score for a fixed-shape batch (jitted, inference mode)."""
    return jax.nn.sigmoid(model(ids, doc_embed=doc_embed, key=None, inference=True))


def _pad_rows(chunk: np.ndarray, pad: int) -> np.ndarray:
    if not pad:
        return chunk
    return np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), dtype=chunk.dtype)], axis=0)


def predict(
    model: FastTransformer,
    ids: np.ndarray,
    batch_size: int | None = None,
    doc_embed: np.ndarray | None = None,
) -> np.ndarray:
    """Sigmoid quality score for every row in ``ids``.

    Chunks are padded to a constant ``batch_size`` so ``predict_batch`` compiles
    once per sequence length and is reused across all calls. When ``batch_size`` is
    not given it is sized from the sequence length to keep the activation footprint
    bounded. ``doc_embed`` rides along row-for-row when the model takes one.
    """
    if batch_size is None:
        batch_size = max(8, _PREDICT_TOKEN_BUDGET // ids.shape[1])
    # Inference is data-parallel too: callers pass the global batch (sized for the
    # whole slice), so shard each chunk across chips -- otherwise a single device
    # would try to hold the full global-batch attention tensor and OOM/segfault.
    ndev, _, batch_shard = data_parallel_shardings()
    batch_size = max(ndev, (batch_size // ndev) * ndev)
    out: list[np.ndarray] = []
    n = ids.shape[0]

    def put(a: np.ndarray):
        return jax.device_put(jnp.asarray(a), batch_shard)

    for start in range(0, n, batch_size):
        chunk = ids[start : start + batch_size]
        pad = batch_size - chunk.shape[0]
        emb = None if doc_embed is None else put(_pad_rows(doc_embed[start : start + batch_size], pad))
        preds = np.asarray(predict_batch(model, put(_pad_rows(chunk, pad)), emb))
        out.append(preds[: batch_size - pad] if pad else preds)
    return np.concatenate(out)
