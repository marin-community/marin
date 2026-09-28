# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Type a document from its Harrier embedding.

An MLP over the 1024-d Harrier embedding separates the 7 rubric content types it
was fitted on well enough to route the per-type quality calibration: 5-fold
held-out 86.0% on the joined 88k label set against a 79.8% source-majority
baseline, and 87.9% agreement with oracle types on the seed-0 holdout. Known
caveat: unweighted cross-entropy depresses recall on the residual ``other`` class
to ~0.56, which is why the content-type step stores the whole distribution from
:func:`predict_probabilities`: a class-prior correction can then be applied
without re-reading the embeddings.

The input is the L2-normalized embedding direction. The training run scaled the
int8 rows by the Harrier quantization constant before normalizing; a uniform scale
cancels under unit norm, so :func:`score_fusion.normalize_embeddings` feeds this
the identical matrix.

The training recipe (1024 -> 512 -> 256 -> 7 GELU MLP, ~658K params, AdamW lr 1e-3
cosine to zero, weight decay 1e-4, batch 4096, 40 epochs, unweighted CE) lives with
the label-join exploration that fitted it. What is kept here is the forward and
the loader.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from rigging.filesystem.storage_path import StoragePath

# Rows per forward. The model is 658K parameters, so activations at this width
# are ~100 MB and the batch is about where the matmuls stop being launch-bound.
PREDICT_BATCH = 8192


class DomainMlp(eqx.Module):
    """1024 -> 512 -> 256 -> num_classes GELU MLP over the embedding direction."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array

    def __call__(self, x: Array) -> Array:
        h = jax.nn.gelu(x @ self.w1 + self.b1)
        h = jax.nn.gelu(h @ self.w2 + self.b2)
        return h @ self.w3 + self.b3


def load(path: str) -> tuple[DomainMlp, tuple[str, ...]]:
    """The fitted weights and their class labels, in the order the head emits."""
    with StoragePath(path).open("rb") as handle:
        data = np.load(handle, allow_pickle=False)
        model = DomainMlp(**{name: jnp.asarray(data[name]) for name in ("w1", "b1", "w2", "b2", "w3", "b3")})
        return model, tuple(str(x) for x in data["labels"])


def predict_probabilities(model: DomainMlp, x: np.ndarray) -> np.ndarray:
    """Full ``[n, num_classes]`` softmax per row, batched to bound activation memory."""
    out = [
        np.asarray(jax.nn.softmax(model(jnp.asarray(x[start : start + PREDICT_BATCH])), axis=1), dtype=np.float32)
        for start in range(0, len(x), PREDICT_BATCH)
    ]
    return np.concatenate(out) if out else np.zeros((0, model.w3.shape[1]), dtype=np.float32)
