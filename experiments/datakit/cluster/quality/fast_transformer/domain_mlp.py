# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Type a document from its stored harrier embedding.

The deployed :mod:`content_type` classifier reads token and structural features
of the text (0.85 held-out accuracy on the 22k set). Every document in the 50M
sample already carries a 1024-d harrier embedding, and a small MLP on that
embedding separates the same 7 rubric types better on the joined 88k label set:
5-fold held-out 86.0% in the exploration run this module retrains, against a
79.8% source-majority baseline. Known caveat carried over from that run:
unweighted cross-entropy depresses recall on the residual ``other`` class
(~0.56), which the model gate already treats as residual.

The input is the L2-normalized embedding direction. The exploration run scaled
the int8 rows by the harrier quantization constant before normalizing; the
scale cancels under unit-norm, so :func:`joined_labels.embedding_matrix` is
reused unchanged.

Training follows the exploration run exactly: 1024 -> 512 -> 256 -> 7 GELU MLP
(~658K params), AdamW lr 1e-3 cosine to zero, weight decay 1e-4, batch 4096,
40 epochs, unweighted CE, no early stopping. ``main`` reports 5-fold held-out
accuracy as the sanity gate, then refits on every row and saves the weights.
"""

import argparse
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES

logger = logging.getLogger(__name__)

HIDDEN_DIMS = (512, 256)
BATCH_SIZE = 4096
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOLDS = 5
PREDICT_BATCH = 8192


class DomainMlp(eqx.Module):
    """1024 -> 512 -> 256 -> num_classes GELU MLP over the embedding direction."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array

    @classmethod
    def initialize(cls, in_dim: int, num_classes: int, *, key: PRNGKeyArray) -> "DomainMlp":
        h1, h2 = HIDDEN_DIMS
        k1, k2, k3 = jr.split(key, 3)

        def glorot(k, shape):
            return jr.normal(k, shape) * (2.0 / sum(shape)) ** 0.5

        return cls(
            w1=glorot(k1, (in_dim, h1)),
            b1=jnp.zeros(h1),
            w2=glorot(k2, (h1, h2)),
            b2=jnp.zeros(h2),
            w3=glorot(k3, (h2, num_classes)),
            b3=jnp.zeros(num_classes),
        )

    def __call__(self, x: Array) -> Array:
        h = jax.nn.gelu(x @ self.w1 + self.b1)
        h = jax.nn.gelu(h @ self.w2 + self.b2)
        return h @ self.w3 + self.b3


def fit(x: np.ndarray, y: np.ndarray, num_classes: int, *, seed: int = 0) -> DomainMlp:
    """Train on all given rows with the exploration run's fixed recipe."""
    steps_per_epoch = max(1, len(x) // BATCH_SIZE)
    schedule = optax.cosine_decay_schedule(LEARNING_RATE, steps_per_epoch * EPOCHS)
    optimizer = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    model = DomainMlp.initialize(x.shape[1], num_classes, key=jr.PRNGKey(seed))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        def loss_fn(m):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(xb), yb))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        perm = rng.permutation(len(x))
        for s in range(steps_per_epoch):
            batch = perm[s * BATCH_SIZE : (s + 1) * BATCH_SIZE]
            model, opt_state, loss = step(model, opt_state, jnp.asarray(x[batch]), jnp.asarray(y[batch]))
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            logger.info("domain_mlp epoch %d: train_loss=%.4f", epoch, float(loss))
    return model


def predict_indices(model: DomainMlp, x: np.ndarray) -> np.ndarray:
    """Argmax class index per row, batched to bound activation memory."""
    out = []
    for start in range(0, len(x), PREDICT_BATCH):
        logits = model(jnp.asarray(x[start : start + PREDICT_BATCH]))
        out.append(np.asarray(logits.argmax(axis=1)))
    return np.concatenate(out)


def predict(model: DomainMlp, labels: list[str], embeddings_int8: list) -> np.ndarray:
    """Most likely content type per document, from the raw int8 embedding rows."""
    idx = predict_indices(model, embedding_matrix(embeddings_int8))
    return np.array([labels[i] for i in idx])


def cross_validate(x: np.ndarray, y: np.ndarray, labels: list[str], *, seed: int = 0) -> float:
    """5-fold held-out accuracy of the fixed recipe, with per-class recall."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    predicted = np.full(len(x), -1)
    for fold in range(FOLDS):
        held = order[fold::FOLDS]
        train = np.setdiff1d(order, held)
        model = fit(x[train], y[train], len(labels), seed=seed)
        predicted[held] = predict_indices(model, x[held])
        logger.info("fold %d: held-out accuracy %.3f", fold, float((predicted[held] == y[held]).mean()))
    accuracy = float((predicted == y).mean())
    for i, label in enumerate(labels):
        mask = y == i
        logger.info("  %-14s n=%-5d recall=%.3f", label, int(mask.sum()), float((predicted[mask] == i).mean()))
    return accuracy


def save(model: DomainMlp, labels: list[str], path: str) -> None:
    with StoragePath(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            w1=np.asarray(model.w1),
            b1=np.asarray(model.b1),
            w2=np.asarray(model.w2),
            b2=np.asarray(model.b2),
            w3=np.asarray(model.w3),
            b3=np.asarray(model.b3),
            labels=np.array(labels),
        )


def load(path: str) -> tuple[DomainMlp, list[str]]:
    with StoragePath(path).open("rb") as handle:
        data = np.load(handle, allow_pickle=False)
        model = DomainMlp(
            w1=jnp.asarray(data["w1"]),
            b1=jnp.asarray(data["b1"]),
            w2=jnp.asarray(data["w2"]),
            b2=jnp.asarray(data["b2"]),
            w3=jnp.asarray(data["w3"]),
            b3=jnp.asarray(data["b3"]),
        )
        return model, [str(x) for x in data["labels"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined-dir", default=DEFAULT_JOINED, help="labels-x-embeddings join output root")
    p.add_argument("--out", required=True, help="where to write the classifier npz")
    p.add_argument("--min-accuracy", type=float, default=0.85, help="fail below this 5-fold held-out accuracy")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    joined = load_joined(args.joined_dir, columns=["id", "embedding", "glm52_content_type"])
    types = joined["glm52_content_type"]
    unknown = sorted(set(types) - set(CONTENT_TYPES))
    if unknown:
        raise ValueError(f"joined rows carry content types outside the rubric: {unknown}")
    labels = [t for t in CONTENT_TYPES if t in set(types)]
    index = {t: i for i, t in enumerate(labels)}
    y = np.array([index[t] for t in types])
    x = embedding_matrix(joined["embedding"])
    majority = float(max(np.bincount(y)) / len(y))
    logger.info("domain_mlp: %d rows, %d classes, majority-class baseline %.3f", len(y), len(labels), majority)

    accuracy = cross_validate(x, y, labels)
    logger.info("domain_mlp: 5-fold held-out accuracy %.3f (exploration-run reference 0.860)", accuracy)
    if accuracy < args.min_accuracy:
        raise SystemExit(
            f"domain_mlp: held-out accuracy {accuracy:.3f} is below {args.min_accuracy:.2f} — "
            "typing the gate metrics with this would blur model failures into classifier failures"
        )

    model = fit(x, y, len(labels))
    train_accuracy = float((predict_indices(model, x) == y).mean())
    logger.info("domain_mlp: final fit on all %d rows, train accuracy %.3f", len(y), train_accuracy)
    save(model, labels, args.out)
    logger.info("domain_mlp: wrote %s", args.out)


if __name__ == "__main__":
    main()
