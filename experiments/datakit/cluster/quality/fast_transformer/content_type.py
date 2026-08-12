# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predict a document's content type, so quality can be calibrated per type.

The oracle grades each document *as an example of its own type*, but the types do
not share a ceiling: solved agent trajectories reach the top score 3.4% of the time
against prose's 26%. Ranking within a type is sound, so the fix is a per-type
calibration offset (:mod:`calibrate`) rather than a different quality model — and
that needs a type at scoring time, where the pipeline has only the document text.

Deliberately not a second transformer. Content type is close to a surface property,
so this is a hashed bag of tokens plus a short vector of structural features, scored
by multinomial logistic regression: a hash, a sparse lookup and one matrix multiply,
which is negligible against the quality model's own budget.

The two feature families were measured separately and fail differently, which is why
both are here. On 22k labels, held out:

===================================  ========  =========================
features                             accuracy  what it gets right
===================================  ========  =========================
structural only                         0.781  agentic 1.00, math 0.53
hashed tokens only                      0.844  multilingual 0.93, math 0.81
both                                    0.850  agentic 1.00, prose 0.89
===================================  ========  =========================

Structural features see tool-call markers and bracket density; tokens see
vocabulary. Neither alone clears the bar the calibration is gated on.

``other`` is the weak class (0.43 recall). It is the residual category — 2.7% of
labels, junk-heavy, mean quality 2.10 — so its calibration sits near the global one
and confusing it with prose costs little. The costly confusion is math against
prose, whose ceilings genuinely differ, and that is what the token features fix.

Accuracy is not the gate on its own: what matters is whether per-type calibration
still lands when the type is *predicted* rather than known. Report parity with
predicted types, never with true ones.
"""

import argparse
import logging
import re

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts
from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES

logger = logging.getLogger(__name__)

TOKENIZER = "intfloat/multilingual-e5-small"
MAX_TOKENS = 512
# Hashed token buckets. Collisions are harmless at this width for a 7-way decision,
# and a fixed width keeps the artifact the same size whatever the label set.
HASH_BUCKETS = 1 << 15
TRAIN_STEPS = 900
LEARNING_RATE = 2.0
L2 = 2e-5
EVAL_FRAC = 0.2

# Structural signals the token bag cannot see: how bracketed, how numeric, how
# non-Latin, and whether the document announces itself as a transcript.
_CODE_MARKER = re.compile(r"\b(def |function |import |class |#include)")
_MATH_MARKER = re.compile(r"(\\frac|\\begin|\\sum|\bTheorem\b|\bLemma\b|\bproof\b)")
_AGENT_MARKER = re.compile(r"(tool_call|Observation:|Action:|terminal|bash)", re.I)


def structural_features(text: str) -> list[float]:
    """Surface statistics of one document, in a fixed order.

    The character-class ratios are computed with numpy over a code-point view
    rather than a Python generator over ``s``. A generator walks every character
    at interpreter speed while holding the GIL, which costs ~4 ms on a 12k-char
    document — invisible in a single-threaded run and crippling once many shards
    share one worker process, where every thread queues behind it. The vectorized
    form releases the GIL inside numpy and is roughly two orders of magnitude
    faster.
    """
    s = text or ""
    n = max(len(s), 1)
    lines = s.split("\n")
    words = s.split()
    codes = np.frombuffer(s.encode("utf-32-le", "ignore"), dtype=np.uint32) if s else np.empty(0, np.uint32)
    n_codes = max(codes.size, 1)
    return [
        float((codes > 127).sum()) / n_codes,
        float(((codes >= 0x3000) & (codes <= 0x9FFF)).sum()) / n_codes,
        s.count("{") / n * 100,
        s.count("(") / n * 100,
        s.count(";") / n * 100,
        s.count("=") / n * 100,
        s.count("<") / n * 100,
        s.count("$") / n * 100,
        s.count("\\") / n * 100,
        s.count("_") / n * 100,
        float(((codes >= 0x30) & (codes <= 0x39)).sum()) / n_codes,
        s.count("\n") / n * 100,
        float(np.mean([len(x) for x in lines])) if lines else 0.0,
        float("<user>" in s or "<system>" in s or "<|" in s),
        float("```" in s),
        float(bool(_CODE_MARKER.search(s))),
        float(bool(_MATH_MARKER.search(s))),
        float(bool(_AGENT_MARKER.search(s))),
        float(np.log1p(len(s))),
        float(np.mean([len(w) for w in words[:400]])) if words else 0.0,
    ]


def _design_matrix(texts: list[str], mean: np.ndarray | None, std: np.ndarray | None):
    """Sparse [tokens | standardized structural features | bias] for ``texts``."""
    ids = encode_texts(TOKENIZER, texts, MAX_TOKENS)
    rows: list[int] = []
    cols: list[int] = []
    for i, seq in enumerate(ids):
        for token in set(seq):
            rows.append(i)
            cols.append(token % HASH_BUCKETS)
    bag = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), HASH_BUCKETS))
    feats = np.array([structural_features(t) for t in texts], np.float32)
    if mean is None or std is None:
        mean, std = feats.mean(0), feats.std(0) + 1e-9
    feats = (feats - mean) / std
    ones = np.ones((len(texts), 1), np.float32)
    design = sp.hstack([bag, sp.csr_matrix(feats), sp.csr_matrix(ones)]).tocsr()
    return design, mean, std


def fit(texts: list[str], types: list[str], *, seed: int = 0) -> tuple[dict, float]:
    """Fit the classifier and report its held-out accuracy."""
    labels = [t for t in CONTENT_TYPES if t in set(types)]
    index = {t: i for i, t in enumerate(labels)}
    y = np.array([index[t] for t in types])
    design, mean, std = _design_matrix(texts, None, None)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    cut = int((1 - EVAL_FRAC) * len(y))
    train, test = order[:cut], order[cut:]

    weights = np.zeros((design.shape[1], len(labels)), np.float32)
    onehot = np.zeros((len(train), len(labels)), np.float32)
    onehot[np.arange(len(train)), y[train]] = 1
    x_train = design[train]
    for _ in range(TRAIN_STEPS):
        logits = x_train @ weights
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        weights -= LEARNING_RATE * (x_train.T @ (probs - onehot) / len(train) + L2 * weights)

    predicted = np.asarray(design[test] @ weights).argmax(1)
    accuracy = float((predicted == y[test]).mean())
    for i, label in enumerate(labels):
        mask = y[test] == i
        if mask.any():
            logger.info("  %-14s n=%-5d recall=%.2f", label, int(mask.sum()), (predicted[mask] == i).mean())
    return {
        "labels": labels,
        "weights": weights,
        "feature_mean": mean,
        "feature_std": std,
    }, accuracy


def predict(model: dict, texts: list[str]) -> list[str]:
    """The most likely content type for each document."""
    design, _, _ = _design_matrix(texts, model["feature_mean"], model["feature_std"])
    chosen = np.asarray(design @ model["weights"]).argmax(1)
    return [model["labels"][i] for i in chosen]


def save(model: dict, path: str) -> None:
    with StoragePath(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            weights=model["weights"],
            feature_mean=model["feature_mean"],
            feature_std=model["feature_std"],
            labels=np.array(model["labels"]),
        )


def load(path: str) -> dict:
    with StoragePath(path).open("rb") as handle:
        data = np.load(handle, allow_pickle=False)
        return {
            "weights": data["weights"],
            "feature_mean": data["feature_mean"],
            "feature_std": data["feature_std"],
            "labels": [str(x) for x in data["labels"]],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="label parquet with text + content_type")
    parser.add_argument("--out", required=True, help="where to write the classifier npz")
    parser.add_argument("--min-accuracy", type=float, default=0.85, help="fail below this held-out accuracy")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    with StoragePath(args.labels).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["text", "content_type"])
    texts = table.column("text").to_pylist()
    types = table.column("content_type").to_pylist()
    logger.info("content_type: fitting on %d labeled documents", len(texts))
    model, accuracy = fit(texts, types)
    logger.info("content_type: held-out accuracy %.3f", accuracy)
    if accuracy < args.min_accuracy:
        raise SystemExit(
            f"content_type: held-out accuracy {accuracy:.3f} is below {args.min_accuracy:.2f} — "
            "per-type calibration applied through this would be guesswork on the types it confuses"
        )
    save(model, args.out)
    logger.info("content_type: wrote %s", args.out)


if __name__ == "__main__":
    main()
