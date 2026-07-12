# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a fasttext quality classifier from LLM-scored documents.

Reads the parquet emitted by :mod:`score`, thresholds the LLM's
continuous quality score into a binary label (high vs. low), trains
``fasttext.train_supervised`` on the training split, evaluates on a held-
out validation split, and writes ``model.bin`` plus a ``metadata.json``
summary into ``--output-dir``.

The binary form mirrors :mod:`allenai/dolma3-fasttext-quality-classifier`
so the trained model plugs directly into
:func:`experiments.datakit.cluster.quality.v0.all_sources_quality_llm.classify_llm_quality_step` with
``score_target_label="1"`` -- inference yields ``P(label == "1") ∈ [0, 1]``
as the continuous quality score for any source.

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=4 --memory=16G \\
        --extra=cpu --priority production --region europe-west4 \\
        --job-name "llm-quality-train-$(date +%Y%m%d-%H%M%S)" -- \\
        python -m experiments.datakit.cluster.quality.v0.train \\
          --input gs://marin-eu-west4/datakit/llm-quality-classifier/scored/train-n7000-seed42-sonnet46.parquet \\
          --output-dir gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/
"""

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass

import fasttext
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


DEFAULT_VAL_FRAC = 0.1
DEFAULT_SEED = 42

# Default cutoff on the normalized [0, 1] score: 0.5 corresponds to raw=3
# ("Average") on the 1-5 rubric, so "high" means "Average or better".
# This matches fineweb-edu's choice and gives a roughly balanced split on
# Sonnet 4.6's distribution (~41% positive on the smoke). Override via
# ``--threshold`` (e.g. for median split, pass the value explicitly).
DEFAULT_THRESHOLD = 0.5

# fasttext.train_supervised defaults aligned with the dolma3-quality /
# fineweb-edu binary classifiers -- dim=100, 5 epochs, lr=0.5 with
# bigrams. Small enough to train on CPU in minutes; gives a competitive
# accuracy floor without tuning.
DEFAULT_HP: dict[str, int | float | str] = {
    "dim": 100,
    "epoch": 5,
    "lr": 0.5,
    "wordNgrams": 2,
    "minCount": 3,
    "bucket": 200_000,
    "loss": "softmax",
}


# Same NumPy 2 / fasttext-wheel 0.9.2 shim as in
# experiments.datakit.fasttext: train_supervised internally calls
# ``np.array(..., copy=False)`` which NumPy 2 rejects. Idempotent.
def _patch_numpy_copy_compat() -> None:
    if getattr(np, "_fasttext_copy_compat", False):
        return
    _orig = np.array

    def _shim(*args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return _orig(*args, **kwargs)

    np.array = _shim
    np._fasttext_copy_compat = True


_NEWLINE_RE = re.compile(r"[\n\r\t]+")


@dataclass
class TrainedModel:
    threshold: float
    n_train: int
    n_val: int
    val_precision: float
    val_recall: float
    val_f1: float
    train_label_balance: dict[str, int]
    per_source_mean_score: dict[str, float]
    hyperparameters: dict[str, int | float | str]


@dataclass
class ScoredRecord:
    source: str
    id: str
    text: str
    score: float


def _read_scored(input_path: str) -> list[ScoredRecord]:
    with StoragePath(input_path).open("rb") as fh:
        table = pq.read_table(fh)
    out: list[ScoredRecord] = []
    cols = {name: table.column(name).to_pylist() for name in ("source", "id", "text", "score_raw", "score_normalized")}
    for s, i, t, raw, norm in zip(
        cols["source"], cols["id"], cols["text"], cols["score_raw"], cols["score_normalized"], strict=True
    ):
        if raw is None or int(raw) < 0:
            continue
        if norm is None or (isinstance(norm, float) and math.isnan(norm)):
            continue
        if not t:
            continue
        out.append(ScoredRecord(source=str(s), id=str(i), text=str(t), score=float(norm)))
    return out


def _format_for_fasttext(text: str, label: int, max_text_chars: int) -> str:
    cleaned = _NEWLINE_RE.sub(" ", text)
    if len(cleaned) > max_text_chars:
        cleaned = cleaned[:max_text_chars]
    return f"__label__{label} {cleaned}\n"


def _evaluate(model, val_lines: list[tuple[int, str]]) -> tuple[float, float, float]:
    """Return (precision, recall, F1) of the positive class on the val split."""
    tp = fp = fn = tn = 0
    for true_label, text in val_lines:
        labels, _ = model.predict(text, k=1)
        pred = int(labels[0].removeprefix("__label__")) if labels else 0
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == 0:
            fp += 1
        elif pred == 0 and true_label == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def train(
    *,
    input_path: str,
    output_dir: str,
    threshold: float | None,
    val_frac: float,
    seed: int,
    max_text_chars: int,
    hp_overrides: dict[str, int | float | str],
) -> TrainedModel:
    _patch_numpy_copy_compat()
    rows = _read_scored(input_path)
    if not rows:
        raise RuntimeError(f"no usable scored rows in {input_path}")
    logger.info("loaded %d scored rows from %s", len(rows), input_path)

    cutoff: float = threshold if threshold is not None else DEFAULT_THRESHOLD
    if threshold is None:
        logger.info("threshold not given; using default = %.4f (raw=3 'Average or better')", cutoff)
    else:
        logger.info("using fixed threshold = %.4f", cutoff)
    # Statistics module used as a safety check + diagnostic.
    scores = [r.score for r in rows]
    logger.info(
        "score stats: n=%d median=%.4f mean=%.4f >=threshold=%d",
        len(scores),
        statistics.median(scores),
        statistics.mean(scores),
        sum(1 for s in scores if s >= cutoff),
    )

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    logger.info("split: %d train / %d val", len(train_rows), len(val_rows))

    hp = {**DEFAULT_HP, **hp_overrides}

    per_source_scores: dict[str, list[float]] = {}
    for r in rows:
        per_source_scores.setdefault(r.source, []).append(r.score)
    per_source_mean = {s: sum(v) / len(v) for s, v in per_source_scores.items()}

    train_labels: list[int] = []
    train_lines: list[str] = []
    for r in train_rows:
        label = 1 if r.score >= cutoff else 0
        train_labels.append(label)
        train_lines.append(_format_for_fasttext(r.text, label, max_text_chars))
    train_balance = Counter(f"__label__{label}" for label in train_labels)

    val_eval: list[tuple[int, str]] = []
    for r in val_rows:
        label = 1 if r.score >= cutoff else 0
        cleaned = _NEWLINE_RE.sub(" ", r.text)
        val_eval.append((label, cleaned[:max_text_chars]))

    logger.info("train label balance: %s", dict(train_balance))

    with tempfile.TemporaryDirectory() as tmp:
        train_path = os.path.join(tmp, "train.txt")
        model_path = os.path.join(tmp, "model.bin")
        with open(train_path, "w", encoding="utf-8") as fh:
            for line in train_lines:
                fh.write(line)

        logger.info("training fasttext with hp=%s", hp)
        model = fasttext.train_supervised(input=train_path, **hp)
        model.save_model(model_path)
        logger.info(
            "trained model: dim=%d, vocab=%d, labels=%s", model.get_dimension(), len(model.words), model.get_labels()
        )

        precision, recall, f1 = _evaluate(model, val_eval)
        logger.info("val precision=%.4f recall=%.4f f1=%.4f (n=%d)", precision, recall, f1, len(val_eval))

        meta = TrainedModel(
            threshold=cutoff,
            n_train=len(train_rows),
            n_val=len(val_rows),
            val_precision=precision,
            val_recall=recall,
            val_f1=f1,
            train_label_balance=dict(train_balance),
            per_source_mean_score=per_source_mean,
            hyperparameters={**hp, "seed": seed, "val_frac": val_frac, "max_text_chars": max_text_chars},
        )
        meta_path = os.path.join(tmp, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta.__dict__, fh, indent=2, sort_keys=True)

        StoragePath(output_dir).mkdirs()
        for fname in ("model.bin", "metadata.json"):
            src = os.path.join(tmp, fname)
            dst = output_dir.rstrip("/") + "/" + fname
            with open(src, "rb") as srcf, open_url(dst, "wb") as dstf:
                while True:
                    chunk = srcf.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dstf.write(chunk)
            logger.info("uploaded %s -> %s", fname, dst)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Scored parquet (from score.py)")
    parser.add_argument("--output-dir", required=True, help="Directory for model.bin + metadata.json")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Score cutoff in [0,1]; default={DEFAULT_THRESHOLD} (raw=3 'Average or better')",
    )
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-text-chars", type=int, default=4_000, help="Per-example char cap fed to fasttext")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--word-ngrams", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    hp_overrides: dict[str, int | float | str] = {}
    if args.epoch is not None:
        hp_overrides["epoch"] = args.epoch
    if args.lr is not None:
        hp_overrides["lr"] = args.lr
    if args.word_ngrams is not None:
        hp_overrides["wordNgrams"] = args.word_ngrams
    if args.dim is not None:
        hp_overrides["dim"] = args.dim

    train(
        input_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        val_frac=args.val_frac,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
        hp_overrides=hp_overrides,
    )


if __name__ == "__main__":
    main()
