# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dolma-3 fasttext baselines for quality and topic classification.

Runs the Allen AI Dolma-3 fasttext quality and WebOrganizer topic classifiers
on sampled documents and writes per-doc predictions to parquet. Downstream
evaluators correlate these predictions against oracle labels so we can ask
Helw150's "can we beat fasttext?" question directly.

Models are downloaded via ``huggingface_hub.hf_hub_download`` and cached on
worker disk. Each model binary is ~4 GB — expect a one-time cold-start cost
per fresh Iris worker.
"""

import contextlib
import logging
import os
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

if TYPE_CHECKING:
    import fasttext

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _fasttext_numpy2_shim() -> Iterator[None]:
    """Make ``np.array(..., copy=False)`` fall back to ``copy=None`` for this call.

    fasttext 0.9.3's ``predict`` does ``np.array(probs, copy=False)``, which
    numpy 2.0+ raises on. ``copy=None`` restores the pre-2.0 "avoid copy if
    possible" behaviour. We're single-threaded in the sampling loop, so the
    global patch is safe for the scope; the try/finally restores the symbol.
    """
    original = np.array

    def patched(*args: object, **kwargs: object):
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return original(*args, **kwargs)

    np.array = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        np.array = original  # type: ignore[assignment]


DOLMA3_QUALITY_MODEL = "allenai/dolma3-fasttext-quality-classifier"
DOLMA3_TOPIC_MODEL = "allenai/dolma3-fasttext-weborganizer-topic-classifier"
MODEL_FILENAME = "model.bin"

# fasttext splits on whitespace; newlines break its tokenizer and embedded tabs
# in text throw off some preprocessing steps. This mirrors the standard
# fasttext classify convention.
_NEWLINE_RE = re.compile(r"[\n\r\t]+")

# Cap per-doc characters passed to fasttext. Keeps predict() latency bounded
# and matches the oracle's truncation so we're scoring the same substring.
MAX_CHARS_FOR_FASTTEXT = 8000


def _load_fasttext_model(repo_id: str) -> "fasttext.FastText._FastText":
    """Download and load a fasttext .bin from a HuggingFace repo."""
    import fasttext
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s (%s) via hf_hub_download", repo_id, MODEL_FILENAME)
    local_path = hf_hub_download(repo_id=repo_id, filename=MODEL_FILENAME)
    logger.info("Loading fasttext model from %s", local_path)
    model = fasttext.load_model(local_path)
    logger.info("fasttext labels (%d): %s", len(model.get_labels()), model.get_labels())
    return model


def _preprocess(text: str, max_chars: int = MAX_CHARS_FOR_FASTTEXT) -> str:
    """Flatten whitespace and truncate; mirrors standard fasttext prep."""
    cleaned = _NEWLINE_RE.sub(" ", text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def _strip_fasttext_prefix(label: str) -> str:
    """Strip the leading ``__label__`` that fasttext prepends to every class."""
    return label.removeprefix("__label__")


def _pick_positive_quality_label(labels: list[str]) -> str:
    """Return the label whose probability we interpret as 'high-quality score'.

    The Dolma 3 quality classifier is binary. Labels are not documented on the
    model card; we detect the positive class by string match with a prioritized
    allow-list, falling back to the first label with a loud warning.
    """
    stripped = {label: _strip_fasttext_prefix(label).lower() for label in labels}
    for token in ("hq", "high", "positive", "good", "1"):
        for raw, bare in stripped.items():
            if bare == token or bare.endswith(f"_{token}") or bare.startswith(f"{token}_"):
                return raw
    logger.warning(
        "Could not auto-detect positive label in %s; defaulting to first label %r. "
        "Inspect logs and adjust if this is the low-quality class.",
        labels,
        labels[0],
    )
    return labels[0]


def score_documents_fasttext_quality(
    output_path: str,
    input_path: str,
    input_filename: str = "quality_samples.parquet",
    output_filename: str = "quality_fasttext_scores.parquet",
    model_repo: str = DOLMA3_QUALITY_MODEL,
) -> None:
    """Score each doc with the Dolma-3 fasttext quality classifier.

    Writes one parquet row per input doc with the positive-class probability
    (``fasttext_quality_score``) and the argmax label (``fasttext_quality_label``).
    """
    model = _load_fasttext_model(model_repo)
    labels = model.get_labels()
    positive_label = _pick_positive_quality_label(labels)
    logger.info("Using %r as the positive (high-quality) class", positive_label)

    input_file = os.path.join(input_path, input_filename)
    docs = list(load_parquet(input_file))
    logger.info("Scoring %d documents with %s", len(docs), model_repo)

    rows: list[dict] = []
    with _fasttext_numpy2_shim():
        for i, doc in enumerate(docs):
            text = _preprocess(doc["text"])
            pred_labels, pred_probs = model.predict(text, k=-1)  # full distribution
            prob_by_label = dict(zip(pred_labels, pred_probs, strict=True))

            rows.append(
                {
                    "doc_id": doc["doc_id"],
                    "split": doc.get("split", "unknown"),
                    "quality_bucket": doc.get("quality_bucket", "unknown"),
                    "fasttext_quality_label": _strip_fasttext_prefix(pred_labels[0]),
                    "fasttext_quality_score": float(prob_by_label.get(positive_label, 0.0)),
                }
            )

            if (i + 1) % 100 == 0:
                logger.info("Quality scoring progress: %d/%d", i + 1, len(docs))

    write_parquet_file(rows, os.path.join(output_path, output_filename))
    logger.info("Wrote %d quality predictions to %s", len(rows), output_path)


def classify_documents_fasttext_topic(
    output_path: str,
    input_path: str,
    input_filename: str = "topic_samples.parquet",
    output_filename: str = "topic_fasttext_predictions.parquet",
    model_repo: str = DOLMA3_TOPIC_MODEL,
) -> None:
    """Classify each doc with the Dolma-3 WebOrganizer topic classifier.

    Writes the raw argmax label stripped of the ``__label__`` prefix plus its
    confidence. Mapping to the human-readable 24-class WebOrganizer taxonomy
    happens at eval time where we normalize both sides.
    """
    model = _load_fasttext_model(model_repo)

    input_file = os.path.join(input_path, input_filename)
    docs = list(load_parquet(input_file))
    logger.info("Classifying %d documents with %s", len(docs), model_repo)

    rows: list[dict] = []
    with _fasttext_numpy2_shim():
        for i, doc in enumerate(docs):
            text = _preprocess(doc["text"])
            pred_labels, pred_probs = model.predict(text, k=1)

            rows.append(
                {
                    "doc_id": doc["doc_id"],
                    "split": doc.get("split", "unknown"),
                    "source_label": doc.get("source_label", "unknown"),
                    "fasttext_topic": _strip_fasttext_prefix(pred_labels[0]),
                    "fasttext_topic_confidence": float(pred_probs[0]),
                }
            )

            if (i + 1) % 100 == 0:
                logger.info("Topic classification progress: %d/%d", i + 1, len(docs))

    write_parquet_file(rows, os.path.join(output_path, output_filename))
    logger.info("Wrote %d topic predictions to %s", len(rows), output_path)
