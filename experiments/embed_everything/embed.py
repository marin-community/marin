# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute document embeddings using HuggingFace sentence-transformers.

Supports Luxical-One and other sentence-transformer compatible models.
Saves embeddings as .npz files alongside document metadata for downstream
evaluation (linear probes, clustering).
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from iris.marin_fs import open_url

from marin.execution import THIS_OUTPUT_PATH

logger = logging.getLogger(__name__)

LUXICAL_MODEL = "DatologyAI/luxical-one"
DEFAULT_BATCH_SIZE = 64


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return list of dicts."""
    docs = []
    with open_url(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _load_model(model_name: str):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    logger.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def _embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode a list of texts into embeddings."""
    logger.info("Encoding %d texts with batch_size=%d", len(texts), batch_size)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)


def _save_embeddings(
    embeddings: np.ndarray,
    doc_ids: list[str],
    labels: list[str],
    splits: list[str],
    output_path: str,
    filename: str,
) -> None:
    """Save embeddings and metadata to a local .npz then upload."""
    import tempfile

    local_path = os.path.join(tempfile.gettempdir(), filename)
    np.savez_compressed(
        local_path,
        embeddings=embeddings,
        doc_ids=np.array(doc_ids),
        labels=np.array(labels),
        splits=np.array(splits),
    )

    dest = os.path.join(output_path, filename)
    with open(local_path, "rb") as src:
        content = src.read()
    with open_url(dest, "wb") as dst:
        dst.write(content)
    os.remove(local_path)
    logger.info("Saved embeddings to %s (shape=%s)", dest, embeddings.shape)


@dataclass(frozen=True)
class EmbedConfig:
    """Config for computing document embeddings."""

    input_path: str
    output_path: str = THIS_OUTPUT_PATH
    model_name: str = LUXICAL_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    input_filename: str = "quality_samples.jsonl"
    output_filename: str = "quality_embeddings.npz"
    label_field: str = "quality_bucket"


def embed_documents(config: EmbedConfig) -> None:
    """Compute embeddings for sampled documents and save as .npz."""
    input_file = os.path.join(config.input_path, config.input_filename)
    docs = _read_jsonl(input_file)
    logger.info("Embedding %d documents with model=%s", len(docs), config.model_name)

    model = _load_model(config.model_name)

    texts = [doc["text"] for doc in docs]
    doc_ids = [doc["doc_id"] for doc in docs]
    labels = [doc.get(config.label_field, "unknown") for doc in docs]
    splits = [doc.get("split", "unknown") for doc in docs]

    embeddings = _embed_texts(model, texts, config.batch_size)

    os.makedirs(config.output_path, exist_ok=True)
    _save_embeddings(embeddings, doc_ids, labels, splits, config.output_path, config.output_filename)
