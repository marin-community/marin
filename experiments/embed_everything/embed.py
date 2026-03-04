# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch embedding computation using sentence-transformers.

Computes embeddings for documents in a JSONL file and saves them as a .npz archive.
Model-agnostic: defaults to Luxical but supports any HuggingFace embedding model.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """Artifact returned by embed_documents."""

    path: str
    """Path to the output .npz file containing embeddings, ids, and labels."""
    n_docs: int
    """Number of documents embedded."""
    embedding_dim: int
    """Dimensionality of the embeddings."""


def embed_documents(
    output_path: str,
    input_path: str,
    model_name: str = "DatologyAI/luxical-one",
    batch_size: int = 64,
    max_length: int = 512,
) -> EmbedResult:
    """Embed documents from a JSONL file using a sentence-transformer model.

    Args:
        output_path: Directory to write the output .npz file.
        input_path: Path to a JSONL file with at minimum "id", "text", and "label" fields.
        model_name: HuggingFace model name for sentence-transformers.
        batch_size: Encoding batch size.
        max_length: Maximum token length for truncation.

    Returns:
        EmbedResult with path to the .npz file and metadata.
    """
    from sentence_transformers import SentenceTransformer

    os.makedirs(output_path, exist_ok=True)

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            ids.append(doc["id"])
            texts.append(doc.get("text", ""))
            labels.append(doc.get("label", ""))

    logger.info(f"Loaded {len(texts)} documents from {input_path}")
    logger.info(f"Loading model {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_length

    logger.info(f"Encoding {len(texts)} documents with batch_size={batch_size}")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    out_file = os.path.join(output_path, "embeddings.npz")
    np.savez(out_file, embeddings=embeddings, ids=np.array(ids), labels=np.array(labels))

    logger.info(f"Saved embeddings ({embeddings.shape}) to {out_file}")
    return EmbedResult(path=out_file, n_docs=len(ids), embedding_dim=embeddings.shape[1])
