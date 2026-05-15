# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute Luxical-One embeddings over a sampled corpus.

Loads ``samples.parquet`` (``id``, ``text``), runs sentence-transformers
``model.encode`` with ``normalize_embeddings=True``, and writes a single
``embeddings.npz`` with aligned ``embeddings`` and ``ids`` arrays.
"""

import logging
import os
import tempfile

import numpy as np
from rigging.filesystem import open_url
from zephyr.readers import load_parquet

logger = logging.getLogger(__name__)

LUXICAL_MODEL = "DatologyAI/luxical-one"
DEFAULT_BATCH_SIZE = 64


def embed_documents(
    output_path: str,
    samples_path: str,
    model_name: str = LUXICAL_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Embed every doc in ``samples.parquet`` and save aligned arrays to ``embeddings.npz``."""
    from sentence_transformers import SentenceTransformer

    docs = list(load_parquet(os.path.join(samples_path, "samples.parquet")))
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    logger.info("Loaded %d docs; loading model %s", len(docs), model_name)

    model = SentenceTransformer(model_name, trust_remote_code=True)
    logger.info("Model loaded, dim=%d; encoding...", model.get_sentence_embedding_dimension())

    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    logger.info("Encoded; embeddings shape=%s", embeddings.shape)

    local = os.path.join(tempfile.gettempdir(), "embeddings.npz")
    np.savez_compressed(local, embeddings=embeddings, ids=np.array(ids))
    with open(local, "rb") as src, open_url(os.path.join(output_path, "embeddings.npz"), "wb") as dst:
        dst.write(src.read())
    os.remove(local)
