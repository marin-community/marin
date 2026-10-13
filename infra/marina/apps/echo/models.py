# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echo's two ONNX models, loaded on first use.

One kernel process serves every Marina app, so loading a model while the app is being
mounted would hold up every other app and every deploy. The first search that needs a
model pays for it instead; the lock keeps two concurrent first searches from loading the
same model twice.
"""

import logging
import threading

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from . import reranking, search_config

logger = logging.getLogger(__name__)


class SearchModels:
    """The query embedder and the cross-encoder reranker, each loaded once per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedding: TextEmbedding | None = None
        self._reranker: TextCrossEncoder | None = None

    def embedding(self) -> TextEmbedding:
        with self._lock:
            if self._embedding is None:
                logger.info("loading embedding model %s", search_config.EMBED_MODEL)
                self._embedding = TextEmbedding(search_config.EMBED_MODEL, threads=search_config.INFERENCE_THREADS)
            return self._embedding

    def reranker(self) -> TextCrossEncoder:
        with self._lock:
            if self._reranker is None:
                logger.info("loading reranker model %s", search_config.RERANK_MODEL)
                self._reranker = reranking.text_cross_encoder()
            return self._reranker
