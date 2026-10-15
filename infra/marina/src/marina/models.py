# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-wide model instances shared by retained application revisions."""

import threading

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

_lock = threading.Lock()
_embeddings: dict[tuple[str, int], TextEmbedding] = {}
_rerankers: dict[tuple[str, str, str, int], TextCrossEncoder] = {}


def shared_embedding(model: str, threads: int) -> TextEmbedding:
    """Load one embedding model per specification in this worker."""
    key = (model, threads)
    with _lock:
        if key not in _embeddings:
            _embeddings[key] = TextEmbedding(model, threads=threads)
        return _embeddings[key]


def shared_reranker(model: str, source: str, model_file: str, threads: int) -> TextCrossEncoder:
    """Load one custom ONNX reranker per specification in this worker."""
    key = (model, source, model_file, threads)
    with _lock:
        if key not in _rerankers:
            TextCrossEncoder.add_custom_model(
                model=model,
                sources=ModelSource(hf=source),
                model_file=model_file,
                description="CPU ONNX reranker.",
                license="apache-2.0",
                size_in_gb=0.03,
            )
            _rerankers[key] = TextCrossEncoder(model, threads=threads)
        return _rerankers[key]
