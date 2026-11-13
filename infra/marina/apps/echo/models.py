# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echo's model construction and process-wide reuse across package revisions."""

from functools import partial

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from marina.process_cache import process_cache

from . import search_config


def load_reranker(model: str, source: str, model_file: str, threads: int) -> TextCrossEncoder:
    TextCrossEncoder.add_custom_model(
        model=model,
        sources=ModelSource(hf=source),
        model_file=model_file,
        description="CPU ONNX reranker.",
        license="apache-2.0",
        size_in_gb=0.03,
    )
    return TextCrossEncoder(model, threads=threads)


class SearchModels:
    def embedding(self) -> TextEmbedding:
        model, threads = search_config.EMBED_MODEL, search_config.INFERENCE_THREADS
        return process_cache.get_or_create(
            "echo.embedding", (model, threads), partial(TextEmbedding, model, threads=threads)
        )

    def reranker(self) -> TextCrossEncoder:
        settings = (
            search_config.RERANK_MODEL,
            search_config.RERANK_MODEL_SOURCE,
            search_config.RERANK_MODEL_FILE,
            search_config.INFERENCE_THREADS,
        )
        return process_cache.get_or_create("echo.reranker", settings, partial(load_reranker, *settings))
