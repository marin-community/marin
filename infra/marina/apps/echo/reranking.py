# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select Echo's quantized cross-encoder reranker."""

from fastembed.rerank.cross_encoder import TextCrossEncoder
from marina.models import shared_reranker

from . import search_config


def text_cross_encoder() -> TextCrossEncoder:
    return shared_reranker(
        search_config.RERANK_MODEL,
        search_config.RERANK_MODEL_SOURCE,
        search_config.RERANK_MODEL_FILE,
        search_config.INFERENCE_THREADS,
    )
