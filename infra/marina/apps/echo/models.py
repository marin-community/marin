# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echo selects its models; Marina shares their instances across app revisions."""

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from marina.models import shared_embedding

from . import reranking, search_config


class SearchModels:
    def embedding(self) -> TextEmbedding:
        return shared_embedding(search_config.EMBED_MODEL, search_config.INFERENCE_THREADS)

    def reranker(self) -> TextCrossEncoder:
        return reranking.text_cross_encoder()
