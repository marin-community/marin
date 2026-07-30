# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct Echo's quantized cross-encoder reranker."""

import search_config
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder


def text_cross_encoder() -> TextCrossEncoder:
    """Load the CPU-optimized INT8 model from the upstream ONNX repository."""
    TextCrossEncoder.add_custom_model(
        model=search_config.RERANK_MODEL,
        sources=ModelSource(hf=search_config.RERANK_MODEL_SOURCE),
        model_file=search_config.RERANK_MODEL_FILE,
        description="INT8 MiniLM-L-6-v2 model optimized for reranking on CPU.",
        license="apache-2.0",
        size_in_gb=0.03,
    )
    return TextCrossEncoder(search_config.RERANK_MODEL)
