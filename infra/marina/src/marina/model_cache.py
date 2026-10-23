# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Warm dependency caches without importing application source."""

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder


def main() -> None:
    # Released apps select their models. These entries only seed the image's disk cache.
    TextEmbedding("BAAI/bge-small-en-v1.5", threads=4)
    model = "Xenova/ms-marco-MiniLM-L-6-v2-int8"
    TextCrossEncoder.add_custom_model(
        model=model,
        sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-6-v2"),
        model_file="onnx/model_int8.onnx",
        description="CPU ONNX reranker.",
        license="apache-2.0",
        size_in_gb=0.03,
    )
    TextCrossEncoder(model, threads=4)


if __name__ == "__main__":
    main()
