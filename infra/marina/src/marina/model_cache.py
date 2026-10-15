# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Warm dependency caches without importing application source."""

from marina.models import shared_embedding, shared_reranker


def main() -> None:
    # Released apps select their models. These entries only seed the image's disk cache.
    shared_embedding("BAAI/bge-small-en-v1.5", 4)
    shared_reranker("Xenova/ms-marco-MiniLM-L-6-v2-int8", "Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model_int8.onnx", 4)


if __name__ == "__main__":
    main()
