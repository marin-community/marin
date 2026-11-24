# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned dense-embedding constants shared by the embedding jobs and retrievers.

Kept dependency-free so the embedding runtime and the pinned retrieval runtime
can import it without the Marin workspace.
"""

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 512
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
