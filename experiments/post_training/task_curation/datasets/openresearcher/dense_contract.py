# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned dense-embedding constants shared by the embedding jobs and retrievers.

The encoder matches the synthesis environment's dense retriever family: full
documents, last-token pooling, empty passage prefix, normalized inner product.
Kept dependency-free so the embedding runtime and the pinned retrieval runtime
can import it without the Marin workspace.
"""

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
EMBEDDING_DIM = 4096
# Default cap for embedding one corpus document; the deployed value is fixed
# from the measured corpus token distribution and recorded in the artifact.
EMBEDDING_MAX_TOKENS = 8192
# The model card documents query-side instructions as
# "Instruct: {task}\nQuery:{query}" with left padding and last-token pooling;
# passages use an empty prefix.
QUERY_TASK_DESCRIPTION = "Given a web search query, retrieve relevant passages that answer the query"


def query_text(query: str) -> str:
    """Format one query with the model card's documented instruction."""
    return f"Instruct: {QUERY_TASK_DESCRIPTION}\nQuery:{query}"


MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
